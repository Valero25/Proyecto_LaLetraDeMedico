"""
Fase 4: arma el dataset SFT a partir de las respuestas de los profesores.

Una respuesta entra al entrenamiento si:
1. pasa las verificaciones de verificacion.es_ejemplo_limpio (JSON válido, un examen por analito,
   estado consistente, 2 a 5 preguntas), y
2. ninguna de sus oraciones es marcada como riesgosa por el clasificador de la Fase 3
   (MrBERT-biomed, P(diagnóstica) + P(alarmista) >= UMBRAL_RIESGO).

Es el mismo clasificador que luego evalúa al modelo ajustado: así el estudiante aprende solo de
ejemplos que el propio guardrail del proyecto aprueba.

Salida (formato chat, para el SFTTrainer de TRL):
    data/generador/sft_train.jsonl y sft_val.jsonl, con {"id", "profesor", "messages": [...]}

Versión 2 (`--version v2` → sft_train_v2.jsonl y sft_val_v2.jsonl; el v1 no se toca). Además de lo anterior:
3. menciona el valor del paciente en todos los exámenes fuera de rango (el v1 heredó un 70,7 % del profesor 14B);
4. sin oraciones de causas repetidas entre exámenes, sin siglas con un significado inventado y sin "usted"
   (verificacion.calidad), errores vistos con PDFs reales;
5. las preguntas se limpian con la misma revisión de coherencia de la app y quedan al menos 2.
Medido el 2026-11-02: quedan 421 de train y 93 de val (v1: 555 y 97); el 14B baja del 49 % al 35 % del train.

Versión 3 (`--version v3` → sft_train_v3.jsonl y sft_val_v3.jsonl). Lee casos_v3.jsonl (pasajes de la app, según
el estado) y data/generador/respuestas_v3/ (profesor_local.py --version v3, varias muestras por caso). Además del v2:
6. ninguna causa sin respaldo en el pasaje (verificacion.causas_sin_respaldo). Con los pasajes viejos, el 46-56 % de
   las causas de los profesores y el 59-65 % de las del v1 y v2 no estaban en el pasaje;
7. de las muestras aprobadas de un caso se elige la que más causas respaldadas tiene.

Uso:
    python training/generador/dataset.py [--sin-guardrails] [--version v2|v3]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(RAIZ / "evaluacion"))
sys.path.insert(0, str(RAIZ / "training" / "guardrails"))

from prompt import mensajes_estudiante  # noqa: E402
from verificacion import calidad, es_ejemplo_limpio, frases_para_guardrails, parsear, verificar  # noqa: E402

CASOS = RAIZ / "data" / "generador" / "casos.jsonl"
RESPUESTAS_DIR = RAIZ / "data" / "generador" / "respuestas"
CASOS_V3 = RAIZ / "data" / "generador" / "casos_v3.jsonl"
RESPUESTAS_V3_DIR = RAIZ / "data" / "generador" / "respuestas_v3"
SALIDA_DIR = RAIZ / "data" / "generador"
CLASIFICADOR = "mrbert-biomed"
UMBRAL_RIESGO = 0.5
# Si un caso tiene respuestas de varios profesores, se usa la del más grande que haya pasado los filtros
PRIORIDAD_PROFESORES = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "Qwen/Qwen2.5-14B-Instruct-AWQ",
                        "Qwen/Qwen2.5-7B-Instruct"]


def clave(respuesta):
    """Una respuesta se identifica por caso, profesor y muestra (el profesor v3 escribe varias por caso)."""
    return respuesta["id"], respuesta["profesor"], respuesta.get("muestra", 0)


def prioridad(respuesta):
    p = respuesta["profesor"]
    return PRIORIDAD_PROFESORES.index(p) if p in PRIORIDAD_PROFESORES else len(PRIORIDAD_PROFESORES)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sin-guardrails", action="store_true", help="No filtrar con el clasificador (más rápido)")
    parser.add_argument("--version", default="v1", choices=["v1", "v2", "v3"])
    args = parser.parse_args()
    sufijo = "" if args.version == "v1" else f"_{args.version}"
    v3 = args.version == "v3"

    with open(CASOS_V3 if v3 else CASOS, encoding="utf-8") as f:
        casos = {c["id"]: c for c in map(json.loads, f)}
    respuestas = []
    for archivo in sorted((RESPUESTAS_V3_DIR if v3 else RESPUESTAS_DIR).glob("*.jsonl")):
        with open(archivo, encoding="utf-8") as f:
            respuestas += [json.loads(linea) for linea in f]

    motivos = Counter()
    limpias = []
    for r in respuestas:
        if es_ejemplo_limpio(r["verificacion"]):
            limpias.append(r)
        else:
            v = r["verificacion"]
            motivo = ("JSON inválido" if not v.get("json_valido") else "falta un examen" if v["cobertura"] < 1
                      else "estado inconsistente" if v["estado_consistente"] < 1 else "otro (preguntas o exámenes extra)")
            motivos[motivo] += 1

    preguntas_limpias = {}
    respaldadas = {}
    if args.version in ("v2", "v3"):
        aprobadas = []
        for r in limpias:
            caso = casos[r["id"]]
            v = verificar(r["respuesta"], caso)
            q = calidad(r["respuesta"], caso)
            if not all(d["menciona_valor"] for d, a in zip(v["detalle"], caso["analitos"]) if a["estado"] != "NORMAL"):
                motivos["v2: no menciona el valor de un examen fuera de rango"] += 1
            elif q["causas_repetidas"]:
                motivos["v2: causas repetidas entre exámenes"] += 1
            elif q["siglas_inventadas"]:
                motivos["v2: sigla con un significado inventado"] += 1
            elif q["usa_usted"]:
                motivos["v2: usa usted en vez de tú"] += 1
            elif len(q["preguntas_validas"]) < 2:
                motivos["v2: menos de 2 preguntas coherentes"] += 1
            elif v3 and q["causas_sin_respaldo"]:
                motivos["v3: una causa que no está en el pasaje"] += 1
            else:
                preguntas_limpias[clave(r)] = q["preguntas_validas"]
                respaldadas[clave(r)] = q["causas_evaluadas"]
                aprobadas.append(r)
        limpias = aprobadas

    if not args.sin_guardrails:
        from evaluar_guardrails import probabilidades_modelo
        frases = [frases_para_guardrails(r["respuesta"]) for r in limpias]
        planas = [o for fs in frases for o in fs]
        print(f"Revisando {len(planas)} oraciones con {CLASIFICADOR}...", flush=True)
        riesgo = iter(1 - p[0] for p in probabilidades_modelo(CLASIFICADOR, planas))
        aprobadas, descartadas = [], []
        for r, fs in zip(limpias, frases):
            marcadas = [(o, round(q, 3)) for o, q in ((o, next(riesgo)) for o in fs) if q >= UMBRAL_RIESGO]
            if marcadas:
                motivos["frase riesgosa según el guardrail"] += 1
                descartadas.append({"id": r["id"], "profesor": r["profesor"], "frases_marcadas": marcadas})
            else:
                aprobadas.append(r)
        limpias = aprobadas
        # Para revisar si el guardrail marca de más en texto real (no sintético)
        with open(SALIDA_DIR / f"descartadas_guardrail{sufijo if v3 else ''}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in descartadas)

    elegidas = {}
    # v3: entre las muestras de un mismo caso, la que más causas respaldadas en el pasaje tiene
    for r in sorted(limpias, key=lambda r: (prioridad(r), -respaldadas.get(clave(r), 0), r.get("muestra", 0))):
        if r["id"] in elegidas:
            motivos["caso repetido (gana otro profesor u otra muestra)"] += 1
        else:
            elegidas[r["id"]] = r
    limpias = list(elegidas.values())

    por_split = {"train": [], "val": []}
    for r in limpias:
        caso = casos[r["id"]]
        respuesta = parsear(r["respuesta"])
        if clave(r) in preguntas_limpias:
            # v2 y v3: el estudiante aprende solo las preguntas que la app dejaría pasar
            respuesta = {**respuesta, "preguntas_sugeridas": preguntas_limpias[clave(r)]}
        por_split[r["split"]].append({"id": r["id"], "profesor": r["profesor"],
                                      "messages": mensajes_estudiante(caso, caso["pasajes"], respuesta)})
    for split, filas in por_split.items():
        with open(SALIDA_DIR / f"sft_{split}{sufijo}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in filas)

    print(f"{len(respuestas)} respuestas; descartadas: {dict(motivos) or 'ninguna'}")
    for split, filas in por_split.items():
        print(f"  {split}: {len(filas)} ejemplos ({dict(Counter(x['profesor'] for x in filas))})")


if __name__ == "__main__":
    main()
