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

Uso:
    python training/generador/dataset.py [--sin-guardrails]
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
from verificacion import es_ejemplo_limpio, frases_para_guardrails, parsear  # noqa: E402

CASOS = RAIZ / "data" / "generador" / "casos.jsonl"
RESPUESTAS_DIR = RAIZ / "data" / "generador" / "respuestas"
SALIDA_DIR = RAIZ / "data" / "generador"
CLASIFICADOR = "mrbert-biomed"
UMBRAL_RIESGO = 0.5
# Si un caso tiene respuestas de varios profesores, se usa la del más grande que haya pasado los filtros
PRIORIDAD_PROFESORES = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "Qwen/Qwen2.5-14B-Instruct-AWQ",
                        "Qwen/Qwen2.5-7B-Instruct"]


def prioridad(respuesta):
    p = respuesta["profesor"]
    return PRIORIDAD_PROFESORES.index(p) if p in PRIORIDAD_PROFESORES else len(PRIORIDAD_PROFESORES)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sin-guardrails", action="store_true", help="No filtrar con el clasificador (más rápido)")
    args = parser.parse_args()

    with open(CASOS, encoding="utf-8") as f:
        casos = {c["id"]: c for c in map(json.loads, f)}
    respuestas = []
    for archivo in sorted(RESPUESTAS_DIR.glob("*.jsonl")):
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

    if not args.sin_guardrails:
        from evaluar_guardrails import probabilidades_modelo
        frases = [frases_para_guardrails(r["respuesta"]) for r in limpias]
        planas = [o for fs in frases for o in fs]
        print(f"Revisando {len(planas)} oraciones con {CLASIFICADOR}...", flush=True)
        riesgo = iter(1 - p[0] for p in probabilidades_modelo(CLASIFICADOR, planas))
        aprobadas = []
        for r, fs in zip(limpias, frases):
            marcadas = [o for o in fs if next(riesgo) >= UMBRAL_RIESGO]
            if marcadas:
                motivos["frase riesgosa según el guardrail"] += 1
                r["frases_marcadas"] = marcadas
            else:
                aprobadas.append(r)
        limpias = aprobadas

    elegidas = {}
    for r in sorted(limpias, key=prioridad):
        if r["id"] in elegidas:
            motivos["caso repetido (gana otro profesor)"] += 1
        else:
            elegidas[r["id"]] = r
    limpias = list(elegidas.values())

    por_split = {"train": [], "val": []}
    for r in limpias:
        caso = casos[r["id"]]
        por_split[r["split"]].append({"id": r["id"], "profesor": r["profesor"],
                                      "messages": mensajes_estudiante(caso, caso["pasajes"], parsear(r["respuesta"]))})
    for split, filas in por_split.items():
        with open(SALIDA_DIR / f"sft_{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in filas)

    print(f"{len(respuestas)} respuestas; descartadas: {dict(motivos) or 'ninguna'}")
    for split, filas in por_split.items():
        print(f"  {split}: {len(filas)} ejemplos ({dict(Counter(x['profesor'] for x in filas))})")


if __name__ == "__main__":
    main()
