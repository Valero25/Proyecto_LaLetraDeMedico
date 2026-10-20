"""
Fase 3 (v2): muestra de oraciones **reales** (escritas por los generadores de la Fase 4) para medir el
guardrail fuera del texto sintético y reentrenarlo con ese tipo de texto.

La muestra tiene dos mitades:
- todas las oraciones que el guardrail v1 marcó como riesgosas (data/generador/descartadas_guardrail.jsonl), y
- otras tantas que no marcó, tomadas al azar de las respuestas que aprobó.

El juez (gpt-oss-120b, el mismo de verificar_etiquetas.py) etiqueta cada oración sin ver la
predicción del guardrail. El equipo puede corregirlas en la columna clase_equipo del CSV.
Cada oración guarda su analito, para respetar el split por analito de la Fase 2.

Salida: data/guardrails/reales.csv con frase, analito, profesor, riesgo_v1, marcada_v1, clase_juez, clase_equipo.
Reanudable: si el CSV existe, solo etiqueta las filas sin clase_juez.

Uso:
    python training/guardrails/muestra_reales.py
"""
import csv
import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "training"))
sys.path.insert(0, str(RAIZ / "training" / "generador"))
sys.path.insert(0, str(Path(__file__).parent))

from verificacion import _sin_tildes, dividir_oraciones, es_ejemplo_limpio, parsear  # noqa: E402

GENERADOR = RAIZ / "data" / "generador"
SALIDA = RAIZ / "data" / "guardrails" / "reales.csv"
COLUMNAS = ["frase", "analito", "profesor", "riesgo_v1", "marcada_v1", "clase_juez", "clase_equipo"]


def oraciones_con_analito(respuesta, caso):
    """(oración, clave canónica del analito) para cada oración de cada explicación."""
    canonico = {_sin_tildes(a["examen"]): a["examen_canonico"] for a in caso["analitos"]}
    datos = parsear(respuesta) or {"explicaciones": []}
    return [(o, canonico.get(_sin_tildes(e["examen"])))
            for e in datos["explicaciones"] for o in dividir_oraciones(e["explicacion"])]


def construir_muestra(seed=13):
    casos = {c["id"]: c for c in map(json.loads, open(GENERADOR / "casos.jsonl", encoding="utf-8"))}
    respuestas = {}
    for archivo in sorted((GENERADOR / "respuestas").glob("*.jsonl")):
        for r in map(json.loads, open(archivo, encoding="utf-8")):
            respuestas[(r["id"], r["profesor"])] = r
    descartadas = [json.loads(l) for l in open(GENERADOR / "descartadas_guardrail.jsonl", encoding="utf-8")]
    marcadas_por_resp = {(d["id"], d["profesor"]): dict(d["frases_marcadas"]) for d in descartadas}

    marcadas, aprobadas = [], []
    for clave, r in respuestas.items():
        if not es_ejemplo_limpio(r["verificacion"]):
            continue  # el guardrail v1 solo revisó las respuestas limpias
        riesgos = marcadas_por_resp.get(clave)
        for frase, analito in oraciones_con_analito(r["respuesta"], casos[r["id"]]):
            if analito is None:
                continue
            fila = {"frase": frase, "analito": analito, "profesor": r["profesor"], "clase_juez": "", "clase_equipo": ""}
            if riesgos is None:
                aprobadas.append({**fila, "riesgo_v1": "", "marcada_v1": 0})
            elif frase in riesgos:
                marcadas.append({**fila, "riesgo_v1": riesgos[frase], "marcada_v1": 1})
            # Las demás oraciones de una respuesta descartada no se usan: no se guardó su probabilidad

    vistas, unicas = set(), {"m": [], "a": []}
    for tipo, filas in (("m", marcadas), ("a", aprobadas)):
        for f in filas:
            if f["frase"].lower() not in vistas:
                vistas.add(f["frase"].lower())
                unicas[tipo].append(f)
    random.Random(seed).shuffle(unicas["a"])
    muestra = unicas["m"] + unicas["a"][:len(unicas["m"])]
    random.Random(seed).shuffle(muestra)
    return muestra


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if SALIDA.exists():
        with open(SALIDA, encoding="utf-8-sig") as f:
            filas = list(csv.DictReader(f))
    else:
        filas = construir_muestra()
    print(f"{len(filas)} oraciones reales ({sum(int(f['marcada_v1']) for f in filas)} marcadas por el guardrail v1)", flush=True)

    from groq_cliente import crear_sesion
    from verificar_etiquetas import POR_PETICION, etiquetar

    sesion = crear_sesion()
    pendientes = [f for f in filas if not f["clase_juez"]]
    for inicio in range(0, len(pendientes), POR_PETICION):
        lote = pendientes[inicio:inicio + POR_PETICION]
        etiquetas = etiquetar(sesion, "openai/gpt-oss-120b", [f["frase"] for f in lote])
        for f in lote:
            f["clase_juez"] = etiquetas.get(f["frase"], "")
        with open(SALIDA, "w", encoding="utf-8-sig", newline="") as fh:  # se guarda tras cada lote
            w = csv.DictWriter(fh, fieldnames=COLUMNAS)
            w.writeheader()
            w.writerows(filas)
        print(f"  {min(inicio + POR_PETICION, len(pendientes))}/{len(pendientes)}", flush=True)
    if not pendientes:
        with open(SALIDA, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNAS)
            w.writeheader()
            w.writerows(filas)

    con_juez = [f for f in filas if f["clase_juez"]]
    for marcada in ("1", "0"):
        grupo = [f for f in con_juez if str(f["marcada_v1"]) == marcada]
        riesgosas = sum(f["clase_juez"] != "informativa" for f in grupo)
        print(f"  {'marcadas' if marcada == '1' else 'no marcadas'} por v1: {len(grupo)}; el juez considera riesgosas {riesgosas}"
              + (f" ({riesgosas / len(grupo):.0%})" if grupo else ""))
    print(f"Guardado en {SALIDA.relative_to(RAIZ)}: el equipo puede corregir la columna clase_equipo")


if __name__ == "__main__":
    main()
