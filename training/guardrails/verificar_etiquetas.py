"""
Fase 3: un segundo modelo (juez) vuelve a etiquetar cada frase sin ver la clase pedida.

Las etiquetas del profesor pueden fallar: se le pidió una frase "alarmista" y escribió una
informativa, por ejemplo. El juez es otro modelo (openai/gpt-oss-120b por defecto), así que sus
errores son independientes de los del profesor.

Resultado:
- data/guardrails/etiquetas_juez.jsonl: la clase que asigna el juez a cada frase.
- Acuerdo y kappa de Cohen entre profesor y juez (se reportan en el informe).
- data/guardrails/revisar_equipo.csv: los desacuerdos, para que el equipo decida la clase.
  El entrenamiento usa solo las frases en que ambos coinciden, más las que el equipo revise.

Reanudable: salta las frases ya etiquetadas.

Uso:
    python training/guardrails/verificar_etiquetas.py [--modelo openai/gpt-oss-120b]
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "training"))
sys.path.insert(0, str(Path(__file__).parent))

from groq_cliente import crear_sesion, pedir_json  # noqa: E402
from generar_frases import CLASES, DEFINICIONES, SALIDA as FRASES_PATH  # noqa: E402

SALIDA = RAIZ / "data" / "guardrails" / "etiquetas_juez.jsonl"
REVISAR = RAIZ / "data" / "guardrails" / "revisar_equipo.csv"
POR_PETICION = 25

PROMPT = """Clasifica frases que un asistente de IA escribió a pacientes para explicarles sus exámenes de laboratorio. El asistente nunca debe diagnosticar ni alarmar.

Clases:
{definiciones}

Si una frase es a la vez diagnóstica y alarmista, elige la que más pese. Juzga solo el texto de la frase.

Frases:
{frases}

Responde solo con JSON: {{"etiquetas": [{{"i": 1, "clase": "informativa"}}, ...]}} con una entrada por frase."""


def kappa_cohen(a, b):
    n = len(a)
    observado = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    esperado = sum(ca[c] * cb[c] for c in set(a) | set(b)) / n ** 2
    return (observado - esperado) / (1 - esperado)


def etiquetar(sesion, modelo, frases):
    lista = "\n".join(f"{i}. {f}" for i, f in enumerate(frases, 1))
    respuesta = pedir_json(sesion, modelo, PROMPT.format(definiciones=DEFINICIONES, frases=lista),
                           temperatura=0, etiqueta="juez", reasoning_effort="low")
    etiquetas = {}
    for e in respuesta.get("etiquetas", []):
        if isinstance(e, dict) and e.get("clase") in CLASES and isinstance(e.get("i"), int) and 1 <= e["i"] <= len(frases):
            etiquetas[frases[e["i"] - 1]] = e["clase"]
    return etiquetas


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="openai/gpt-oss-120b")
    args = parser.parse_args()

    with open(FRASES_PATH, encoding="utf-8") as f:
        filas = [json.loads(linea) for linea in f]
    juez = {}
    if SALIDA.exists():
        with open(SALIDA, encoding="utf-8") as f:
            juez = {d["frase"]: d["clase_juez"] for d in map(json.loads, f)}

    # Mezcladas, para que cada petición tenga frases de varias clases y analitos
    pendientes = [d["frase"] for d in filas if d["frase"] not in juez]
    random.Random(13).shuffle(pendientes)
    print(f"{len(juez)} frases ya etiquetadas; faltan {len(pendientes)}", flush=True)

    sesion = crear_sesion()
    with open(SALIDA, "a", encoding="utf-8") as f:
        for inicio in range(0, len(pendientes), POR_PETICION):
            lote = pendientes[inicio:inicio + POR_PETICION]
            # Las frases que el juez omita quedan pendientes para la próxima corrida
            for frase, clase in etiquetar(sesion, args.modelo, lote).items():
                juez[frase] = clase
                f.write(json.dumps({"frase": frase, "clase_juez": clase, "juez": args.modelo}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(inicio + POR_PETICION, len(pendientes))}/{len(pendientes)}", flush=True)

    pares = [(d["clase"], juez[d["frase"]]) for d in filas if d["frase"] in juez]
    profesor, jueces = zip(*pares)
    acuerdo = sum(a == b for a, b in pares) / len(pares)
    print(f"\n{len(pares)} frases con ambas etiquetas: acuerdo {acuerdo:.1%}, kappa de Cohen {kappa_cohen(profesor, jueces):.3f}")
    print("Matriz (filas = profesor, columnas = juez):")
    print(f"{'':>14}" + "".join(f"{c:>14}" for c in CLASES))
    conteo = Counter(pares)
    for c in CLASES:
        print(f"{c:>14}" + "".join(f"{conteo[(c, j)]:>14}" for j in CLASES))

    desacuerdos = [d for d in filas if d["frase"] in juez and juez[d["frase"]] != d["clase"]]
    with open(REVISAR, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frase", "clase_profesor", "clase_juez", "clase_equipo"])
        for d in desacuerdos:
            w.writerow([d["frase"], d["clase"], juez[d["frase"]], ""])
    print(f"\n{len(desacuerdos)} desacuerdos en {REVISAR.relative_to(RAIZ)}: llenar la columna clase_equipo "
          "(informativa, diagnostica, alarmista o descartar)")


if __name__ == "__main__":
    main()
