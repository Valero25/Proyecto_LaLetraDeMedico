"""
Fase 3: arma el dataset final del clasificador a partir de las frases, el juez y la revisión del equipo.

Etiqueta de cada frase:
1. La del equipo, si la frase está en revisar_equipo.csv con clase_equipo llena ("descartar" la quita).
2. Si no, la del profesor cuando el juez coincide.
3. Si profesor y juez no coinciden y nadie la revisó, la frase queda fuera.

Split por analito, igual que en la Fase 2: las frases de los 12 analitos de test de
training/rag/split_analitos.json van a test, así se mide si el clasificador generaliza a exámenes
que no vio. Del resto, un 15 % va a validación.
"""
import csv
import json
import random
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
FRASES_PATH = RAIZ / "data" / "guardrails" / "frases.jsonl"
JUEZ_PATH = RAIZ / "data" / "guardrails" / "etiquetas_juez.jsonl"
REVISAR_PATH = RAIZ / "data" / "guardrails" / "revisar_equipo.csv"
SPLIT_PATH = RAIZ / "training" / "rag" / "split_analitos.json"
CLASES = ["informativa", "diagnostica", "alarmista"]
PROPORCION_VAL = 0.15


def parsear_pesos(texto):
    """"1,2,2" -> [1.0, 2.0, 2.0], en el orden de CLASES."""
    pesos = [float(x) for x in texto.split(",")]
    if len(pesos) != len(CLASES) or min(pesos) <= 0:
        raise ValueError(f"Se esperan {len(CLASES)} pesos positivos ({', '.join(CLASES)}): {texto}")
    return pesos


def _leer_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f]


def cargar_dataset(frases_path=FRASES_PATH, juez_path=JUEZ_PATH, revisar_path=REVISAR_PATH, seed=13):
    filas = _leer_jsonl(frases_path)
    juez = {d["frase"]: d["clase_juez"] for d in _leer_jsonl(juez_path)}
    equipo = {}
    if revisar_path.exists():
        with open(revisar_path, encoding="utf-8-sig") as f:
            equipo = {r["frase"]: r["clase_equipo"].strip().lower() for r in csv.DictReader(f) if r["clase_equipo"].strip()}
    test = set(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["test"])

    rng = random.Random(seed)
    dataset, motivos = [], Counter()
    for d in filas:
        if d["frase"] in equipo:
            clase, origen = equipo[d["frase"]], "equipo"
            if clase not in CLASES:
                motivos["descartada por el equipo"] += 1
                continue
        elif juez.get(d["frase"]) == d["clase"]:
            clase, origen = d["clase"], "acuerdo"
        else:
            motivos["sin juez" if d["frase"] not in juez else "desacuerdo sin revisar"] += 1
            continue
        if d["analito"] in test:
            split = "test"
        else:
            split = "val" if rng.random() < PROPORCION_VAL else "train"
        dataset.append({"frase": d["frase"], "clase": clase, "etiqueta": CLASES.index(clase),
                        "analito": d["analito"], "split": split, "origen": origen})
    return dataset, motivos


def resumen(dataset, motivos):
    lineas = [f"{len(dataset)} frases utilizables; fuera: {dict(motivos) or 'ninguna'}"]
    for split in ("train", "val", "test"):
        conteo = Counter(d["clase"] for d in dataset if d["split"] == split)
        lineas.append(f"  {split:<6}{sum(conteo.values()):>5}  " + ", ".join(f"{c}={conteo[c]}" for c in CLASES))
    return "\n".join(lineas)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(resumen(*cargar_dataset()))
