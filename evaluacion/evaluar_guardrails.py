"""
Fase 3: compara el filtro de palabras actual con el clasificador entrenado sobre el set de test.

El test son frases de los 12 analitos que el clasificador no vio al entrenar (ver training/guardrails/dataset.py).

- Tarea binaria (lo que importa en la app): ¿la frase es riesgosa (diagnóstica o alarmista) o no?
  El filtro de palabras solo puede hacer esta tarea.
- Tarea de 3 clases: solo para el clasificador.

Uso:
    python evaluacion/evaluar_guardrails.py [--modelos palabras mrbert-biomed]
"""
import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "training" / "guardrails"))

from dataset import CLASES, cargar_dataset, resumen  # noqa: E402

SALIDA = RAIZ / "evaluacion" / "resultados" / "fase3_guardrails.json"
# Copia de la lista de step_4_validation en backend.py
PALABRAS_RIESGO = ["diagnóstico", "tienes", "padeces", "sufres de"]


def predecir_palabras(frases):
    """El filtro actual no distingue clases: marca la frase (1) o no (0)."""
    return [int(any(p in f.lower() for p in PALABRAS_RIESGO)) for f in frases]


def predecir_modelo(nombre, frases):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ruta = RAIZ / "models" / "guardrails" / nombre
    tokenizador = AutoTokenizer.from_pretrained(ruta)
    modelo = AutoModelForSequenceClassification.from_pretrained(ruta).eval()
    predicciones = []
    with torch.no_grad():
        for i in range(0, len(frases), 32):
            lote = tokenizador(frases[i:i + 32], truncation=True, max_length=96, padding=True, return_tensors="pt")
            predicciones += modelo(**lote).logits.argmax(-1).tolist()
    return predicciones


def metricas_binarias(y, p):
    from sklearn.metrics import f1_score, precision_score, recall_score
    return {"precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0),
            "falsos_positivos": sum(a == 0 and b == 1 for a, b in zip(y, p)),
            "falsos_negativos": sum(a == 1 and b == 0 for a, b in zip(y, p))}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelos", nargs="+", default=["palabras", "mrbert-biomed"])
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    from sklearn.metrics import classification_report, confusion_matrix

    dataset, motivos = cargar_dataset()
    print(resumen(dataset, motivos) + "\n")
    filas = [d for d in dataset if d["split"] == args.split]
    frases = [d["frase"] for d in filas]
    y3 = [d["etiqueta"] for d in filas]
    y2 = [int(e != 0) for e in y3]  # 0 = informativa

    resultados = {}
    for nombre in args.modelos:
        if nombre == "palabras":
            p2 = predecir_palabras(frases)
            r = {"binaria": metricas_binarias(y2, p2)}
        else:
            if not (RAIZ / "models" / "guardrails" / nombre / "config.json").exists():
                print(f"== {nombre}: se omite, no existe models/guardrails/{nombre}\n")
                continue
            p3 = predecir_modelo(nombre, frases)
            p2 = [int(e != 0) for e in p3]
            r = {"binaria": metricas_binarias(y2, p2),
                 "tres_clases": classification_report(y3, p3, target_names=CLASES, output_dict=True, zero_division=0),
                 "matriz_confusion": confusion_matrix(y3, p3, labels=range(len(CLASES))).tolist()}
            print(f"== {nombre}: 3 clases (filas = real, columnas = predicho)")
            print(classification_report(y3, p3, target_names=CLASES, digits=3, zero_division=0))
            print(f"{'':>14}" + "".join(f"{c:>14}" for c in CLASES))
            for c, fila in zip(CLASES, r["matriz_confusion"]):
                print(f"{c:>14}" + "".join(f"{v:>14}" for v in fila))
        b = r["binaria"]
        print(f"== {nombre}: riesgosa sí/no  precision {b['precision']:.3f}  recall {b['recall']:.3f}  "
              f"F1 {b['f1']:.3f}  FP {b['falsos_positivos']}  FN {b['falsos_negativos']}\n")
        r["errores"] = [{"frase": f, "real": CLASES[a], "riesgosa_predicha": bool(q)}
                        for f, a, q in zip(frases, y3, p2) if int(a != 0) != q]
        resultados[nombre] = r

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "n": len(filas), "resultados": resultados}, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
