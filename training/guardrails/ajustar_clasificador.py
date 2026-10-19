"""
Fase 3: fine-tuning de MrBERT-biomed como clasificador de frases (informativa / diagnóstica / alarmista).

Se agrega una cabeza de clasificación sobre BSC-LT/MrBERT-biomed (ModernBERT, 308M parámetros) y se
entrena el modelo completo. Se elige la época con mejor F1 macro en validación; el test (analitos no
vistos) se evalúa aparte con evaluacion/evaluar_guardrails.py.

Con --pesos, la entropía cruzada pondera cada clase (ver sección 7 de informe/fundamentos.md) y el
modelo se guarda con el sufijo "-ponderado".

Uso:
    python training/guardrails/ajustar_clasificador.py [--modelo mrbert-biomed] [--epocas 4] [--pesos 1,2,2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from dataset import CLASES, cargar_dataset, parsear_pesos, resumen  # noqa: E402

SALIDA_DIR = RAIZ / "models" / "guardrails"
MODELOS = {"mrbert-biomed": "BSC-LT/MrBERT-biomed", "mrbert-es": "BSC-LT/MrBERT-es"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", choices=MODELOS, default="mrbert-biomed")
    parser.add_argument("--epocas", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--pesos", help="Pesos de la entropía cruzada por clase, ej. 1,2,2 (por defecto, sin ponderar)")
    parser.add_argument("--limite", type=int, help="Usar solo N frases de train (prueba rápida)")
    args = parser.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    dataset, motivos = cargar_dataset()
    print(resumen(dataset, motivos), flush=True)
    train = [d for d in dataset if d["split"] == "train"][:args.limite]
    val = [d for d in dataset if d["split"] == "val"]

    nombre = MODELOS[args.modelo]
    tokenizador = AutoTokenizer.from_pretrained(nombre)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        nombre, num_labels=len(CLASES),
        id2label=dict(enumerate(CLASES)), label2id={c: i for i, c in enumerate(CLASES)})

    def a_dataset(filas):
        ds = Dataset.from_dict({"text": [d["frase"] for d in filas], "label": [d["etiqueta"] for d in filas]})
        return ds.map(lambda b: tokenizador(b["text"], truncation=True, max_length=args.max_len), batched=True)

    def metricas(pred):
        y = pred.label_ids
        p = np.argmax(pred.predictions, axis=-1)
        return {"accuracy": accuracy_score(y, p), "f1_macro": f1_score(y, p, average="macro")}

    pesos = parsear_pesos(args.pesos) if args.pesos else None
    salida = SALIDA_DIR / (args.modelo + ("-ponderado" if pesos else ""))
    argumentos = TrainingArguments(
        output_dir=str(salida / "checkpoints"),
        num_train_epochs=args.epocas,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=0.1,  # En transformers 5 un float es la proporción de pasos de calentamiento
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        report_to="none",
        seed=13,
    )
    class TrainerPonderado(Trainer):
        """Igual que Trainer, pero con entropía cruzada ponderada por clase."""
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            etiquetas = inputs.pop("labels")
            salidas = model(**inputs)
            perdida = torch.nn.functional.cross_entropy(
                salidas.logits, etiquetas, weight=torch.tensor(pesos, device=salidas.logits.device))
            return (perdida, salidas) if return_outputs else perdida

    entrenador = (TrainerPonderado if pesos else Trainer)(
        model=modelo,
        args=argumentos,
        train_dataset=a_dataset(train),
        eval_dataset=a_dataset(val),
        processing_class=tokenizador,
        data_collator=DataCollatorWithPadding(tokenizador),
        compute_metrics=metricas,
    )
    inicio = time.time()
    entrenador.train()
    minutos = (time.time() - inicio) / 60
    final = entrenador.evaluate()

    entrenador.save_model(str(salida))
    tokenizador.save_pretrained(str(salida))
    with open(salida / "entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump({"modelo_base": nombre, "frases_train": len(train), "frases_val": len(val),
                   "epocas": args.epocas, "pesos_clase": pesos, "lr": args.lr, "batch": args.batch, "max_len": args.max_len,
                   "minutos": round(minutos, 1), "val_mejor_epoca": final,
                   "historial": entrenador.state.log_history}, f, indent=2, ensure_ascii=False)
    print(f"Validación (mejor época): F1 macro {final['eval_f1_macro']:.3f}, accuracy {final['eval_accuracy']:.3f}")
    print(f"Modelo guardado en {salida.relative_to(RAIZ)} ({minutos:.1f} min)")


if __name__ == "__main__":
    main()
