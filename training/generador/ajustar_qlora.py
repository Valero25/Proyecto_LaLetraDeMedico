"""
Fase 4: fine-tuning QLoRA del generador de explicaciones.

- El modelo base queda congelado y cuantizado a 4 bits (NF4). Solo se entrenan las matrices A y B de
  LoRA en todas las capas lineales de atención y MLP: cada W de d × k pasa a W + (alpha/r)·BA.
- La pérdida es la entropía cruzada del siguiente token **solo sobre la respuesta** (el JSON del
  asistente): el dataset va en formato prompt/completion y TRL enmascara el prompt.
- Guarda el conteo de parámetros entrenables por tipo de capa y el historial de pérdidas de train y
  validación, que usa informe/fundamentos.md (sección 10).
- Si existe un checkpoint en --checkpoints, reanuda desde ahí (útil cuando Colab se desconecta).

Uso en Colab (T4):
    python training/generador/ajustar_qlora.py --checkpoints /content/drive/MyDrive/letrademedico/checkpoints
Prueba en CPU con un modelo diminuto, sin cuantizar:
    python training/generador/ajustar_qlora.py --modelo HuggingFaceTB/SmolLM2-135M-Instruct --sin-cuantizar --limite 8
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data" / "generador"
SALIDA_DIR = RAIZ / "models" / "generador"
CAPAS_LORA = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def cargar_split(ruta, limite=None):
    """Convierte {"messages": [system, user, assistant]} al formato prompt/completion de TRL."""
    with open(ruta, encoding="utf-8") as f:
        filas = [json.loads(linea) for linea in f][:limite]
    return [{"prompt": x["messages"][:-1], "completion": x["messages"][-1:]} for x in filas]


def conteo_lora(modelo):
    """Parámetros de LoRA por tipo de capa: r·(d + k) por cada matriz W de d × k."""
    por_capa = defaultdict(lambda: {"matrices": 0, "lora": 0, "base": 0})
    for nombre, modulo in modelo.named_modules():
        if hasattr(modulo, "lora_A") and "default" in getattr(modulo, "lora_A", {}):
            a, b = modulo.lora_A["default"].weight, modulo.lora_B["default"].weight  # (r, k) y (d, r)
            tipo = nombre.rsplit(".", 1)[-1]
            por_capa[tipo]["matrices"] += 1
            por_capa[tipo]["lora"] += a.numel() + b.numel()
            por_capa[tipo]["base"] += b.shape[0] * a.shape[1]
            por_capa[tipo]["forma"] = f"{b.shape[0]} x {a.shape[1]}, r = {a.shape[0]}"
    return dict(por_capa)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epocas", type=float, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--acumulacion", type=int, default=8, help="Pasos de acumulación: lote efectivo = batch x acumulación")
    parser.add_argument("--max-len", type=int, default=3072)
    parser.add_argument("--sin-cuantizar", action="store_true", help="Sin 4 bits (para probar en CPU)")
    parser.add_argument("--limite", type=int, help="Usar solo N ejemplos (prueba rápida)")
    parser.add_argument("--checkpoints", type=Path, help="Carpeta de checkpoints (ej. en Google Drive)")
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    gpu = torch.cuda.is_available()
    nombre_corto = args.modelo.split("/")[-1].lower()
    salida = args.salida or SALIDA_DIR / f"{nombre_corto}-lora"
    checkpoints = args.checkpoints or salida / "checkpoints"

    train = Dataset.from_list(cargar_split(DATOS / "sft_train.jsonl", args.limite))
    val = Dataset.from_list(cargar_split(DATOS / "sft_val.jsonl", args.limite))
    print(f"{len(train)} ejemplos de train, {len(val)} de validación", flush=True)

    tokenizador = AutoTokenizer.from_pretrained(args.modelo)
    cuantizacion = None if args.sin_cuantizar else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16)  # la T4 no soporta bfloat16
    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo, quantization_config=cuantizacion, dtype=torch.float16 if gpu else torch.float32,
        device_map="auto" if gpu else None)

    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=CAPAS_LORA, task_type="CAUSAL_LM")
    config = SFTConfig(
        output_dir=str(checkpoints),
        num_train_epochs=args.epocas,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.acumulacion,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # En transformers 5 un float es la proporción de pasos de calentamiento
        max_length=args.max_len,
        completion_only_loss=True,  # pérdida solo sobre la respuesta del asistente
        gradient_checkpointing=gpu,
        fp16=gpu,
        bf16=False,  # TRL lo activa por defecto, pero la T4 (y la CPU) no lo soportan
        eval_strategy="steps",
        eval_steps=25,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        report_to="none",
        seed=13,
    )
    entrenador = SFTTrainer(model=modelo, args=config, train_dataset=train, eval_dataset=val,
                            processing_class=tokenizador, peft_config=lora)

    total = sum(p.numel() for p in entrenador.model.parameters())
    entrenables = sum(p.numel() for p in entrenador.model.parameters() if p.requires_grad)
    capas = conteo_lora(entrenador.model)
    print(f"Parámetros entrenables: {entrenables:,} de {total:,} ({entrenables / total:.2%})", flush=True)
    for tipo, c in capas.items():
        print(f"  {tipo:<10} {c['matrices']:>3} matrices de {c['forma']}: {c['lora']:,} de LoRA frente a {c['base']:,}", flush=True)

    hay_checkpoint = checkpoints.exists() and any(checkpoints.glob("checkpoint-*"))
    inicio = time.time()
    entrenador.train(resume_from_checkpoint=hay_checkpoint or None)
    minutos = (time.time() - inicio) / 60

    entrenador.save_model(str(salida))
    tokenizador.save_pretrained(str(salida))
    historial = entrenador.state.log_history
    with open(salida / "entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump({"modelo_base": args.modelo, "cuantizado_4bit": not args.sin_cuantizar,
                   "lora": {"r": args.r, "alpha": args.alpha, "dropout": args.dropout, "capas": CAPAS_LORA},
                   "parametros_totales": total, "parametros_entrenables": entrenables, "conteo_por_capa": capas,
                   "ejemplos_train": len(train), "ejemplos_val": len(val), "epocas": args.epocas, "lr": args.lr,
                   "lote_efectivo": args.batch * args.acumulacion, "max_len": args.max_len, "minutos": round(minutos, 1),
                   "perdida_train": [{"paso": h["step"], "perdida": h["loss"]} for h in historial if "loss" in h],
                   "perdida_val": [{"paso": h["step"], "perdida": h["eval_loss"]} for h in historial if "eval_loss" in h]},
                  f, indent=2, ensure_ascii=False)
    print(f"Adaptador LoRA guardado en {salida} ({minutos:.1f} min)")


if __name__ == "__main__":
    main()
