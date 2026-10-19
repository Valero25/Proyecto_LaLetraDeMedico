"""
Fase 4: genera las respuestas de un modelo local sobre los casos de test, para evaluarlas después.

Corre en Colab (T4) con el modelo base y con el adaptador LoRA:

    python training/generador/generar_respuestas.py                                   # base
    python training/generador/generar_respuestas.py --adaptador models/generador/qwen2.5-3b-instruct-lora

Las respuestas se guardan en data/generador/salidas/{nombre}.jsonl y se evalúan con
evaluacion/evaluar_generador.py. Decodificación voraz (sin muestreo) para que sea reproducible.
Reanudable: salta los casos ya respondidos.
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from prompt import mensajes_estudiante  # noqa: E402

CASOS = RAIZ / "data" / "generador" / "casos.jsonl"
SALIDAS = RAIZ / "data" / "generador" / "salidas"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adaptador", type=Path, help="Carpeta del adaptador LoRA (sin él, se evalúa el modelo base)")
    parser.add_argument("--nombre", help="Nombre del archivo de salida (por defecto, según modelo y adaptador)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--sin-cuantizar", action="store_true")
    parser.add_argument("--limite", type=int)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    gpu = torch.cuda.is_available()
    nombre = args.nombre or (args.modelo.split("/")[-1].lower() + ("-ajustado" if args.adaptador else "-base"))
    salida = SALIDAS / f"{nombre}.jsonl"
    salida.parent.mkdir(parents=True, exist_ok=True)
    hechos = set()
    if salida.exists():
        with open(salida, encoding="utf-8") as f:
            hechos = {json.loads(linea)["id"] for linea in f}
    with open(CASOS, encoding="utf-8") as f:
        casos = [c for c in map(json.loads, f) if c["split"] == args.split and c["id"] not in hechos][:args.limite]
    print(f"{nombre}: {len(hechos)} hechos, faltan {len(casos)}", flush=True)

    tokenizador = AutoTokenizer.from_pretrained(args.adaptador or args.modelo)
    tokenizador.padding_side = "left"  # para generar en lote, el relleno va a la izquierda
    cuantizacion = None if args.sin_cuantizar else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo, quantization_config=cuantizacion, dtype=torch.float16 if gpu else torch.float32,
        device_map="auto" if gpu else None)
    if args.adaptador:
        from peft import PeftModel
        modelo = PeftModel.from_pretrained(modelo, str(args.adaptador))
    modelo.eval()

    with open(salida, "a", encoding="utf-8") as f:
        for i in range(0, len(casos), args.batch):
            lote = casos[i:i + args.batch]
            textos = [tokenizador.apply_chat_template(mensajes_estudiante(c, c["pasajes"]), tokenize=False,
                                                      add_generation_prompt=True) for c in lote]
            entrada = tokenizador(textos, return_tensors="pt", padding=True).to(modelo.device)
            inicio = time.time()
            with torch.no_grad():
                generado = modelo.generate(**entrada, max_new_tokens=args.max_tokens, do_sample=False,
                                           pad_token_id=tokenizador.pad_token_id or tokenizador.eos_token_id)
            segundos = (time.time() - inicio) / len(lote)
            nuevos = generado[:, entrada["input_ids"].shape[1]:]
            for caso, tokens in zip(lote, nuevos):
                f.write(json.dumps({"id": caso["id"], "modelo": nombre,
                                    "respuesta": tokenizador.decode(tokens, skip_special_tokens=True),
                                    "tokens": int((tokens != tokenizador.pad_token_id).sum()),
                                    "segundos": round(segundos, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(i + args.batch, len(casos))}/{len(casos)} ({segundos:.0f} s por caso)", flush=True)


if __name__ == "__main__":
    main()
