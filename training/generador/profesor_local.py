"""
Fase 4: un profesor propio en GPU (Colab o Kaggle) que escribe respuestas de referencia sin cuotas de API.

Usa el mismo prompt y las mismas verificaciones que generar_sft.py, así que sus respuestas entran al
mismo dataset. Recorre los casos de train **desde el final** de la lista, mientras los profesores
de Groq avanzan desde el principio: casi no se pisan. Si un caso queda con dos respuestas,
dataset.py se queda con la del profesor más grande.

Dos motores:
- vllm: procesa muchos casos a la vez (rápido). Requiere una versión de vLLM compatible con la GPU.
- transformers: generación por lotes con el modelo en 4 bits. Más lento, pero funciona en cualquier GPU.

Uso en Colab:
    python training/generador/profesor_local.py --motor vllm --salida /content/drive/MyDrive/letrademedico/respuestas
    python training/generador/profesor_local.py --motor transformers --modelo Qwen/Qwen2.5-7B-Instruct --salida ...

Reanudable: salta los casos que ya están en su archivo de salida.
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from prompt import prompt_profesor  # noqa: E402
from verificacion import es_ejemplo_limpio, parsear, verificar  # noqa: E402

CASOS = RAIZ / "data" / "generador" / "casos.jsonl"
RESPUESTAS_DIR = RAIZ / "data" / "generador" / "respuestas"


def ids_ya_respondidos(carpetas):
    hechos = set()
    for carpeta in carpetas:
        for archivo in Path(carpeta).glob("*.jsonl"):
            with open(archivo, encoding="utf-8") as f:
                hechos |= {json.loads(linea)["id"] for linea in f}
    return hechos


def normalizar(texto):
    """Si la respuesta es JSON (aunque venga entre ```json), se guarda compacta como las de Groq."""
    datos = parsear(texto)
    return json.dumps(datos, ensure_ascii=False) if datos is not None else texto


def generador_vllm(modelo, max_tokens, max_len):
    from vllm import LLM, SamplingParams
    llm = LLM(model=modelo, max_model_len=max_len, gpu_memory_utilization=0.92, dtype="half", seed=13)
    muestreo = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens, seed=13)

    def generar(prompts):
        salidas = llm.chat([[{"role": "user", "content": p}] for p in prompts], muestreo, use_tqdm=False)
        return [s.outputs[0].text for s in salidas]
    return generar


def generador_transformers(modelo, max_tokens, max_len):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizador = AutoTokenizer.from_pretrained(modelo)
    tokenizador.padding_side = "left"
    cuantizado = "awq" in modelo.lower() or not torch.cuda.is_available()  # sin GPU no hay 4 bits
    llm = AutoModelForCausalLM.from_pretrained(
        modelo, device_map="auto" if torch.cuda.is_available() else None,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=None if cuantizado else BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
    torch.manual_seed(13)

    def generar(prompts):
        textos = [tokenizador.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                  add_generation_prompt=True) for p in prompts]
        entrada = tokenizador(textos, return_tensors="pt", padding=True, truncation=True,
                              max_length=max_len - max_tokens).to(llm.device)
        with torch.no_grad():
            salida = llm.generate(**entrada, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                                  pad_token_id=tokenizador.pad_token_id or tokenizador.eos_token_id)
        return tokenizador.batch_decode(salida[:, entrada["input_ids"].shape[1]:], skip_special_tokens=True)
    return generar


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--motor", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--modelo", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    parser.add_argument("--salida", type=Path, default=RESPUESTAS_DIR, help="Carpeta donde escribir (ej. en Drive)")
    parser.add_argument("--lote", type=int, help="Casos por lote (por defecto: 64 con vllm, 8 con transformers)")
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--limite", type=int, help="Hacer solo N casos")
    args = parser.parse_args()

    nombre = args.modelo.split("/")[-1].lower()
    args.salida.mkdir(parents=True, exist_ok=True)
    archivo = args.salida / f"{nombre}.jsonl"
    # Se saltan los casos ya respondidos por cualquier profesor (los del repo y los de la carpeta de salida)
    hechos = ids_ya_respondidos({RESPUESTAS_DIR, args.salida})
    with open(CASOS, encoding="utf-8") as f:
        casos = [c for c in map(json.loads, f) if c["split"] == "train" and c["id"] not in hechos]
    casos = casos[::-1][:args.limite]  # desde el final: los de Groq van desde el principio
    lote = args.lote or (64 if args.motor == "vllm" else 8)
    print(f"{nombre} ({args.motor}): {len(hechos)} casos ya respondidos; este profesor hará {len(casos)}", flush=True)

    generar = (generador_vllm if args.motor == "vllm" else generador_transformers)(args.modelo, args.max_tokens, args.max_len)
    limpios, inicio = 0, time.time()
    with open(archivo, "a", encoding="utf-8") as f:
        for i in range(0, len(casos), lote):
            grupo = casos[i:i + lote]
            for caso, texto in zip(grupo, generar([prompt_profesor(c, c["pasajes"]) for c in grupo])):
                texto = normalizar(texto)
                v = verificar(texto, caso)
                limpios += bool(es_ejemplo_limpio(v))
                f.write(json.dumps({"id": caso["id"], "split": caso["split"], "profesor": args.modelo,
                                    "respuesta": texto, "verificacion": v}, ensure_ascii=False) + "\n")
            f.flush()
            hechos_ahora = min(i + lote, len(casos))
            minutos = (time.time() - inicio) / 60
            print(f"  {hechos_ahora}/{len(casos)} casos, {limpios} limpios, {hechos_ahora / minutos:.1f} casos/min", flush=True)
    print(f"Respuestas en {archivo}")


if __name__ == "__main__":
    main()
