"""
Fase 4 (v3): respuestas de un generador en GGUF (el mismo archivo que usa la app) sobre los casos de test, en el
computador local con llama.cpp (GPU si está disponible).

Sirve para comparar v1, v2 y v3 sobre los mismos casos y con los pasajes que da la app (casos_v3.jsonl), en el mismo
formato que generar_respuestas.py (Colab, transformers). Decodificación voraz, igual que allá. Reanudable.

    python training/generador/generar_respuestas_gguf.py --gguf models/generador/letrademedico-qwen2.5-3b-q4_k_m.gguf --nombre ajustado-v1
    python training/generador/generar_respuestas_gguf.py --gguf models/generador/letrademedico-qwen2.5-3b-v2-q4_k_m.gguf --nombre ajustado-v2
    python training/generador/generar_respuestas_gguf.py --gguf models/generador/letrademedico-qwen2.5-3b-v3-q4_k_m.gguf --nombre ajustado-v3
    python evaluacion/evaluar_generador.py --version v3
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(RAIZ))

from prompt import mensajes_estudiante  # noqa: E402

CASOS_V3 = RAIZ / "data" / "generador" / "casos_v3.jsonl"
SALIDAS_V3 = RAIZ / "data" / "generador" / "salidas_v3"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--nombre", required=True, help="Nombre del archivo de salida, ej. ajustado-v3")
    parser.add_argument("--casos", type=Path, default=CASOS_V3)
    parser.add_argument("--salidas", type=Path, default=SALIDAS_V3)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--limite", type=int)
    args = parser.parse_args()

    import modelos_propios as mp
    mp.registrar_dlls_cuda()
    from llama_cpp import Llama

    args.salidas.mkdir(parents=True, exist_ok=True)
    salida = args.salidas / f"{args.nombre}.jsonl"
    hechos = set()
    if salida.exists():
        with open(salida, encoding="utf-8") as f:
            hechos = {json.loads(linea)["id"] for linea in f}
    with open(args.casos, encoding="utf-8") as f:
        casos = [c for c in map(json.loads, f) if c["split"] == args.split and c["id"] not in hechos][:args.limite]
    print(f"{args.nombre}: {len(hechos)} hechos, faltan {len(casos)}", flush=True)
    if not casos:
        return

    llm = Llama(model_path=str(args.gguf), n_ctx=mp.N_CTX_GENERADOR, n_gpu_layers=mp.capas_gpu_por_defecto(),
                verbose=False)
    with open(salida, "a", encoding="utf-8") as f:
        for i, caso in enumerate(casos, 1):
            inicio = time.time()
            r = llm.create_chat_completion(mensajes_estudiante(caso, caso["pasajes"]), temperature=0.0,
                                           max_tokens=args.max_tokens)
            segundos = time.time() - inicio
            f.write(json.dumps({"id": caso["id"], "modelo": args.nombre, "gguf": args.gguf.name,
                                "respuesta": r["choices"][0]["message"]["content"],
                                "tokens": r["usage"]["completion_tokens"], "segundos": round(segundos, 1)},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(casos):
                print(f"  {i}/{len(casos)} ({segundos:.0f} s el último)", flush=True)
    print(f"Respuestas en {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
