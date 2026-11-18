"""
Fase 4: respuestas de Gemini sobre los casos de test, como línea base de comparación para el generador propio.

SOLO PARA EVALUAR: los términos de Gemini prohíben usar sus salidas para entrenar modelos que compitan con él.
Estas respuestas no entran a ningún dataset de entrenamiento.

Gemini recibe exactamente lo mismo que el estudiante: el prompt de sistema con las reglas, los datos del caso
y los mismos pasajes del RAG (training/generador/prompt.py). No ve la guía de estilo del profesor. Así la
comparación mide el modelo, no el prompt.

    python training/generador/generar_gemini.py [--limite N] [--pausa 4]

Guarda en data/generador/salidas/gemini.jsonl (lo lee evaluacion/evaluar_generador.py). Reanudable: salta los
casos ya respondidos. Usa los reintentos y el modelo de respaldo de backend.py; si se agota la cuota diaria,
se detiene y se puede reanudar otro día.
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).parent))

from prompt import SISTEMA, construir_usuario  # noqa: E402

CASOS = RAIZ / "data" / "generador" / "casos.jsonl"
SALIDA = RAIZ / "data" / "generador" / "salidas" / "gemini.jsonl"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--limite", type=int, help="Máximo de casos nuevos en esta corrida")
    parser.add_argument("--pausa", type=float, default=4.0, help="Segundos entre peticiones (límite por minuto)")
    args = parser.parse_args()

    from backend import LetraDeMedicoBackend
    from google.genai import errors as genai_errors
    backend = LetraDeMedicoBackend(modo="gemini")

    with open(CASOS, encoding="utf-8") as f:
        casos = [c for c in map(json.loads, f) if c["split"] == args.split]
    hechos = set()
    if SALIDA.exists():
        with open(SALIDA, encoding="utf-8") as f:
            hechos = {json.loads(linea)["id"] for linea in f if linea.strip()}
    pendientes = [c for c in casos if c["id"] not in hechos][:args.limite]
    print(f"{len(hechos)} hechos, {len(pendientes)} por hacer (split {args.split})", flush=True)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    for i, caso in enumerate(pendientes, 1):
        prompt = f"{SISTEMA}\n\n{construir_usuario(caso, caso['pasajes'])}"
        inicio = time.time()
        try:
            respuesta = backend._generar(prompt)
        except genai_errors.APIError as e:
            if e.code == 429 and "PerDay" in str(e):
                print("Cuota diaria de Gemini agotada: se reanuda otro día con el mismo comando.", flush=True)
                break
            print(f"{caso['id']}: error {e.code}; se salta por ahora", flush=True)
            continue
        fila = {"id": caso["id"], "modelo": "gemini", "respuesta": (respuesta.text or "").strip(),
                "segundos": round(time.time() - inicio, 1)}
        with open(SALIDA, "a", encoding="utf-8") as f:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
        print(f"[{len(hechos) + i}/{len(casos)}] {caso['id']} en {fila['segundos']} s", flush=True)
        time.sleep(args.pausa)


if __name__ == "__main__":
    main()
