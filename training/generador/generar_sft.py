"""
Fase 4: un modelo profesor en Groq redacta la respuesta de referencia para cada caso de train y val.

Groq limita la salida a unos 1.000 tokens por minuto por modelo. Para repartir la carga se corren
dos procesos en paralelo, cada uno con un profesor y una mitad de los casos:

    python training/generador/generar_sft.py --profesor qwen/qwen3.8-27b --particion 0
    python training/generador/generar_sft.py --profesor openai/gpt-oss-120b --particion 1

Se guarda la respuesta cruda junto con sus verificaciones (verificacion.py). El filtro final y el
chequeo de guardrails se hacen después, en dataset.py, para poder cambiar el criterio sin regenerar.

Reanudable: salta los casos que ya tienen respuesta en el archivo de salida.
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "training"))
sys.path.insert(0, str(Path(__file__).parent))

from groq_cliente import crear_sesion, pedir_json  # noqa: E402
from prompt import prompt_profesor  # noqa: E402
from verificacion import es_ejemplo_limpio, verificar  # noqa: E402

CASOS = RAIZ / "data" / "generador" / "casos.jsonl"
SALIDA_DIR = RAIZ / "data" / "generador" / "respuestas"
# Opciones por profesor para no gastar tokens de salida en razonamiento
EXTRA = {"openai/gpt-oss-120b": {"reasoning_effort": "low"}, "openai/gpt-oss-20b": {"reasoning_effort": "low"}}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--profesor", default="qwen/qwen3.8-27b")
    parser.add_argument("--particion", type=int, default=0, help="Este proceso toma los casos i con i %% partes == particion")
    parser.add_argument("--partes", type=int, default=2)
    parser.add_argument("--limite", type=int, help="Hacer solo N casos (prueba rápida)")
    args = parser.parse_args()

    with open(CASOS, encoding="utf-8") as f:
        casos = [c for i, c in enumerate(map(json.loads, f))
                 if c["split"] in ("train", "val") and i % args.partes == args.particion]
    salida = SALIDA_DIR / f"{args.profesor.replace('/', '_')}.jsonl"
    salida.parent.mkdir(parents=True, exist_ok=True)
    hechos = set()
    if salida.exists():
        with open(salida, encoding="utf-8") as f:
            hechos = {json.loads(linea)["id"] for linea in f}
    # Validación primero: si la cuota de Groq no alcanza para todo el train, igual hay con qué validar
    pendientes = sorted((c for c in casos if c["id"] not in hechos), key=lambda c: c["split"] != "val")[:args.limite]
    print(f"{args.profesor}: {len(hechos)} casos hechos; faltan {len(pendientes)}", flush=True)

    sesion = crear_sesion()
    limpios = 0
    inicio = time.time()
    with open(salida, "a", encoding="utf-8") as f:
        for i, caso in enumerate(pendientes, 1):
            try:
                respuesta = pedir_json(sesion, args.profesor, prompt_profesor(caso, caso["pasajes"]),
                                       temperatura=0.7, etiqueta=caso["id"], **EXTRA.get(args.profesor, {}))
            except RuntimeError as e:
                print(f"    se salta {caso['id']}: {e}", flush=True)
                continue
            texto = json.dumps(respuesta, ensure_ascii=False)
            v = verificar(texto, caso)
            limpios += bool(es_ejemplo_limpio(v))
            f.write(json.dumps({"id": caso["id"], "split": caso["split"], "profesor": args.profesor,
                                "respuesta": texto, "verificacion": v}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(pendientes):
                minutos = (time.time() - inicio) / 60
                print(f"  {i}/{len(pendientes)} casos, {limpios} limpios, {i / minutos:.1f} casos/min", flush=True)


if __name__ == "__main__":
    main()
