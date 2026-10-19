"""
Fase 4: arma los casos de entrada del generador, con los pasajes que recupera el RAG de la Fase 2.

Por cada analito se busca "{examen} ({tipo de examen})" con el modelo elegido en la Fase 2
(alia-ajustado-completo + alias) y se toma el primer fragmento. Así el generador aprende con la misma
calidad de recuperación que tendrá en la app, errores incluidos (hit@1 de 95 %).

Casos: los de training/sintetico/generar_examenes.py (seed 42, el mismo split train/val/test de la
Fase 1), con un máximo de MAX_ANALITOS analitos para que el ejemplo quepa en la secuencia de entrenamiento.

Uso:
    python training/generador/preparar_casos.py [--train 1000] [--val 100] [--test 150]
"""
import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "training" / "rag"))
sys.path.insert(0, str(RAIZ / "training" / "sintetico"))

from generar_examenes import generar_dataset  # noqa: E402

SALIDA = RAIZ / "data" / "generador" / "casos.jsonl"
MODELO_RAG = RAIZ / "models" / "rag" / "alia-ajustado-completo"
MAX_ANALITOS = 8
MAX_CARACTERES_PASAJE = 700


def seleccionar(casos, cuotas):
    elegidos = {s: [] for s in cuotas}
    for c in casos:
        if len(c["analitos"]) <= MAX_ANALITOS and len(elegidos[c["split"]]) < cuotas[c["split"]]:
            elegidos[c["split"]].append(c)
    return [c for s in cuotas for c in elegidos[s]]


def recortar(texto, limite=MAX_CARACTERES_PASAJE):
    if len(texto) <= limite:
        return texto
    corte = texto[:limite].rsplit(" ", 1)[0]
    return corte + "…"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=150)
    args = parser.parse_args()

    from recuperador import RecuperadorDenso, cargar_documentos

    casos = seleccionar(generar_dataset(2000, seed=42), {"train": args.train, "val": args.val, "test": args.test})
    print(f"{len(casos)} casos con hasta {MAX_ANALITOS} analitos", flush=True)

    recuperador = RecuperadorDenso(cargar_documentos(con_alias=True), str(MODELO_RAG))
    consultas = sorted({f"{a['examen']} ({c['tipo_de_examen']})" for c in casos for a in c["analitos"]})
    print(f"{len(consultas)} consultas distintas al RAG", flush=True)
    primero = {q: recuperador.buscar(q, k=1)[0][0] for q in consultas}

    aciertos = total = 0
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as f:
        for c in casos:
            pasajes = []
            for a in c["analitos"]:
                doc = primero[f"{a['examen']} ({c['tipo_de_examen']})"]
                pasajes.append({"examen": a["examen"], "titulo": doc["titulo"], "texto": recortar(doc["texto"]),
                                "doc_id": doc["id"], "acierto_rag": a["examen_canonico"] in doc["analitos"]})
                aciertos += pasajes[-1]["acierto_rag"]
                total += 1
            f.write(json.dumps({**c, "pasajes": pasajes}, ensure_ascii=False) + "\n")
    print(f"Pasajes correctos según el catálogo: {aciertos}/{total} ({aciertos / total:.1%})")
    print(f"Casos guardados en {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
