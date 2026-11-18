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


def casos_v3():
    """
    v3 (2026-11-04): los mismos casos y splits, con los pasajes que da la APP (modelos_propios.RecuperadorRAG:
    alias, umbral y fragmento según el estado). En el v1 y v2 los pasajes venían solo de la búsqueda densa por nombre,
    y en el 92 % de los analitos alterados de train no coinciden con los que el generador recibe en la app.
    Los exámenes sin guía quedan con texto vacío, igual que en la app.
    """
    sys.path.insert(0, str(RAIZ))
    import modelos_propios as mp
    recuperador = mp.RecuperadorRAG()
    with open(SALIDA, encoding="utf-8") as f:
        casos = [json.loads(linea) for linea in f]
    cache, cambian, alterados = {}, 0, 0
    destino = SALIDA.with_name("casos_v3.jsonl")
    with open(destino, "w", encoding="utf-8") as f:
        for c in casos:
            pasajes = []
            for a, viejo in zip(c["analitos"], c["pasajes"]):
                clave = (a["examen"], c["tipo_de_examen"], a["estado"] if a["estado"] in ("ALTO", "BAJO") else None)
                if clave not in cache:
                    cache[clave] = recuperador.buscar(f"{a['examen']} ({c['tipo_de_examen']})", examen=a["examen"],
                                                      tipo_examen=c["tipo_de_examen"], estado=clave[2])
                doc = cache[clave]
                pasajes.append({"examen": a["examen"], "titulo": doc["titulo"] if doc else "",
                                "texto": recortar(doc["texto"]) if doc else "", "doc_id": doc["id"] if doc else None,
                                "acierto_rag": bool(doc) and a["examen_canonico"] in doc["analitos"]})
                if clave[2]:
                    alterados += 1
                    cambian += pasajes[-1]["doc_id"] != viejo["doc_id"]
            f.write(json.dumps({**c, "pasajes": pasajes}, ensure_ascii=False) + "\n")
    print(f"{len(casos)} casos, {len(cache)} consultas distintas. Analitos alterados con otro pasaje: "
          f"{cambian}/{alterados} ({cambian / alterados:.1%})")
    print(f"Casos guardados en {destino.relative_to(RAIZ)}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--version", default="v1", choices=["v1", "v3"],
                        help="v3: mismos casos de casos.jsonl con los pasajes de la app -> casos_v3.jsonl")
    args = parser.parse_args()
    if args.version == "v3":
        return casos_v3()

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
