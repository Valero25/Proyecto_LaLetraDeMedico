"""
Fase 2: Evalúa la recuperación del RAG.

Compara cuatro recuperadores sobre las mismas consultas:
- subcadena: la búsqueda actual del backend sobre mock_medical_kb.json (5 entradas)
- bm25: baseline léxico sobre la base de conocimiento nueva
- e5-small: embeddings genéricos multilingües (intfloat/multilingual-e5-small)
- alia: embeddings biomédicos en español (SINAI/ALIA-MrBERT-es-biomedical-embeddings)
Con el sufijo "+alias", los fragmentos se indexan con los alias del catálogo (ver recuperador.py).
Ojo: las consultas de tipo alias salen de esos mismos alias, así que con "+alias" son fáciles por
construcción. La prueba justa de esa variante son las consultas "manual".

Las consultas salen del catálogo (cada alias tal como aparece en un examen, con estado, como
pregunta y con el panel) más las consultas difíciles de training/rag/consultas_manuales.json.

Una consulta acierta en k si alguno de los k primeros fragmentos explica un analito relevante.

Uso:
    python evaluacion/evaluar_rag.py [--modelos subcadena bm25 e5-small alia]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "training" / "rag"))

from ajustar_embeddings import MODELOS as MODELOS_AJUSTABLES  # noqa: E402
from recuperador import (  # noqa: E402
    MODELO_ALIA, RecuperadorBM25, RecuperadorDenso, RecuperadorSubcadena, cargar_documentos, normalizar,
)

CATALOGO_PATH = RAIZ / "training" / "sintetico" / "catalogo_analitos.json"
MANUALES_PATH = RAIZ / "training" / "rag" / "consultas_manuales.json"
SALIDA = RAIZ / "evaluacion" / "resultados" / "fase2_rag.json"
SPLIT_PATH = RAIZ / "training" / "rag" / "split_analitos.json"
KS = (1, 3, 5)
K_MAX = 10

ESTADOS_POR_TIPO = {"rango": ["alto", "bajo"], "max": ["alto"], "min": ["bajo"], "cualitativo": ["positivo"]}


def construir_consultas():
    with open(CATALOGO_PATH, encoding="utf-8") as f:
        catalogo = json.load(f)

    # Un alias como "Glucosa" pertenece a glucosa (sangre) y a glucosa_orina: sin más contexto ambas valen
    claves_por_alias = defaultdict(set)
    for clave, a in catalogo["analitos"].items():
        for nombre in a["nombres"]:
            claves_por_alias[normalizar(nombre)].add(clave)

    consultas = []
    for clave, a in catalogo["analitos"].items():
        for nombre in a["nombres"]:
            relevantes = sorted(claves_por_alias[normalizar(nombre)])
            consultas.append({"tipo": "alias", "consulta": nombre, "relevantes": relevantes})
            for estado in ESTADOS_POR_TIPO[a["tipo"]]:
                consultas.append({"tipo": "alias_estado", "consulta": f"{nombre} {estado}", "relevantes": relevantes})
            estado = ESTADOS_POR_TIPO[a["tipo"]][0]
            consultas.append({"tipo": "pregunta", "consulta": f"¿Qué significa tener {nombre} {estado}?", "relevantes": relevantes})
            # Con el panel el alias deja de ser ambiguo: así consultaría la app, que conoce el tipo de examen
            for panel, claves in catalogo["paneles"].items():
                if clave in claves:
                    consultas.append({"tipo": "alias_panel", "consulta": f"{nombre} ({panel})", "relevantes": [clave]})

    # Quitar duplicados exactos (ej. "Glucosa" aparece desde dos claves)
    unicas = {(c["tipo"], c["consulta"]): c for c in consultas}
    consultas = list(unicas.values())

    with open(MANUALES_PATH, encoding="utf-8") as f:
        consultas += [{"tipo": "manual", **c} for c in json.load(f)["consultas"]]
    return consultas


def crear_recuperador(nombre, documentos):
    """Un sufijo "+alias" indexa los fragmentos enriquecidos con los alias del catálogo."""
    nombre = nombre.removesuffix("+alias")
    if nombre == "subcadena":
        return RecuperadorSubcadena()
    if nombre == "bm25":
        return RecuperadorBM25(documentos)
    if nombre == "e5-small":
        return RecuperadorDenso(documentos, "intfloat/multilingual-e5-small", prefijo_consulta="query: ", prefijo_documento="passage: ")
    if nombre == "bge-m3":
        return RecuperadorDenso(documentos, "BAAI/bge-m3")
    if nombre == "alia":
        return RecuperadorDenso(documentos, MODELO_ALIA)
    if "-ajustado-" in nombre:
        # Modelos de training/rag/ajustar_embeddings.py, ej. "alia-ajustado-split"
        base = MODELOS_AJUSTABLES[nombre.split("-ajustado-")[0]]
        return RecuperadorDenso(documentos, str(RAIZ / "models" / "rag" / nombre),
                                prefijo_consulta=base["prefijo_consulta"] or None, prefijo_documento=base["prefijo_documento"])
    raise ValueError(nombre)


def grupo_de_split(consulta, split):
    """'no_vistos' si todos sus analitos relevantes quedaron fuera del entrenamiento del ajuste contrastivo."""
    relevantes = set(consulta["relevantes"])
    if relevantes <= set(split["test"]):
        return "no_vistos"
    if relevantes <= set(split["train"]):
        return "vistos"
    return None


def evaluar(recuperador, consultas, split=None):
    por_tipo = defaultdict(lambda: defaultdict(float))
    fallos = []
    for c in consultas:
        resultados = recuperador.buscar(c["consulta"], k=K_MAX)
        relevantes = set(c["relevantes"])
        posicion = next((i for i, (doc, _) in enumerate(resultados, 1) if relevantes & set(doc["analitos"])), None)
        grupos = [c["tipo"], "total"]
        if split and grupo_de_split(c, split):
            grupos.append(grupo_de_split(c, split))
        for grupo in grupos:
            m = por_tipo[grupo]
            m["n"] += 1
            for k in KS:
                m[f"hit@{k}"] += posicion is not None and posicion <= k
            m["mrr@10"] += 1 / posicion if posicion else 0
        if posicion != 1:
            primero = resultados[0][0].get("titulo", resultados[0][0]["id"]) if resultados else None
            fallos.append({"tipo": c["tipo"], "consulta": c["consulta"], "relevantes": c["relevantes"],
                           "posicion": posicion, "primer_resultado": primero})
    metricas = {
        grupo: {k: (v if k == "n" else v / m["n"]) for k, v in m.items()}
        for grupo, m in por_tipo.items()
    }
    return metricas, fallos


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelos", nargs="+",
                        default=["subcadena", "bm25", "e5-small", "alia", "bm25+alias", "e5-small+alias", "alia+alias"])
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8")) if SPLIT_PATH.exists() else None
    documentos = {False: cargar_documentos(con_alias=False), True: cargar_documentos(con_alias=True)}
    consultas = construir_consultas()
    tipos_consulta = sorted({c["tipo"] for c in consultas})
    tipos = tipos_consulta + ["total"] + (["vistos", "no_vistos"] if split else [])
    print(f"{len(documentos[False])} fragmentos, {len(consultas)} consultas: "
          + ", ".join(f"{t}={sum(c['tipo'] == t for c in consultas)}" for t in tipos_consulta) + "\n")

    resultados = {}
    for nombre in args.modelos:
        ruta = RAIZ / "models" / "rag" / nombre.removesuffix("+alias")
        if "-ajustado-" in nombre and not (ruta / "model.safetensors").exists():
            print(f"== {nombre}: se omite, no existe {ruta.relative_to(RAIZ)} (¿falló su entrenamiento?)\n")
            continue
        inicio = time.time()
        recuperador = crear_recuperador(nombre, documentos[nombre.endswith("+alias")])
        metricas, fallos = evaluar(recuperador, consultas, split)
        resultados[nombre] = {"metricas": metricas, "segundos": round(time.time() - inicio, 1), "fallos": fallos}
        print(f"== {nombre} ({resultados[nombre]['segundos']}s)")
        print(f"   {'tipo':<14}{'n':>5}" + "".join(f"{f'hit@{k}':>9}" for k in KS) + f"{'mrr@10':>9}")
        for t in tipos:
            m = metricas.get(t)
            if not m:
                continue
            print(f"   {t:<14}{int(m['n']):>5}" + "".join(f"{m[f'hit@{k}']:>9.1%}" for k in KS) + f"{m['mrr@10']:>9.3f}")
        print()

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump({"fragmentos": len(documentos[False]), "consultas": len(consultas), "split": split,
                   "resultados": resultados}, f, indent=2, ensure_ascii=False)
    print(f"Resultados (incluye cada fallo) guardados en {args.salida}")


if __name__ == "__main__":
    main()
