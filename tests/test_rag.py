"""
Pruebas de la Fase 2: base de conocimiento y recuperadores (sin descargar modelos de embeddings).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests -v
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "training" / "rag"))
sys.path.insert(0, str(RAIZ / "evaluacion"))

from recuperador import RecuperadorBM25, RecuperadorSubcadena, cargar_documentos, texto_para_indexar  # noqa: E402
from evaluar_rag import construir_consultas  # noqa: E402

CATALOGO = json.loads((RAIZ / "training" / "sintetico" / "catalogo_analitos.json").read_text(encoding="utf-8"))


def test_base_de_conocimiento_cubre_todo_el_catalogo():
    documentos = cargar_documentos()
    cubiertas = {a for d in documentos for a in d["analitos"]}
    assert cubiertas == set(CATALOGO["analitos"])


def test_documentos_tienen_fuente_y_texto():
    for d in cargar_documentos():
        assert d["texto"].strip() and d["titulo"] and d["fuente"]
        assert d["url"] is None or d["url"].startswith("https://medlineplus.gov/spanish/")
        assert len({d["id"]}) == 1


def test_ids_unicos():
    ids = [d["id"] for d in cargar_documentos()]
    assert len(ids) == len(set(ids))


def test_alias_se_agregan_al_texto_indexado():
    doc = next(d for d in cargar_documentos() if "creatinina" in d["analitos"])
    assert "También se reporta como: Creatinina, CREA." in texto_para_indexar(doc)
    sin_alias = next(d for d in cargar_documentos(con_alias=False) if "creatinina" in d["analitos"])
    assert "También se reporta como" not in texto_para_indexar(sin_alias)


def test_bm25_con_alias_encuentra_abreviaturas():
    bm25 = RecuperadorBM25(cargar_documentos())
    doc, _ = bm25.buscar("TGO", k=1)[0]
    assert "ast" in doc["analitos"]


def test_subcadena_replica_el_error_del_backend_con_hemoglobina():
    """La búsqueda actual devuelve la ficha de HbA1c para 'Hemoglobina': por eso se reemplaza."""
    doc, _ = RecuperadorSubcadena().buscar("Hemoglobina")[0]
    assert doc["analitos"] == ["hemoglobina_a1c"]


def test_consultas_de_evaluacion_validas():
    consultas = construir_consultas()
    claves = set(CATALOGO["analitos"])
    assert len(consultas) > 400
    for c in consultas:
        assert c["relevantes"] and set(c["relevantes"]) <= claves, c
    # Un alias ambiguo sin panel acepta ambas claves; con panel, solo la correcta
    glucosa = next(c for c in consultas if c["tipo"] == "alias" and c["consulta"] == "Glucosa")
    assert glucosa["relevantes"] == ["glucosa", "glucosa_orina"]
    con_panel = next(c for c in consultas if c["consulta"] == "Glucosa (Análisis de Orina Completo)")
    assert con_panel["relevantes"] == ["glucosa_orina"]


def test_split_no_filtra_analitos_de_test_al_entrenamiento():
    """Ningún fragmento puede explicar a la vez un analito de train y uno de test."""
    from ajustar_embeddings import cargar_split, construir_ejemplos, MODELOS

    documentos = cargar_documentos()
    split = cargar_split(documentos)
    train, test = set(split["train"]), set(split["test"])
    assert not train & test
    assert train | test == set(CATALOGO["analitos"])
    for d in documentos:
        analitos = set(d["analitos"])
        assert analitos <= train or analitos <= test, d["id"]

    # Los ejemplos de la variante split no contienen fragmentos de analitos de test
    consultas = [{"consulta": "x", "doc_id": d["id"]} for d in documentos]
    anclas, positivos, negativos = construir_ejemplos(documentos, consultas, train, MODELOS["alia"])
    textos_test = {texto_para_indexar(d) for d in documentos if set(d["analitos"]) <= test}
    assert len(anclas) == sum(set(d["analitos"]) <= train for d in documentos)
    assert not (set(positivos) | set(negativos)) & textos_test
