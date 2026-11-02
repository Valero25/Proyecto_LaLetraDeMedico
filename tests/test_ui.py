"""
Pruebas para las nuevas funcionalidades visuales e interactivas de LetraDeMédico.
"""
import sys
from pathlib import Path
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import ui_components
from backend import LetraDeMedicoBackend


def test_presets_structure():
    assert "hemograma_lipidos" in ui_components.PRESETS
    assert "metabolismo_tiroides" in ui_components.PRESETS
    assert "renal_urolisis" in ui_components.PRESETS

    for k, p in ui_components.PRESETS.items():
        assert "nombre" in p and "analitos" in p
        assert len(p["analitos"]) >= 4
        assert len(p["explicaciones"]) >= 4
        assert len(p["preguntas"]) >= 3


def test_calcular_gauge_html_rango_numerico():
    # Rango con dos límites (ej. Glucosa: 108 en rango 70 - 99)
    gauge_alto = ui_components.calcular_gauge_html("108", "70 - 99", "ALTO")
    assert gauge_alto is not None
    assert "ldm-gauge-marker" in gauge_alto
    assert "ldm-marker-pin" in gauge_alto

    # Rango con límite superior (ej. Colesterol: 178 en < 200)
    gauge_menor = ui_components.calcular_gauge_html("178", "< 200", "NORMAL")
    assert gauge_menor is not None
    assert "Deseable" in gauge_menor

    # Rango con límite inferior (ej. HDL: 36 en > 40)
    gauge_mayor = ui_components.calcular_gauge_html("36", "> 40", "BAJO")
    assert gauge_mayor is not None
    assert "Bajo" in gauge_mayor

    # Cualitativo (Negativo)
    gauge_cualitativo = ui_components.calcular_gauge_html("Negativo", "Negativo", "NORMAL")
    assert gauge_cualitativo is None


def test_render_tarjeta_html():
    html = ui_components.render_tarjeta_html(
        nombre_examen="Glucosa",
        valor="108",
        unidad="mg/dL",
        estado="ALTO",
        rango="70 - 99",
        explicacion="La glucosa está elevada."
    )
    assert "ldm-card-alto" in html
    assert "Glucosa" in html
    assert "108 mg/dL" in html
    assert "ELEVADO" in html
    assert "ldm-gauge-box" in html
    assert "La glucosa está elevada." in html



def test_generar_ficha_medica_txt():
    analitos = [
        {"examen": "Glucosa", "valor": "115", "unidad": "mg/dL", "rango_referencia": "70 - 99", "estado": "ALTO"},
        {"examen": "Hemoglobina", "valor": "14.0", "unidad": "g/dL", "rango_referencia": "13.0 - 17.0", "estado": "NORMAL"},
    ]
    preguntas = ["¿Debo repetir la prueba de glucosa?", "¿Debo consultar a un nutricionista?"]
    ficha = ui_components.generar_ficha_medica_txt("Perfil Metabólico", analitos, preguntas, "Tengo antecedentes familiares")

    assert "GUÍA PARA LA CONSULTA MÉDICA" in ficha
    assert "Glucosa: 115 mg/dL" in ficha
    assert "Hemoglobina: 14.0 g/dL" in ficha
    assert "¿Debo repetir la prueba de glucosa?" in ficha
    assert "Tengo antecedentes familiares" in ficha
    assert "AVISO IMPORTANTE" in ficha


def test_responder_pregunta_seguimiento_modo_propio():
    class RAGQueFalla:
        def buscar(self, consulta, examen=None):
            raise RuntimeError("sin índice")

    # Si el RAG no está disponible, el chat responde con el mensaje prudente en vez de caerse
    b = LetraDeMedicoBackend(modo="propio", modelos={"rag": RAGQueFalla()})
    analitos = [{"examen": "Glucosa", "valor": "115", "unidad": "mg/dL", "rango_referencia": "70 - 99", "estado": "ALTO"}]
    respuesta = b.responder_pregunta_seguimiento(analitos, "Control de Glucosa", "¿Es peligroso tener 115?")
    assert "¿Es peligroso tener 115?" in respuesta
    assert "consulta médica" in respuesta.lower()
