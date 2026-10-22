"""
Pruebas de la Fase 5: el backend en modo "propio" con dobles de los modelos (sin cargar modelos reales ni llamar a Gemini).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests -v
"""
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from backend import NOTA_GUARDRAIL, LetraDeMedicoBackend  # noqa: E402

ANALITOS = [
    {"examen": "Glucosa", "valor": "126", "unidad": "mg/dL", "rango_referencia": "70 a 99"},
    {"examen": "Colesterol HDL", "valor": "55", "unidad": "mg/dL", "rango_referencia": "Mayor de 40"},
]


class RAGFalso:
    def recuperar(self, analitos, tipo_examen):
        self.consulta = (tuple(a["examen"] for a in analitos), tipo_examen)
        return [{"examen": a["examen"], "titulo": f"Prueba de {a['examen']}", "texto": "Texto de MedlinePlus."} for a in analitos]


class GeneradorFalso:
    def generar(self, caso, pasajes):
        self.caso, self.pasajes = caso, pasajes
        return {"explicaciones": [
            {"examen": "Glucosa", "explicacion": "Tu glucosa de 126 está por encima del rango. Esto confirma que tienes diabetes."},
            {"examen": "Colesterol HDL", "explicacion": "Tu HDL de 55 está dentro de lo esperado."},
        ], "preguntas_sugeridas": ["¿Debo repetir el examen?"]}


class GuardrailFalso:
    """Marca las oraciones que contienen "confirma"."""
    def revisar(self, resultado):
        for e in resultado["explicaciones"]:
            e["oraciones_marcadas"] = [o for o in e["explicacion"].split(". ") if "confirma" in o]
        return resultado


@pytest.fixture
def backend():
    return LetraDeMedicoBackend(modo="propio", modelos={"rag": RAGFalso(), "generador": GeneradorFalso(),
                                                         "guardrail": GuardrailFalso()})


def test_modo_propio_usa_los_tres_modelos(backend):
    salida = backend.analizar_entidades([dict(a) for a in ANALITOS], "Control de Diabetes", "Tomo metformina.")
    assert salida["modo"] == "propio"
    # Reglas corregidas en la Fase 5: "70 a 99" y "Mayor de 40" ya no quedan como DESCONOCIDO
    assert [a["estado"] for a in salida["datos_extraidos"]] == ["ALTO", "NORMAL"]
    # El RAG recibe el tipo de examen, y el generador el caso en el formato de su entrenamiento
    assert backend.modelo_propio("rag").consulta == (("Glucosa", "Colesterol HDL"), "Control de Diabetes")
    caso = backend.modelo_propio("generador").caso
    assert caso["tipo_de_examen"] == "Control de Diabetes" and caso["contexto_paciente"] == "Tomo metformina."
    assert len(salida["pasajes_rag"]) == 2


def test_guardrail_propio_marca_solo_la_explicacion_riesgosa(backend):
    salida = backend.analizar_entidades([dict(a) for a in ANALITOS], "Control de Diabetes")
    glucosa, hdl = salida["analisis_final"]["explicaciones"]
    assert glucosa["oraciones_marcadas"] and glucosa["explicacion"].endswith(NOTA_GUARDRAIL)
    assert not hdl["oraciones_marcadas"] and not hdl["explicacion"].endswith(NOTA_GUARDRAIL)
    assert "disclaimer" in salida["analisis_final"]


def test_modo_gemini_conserva_el_filtro_de_palabras():
    b = LetraDeMedicoBackend(modo="gemini")
    r = b.step_4_validation({"explicaciones": [{"examen": "X", "explicacion": "Esto es una anemia."},
                                                {"examen": "Y", "explicacion": "Tienes el valor en rango."}]})
    # La línea base se mantiene tal cual, con sus defectos: no ve la primera y marca la segunda
    assert not r["explicaciones"][0]["explicacion"].endswith(NOTA_GUARDRAIL)
    assert r["explicaciones"][1]["explicacion"].endswith(NOTA_GUARDRAIL)


def test_modo_invalido():
    with pytest.raises(ValueError):
        LetraDeMedicoBackend(modo="otro")


def test_prompt_sin_edad_ni_sexo():
    sys.path.insert(0, str(RAIZ / "training" / "generador"))
    from prompt import construir_usuario
    analitos = [{**ANALITOS[0], "estado": "ALTO"}]
    texto = construir_usuario({"tipo_de_examen": "Hemograma", "paciente": {}, "analitos": analitos}, [])
    assert "Paciente: sexo no indicado, edad no indicada." in texto
    # Los casos de entrenamiento (con sexo y edad) mantienen el formato exacto
    assert "Paciente: mujer, 52 años." in construir_usuario(
        {"tipo_de_examen": "Hemograma", "paciente": {"sexo": "F", "edad": 52}, "analitos": analitos}, [])
