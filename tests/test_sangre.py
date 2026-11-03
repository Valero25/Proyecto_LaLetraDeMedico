"""
Robustez con formatos reales de laboratorios de sangre (2026-11-02) y el alcance del proyecto: solo sangre.

Antes de estos cambios, 11 de 33 formatos reales fallaban. 4 daban un estado EQUIVOCADO:
- "Hombres: 13.5 - 17.5 Mujeres: 12.0 - 15.5": siempre el de hombres (una mujer con 13,0 salía BAJO).
- "Normal: < 5.7 Prediabetes: 5.7 - 6.4 ...": una HbA1c de 5,9 % salía NORMAL.
- "Deseable: < 200 Limítrofe: 200 - 239 ...": un colesterol de 110 salía BAJO.
Sin modelos reales ni Gemini.
"""
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import ficha_consulta  # noqa: E402
import ui_components  # noqa: E402
from backend import (FUERA_DE_ALCANCE, LetraDeMedicoBackend, es_fuera_de_alcance, limpiar_paciente,  # noqa: E402
                     normalizar_rango, segmentos_rango)

evaluar = LetraDeMedicoBackend.evaluar_estado_analito


@pytest.mark.parametrize("valor, rango, esperado", [
    ("150", "Menor a 200", "NORMAL"), ("250", "Menor a 200", "ALTO"), ("35", "Mayor a 40", "BAJO"),
    ("120", "Inferior a 150", "NORMAL"), ("30", "Superior a 40", "BAJO"), ("45", "Máximo 40", "ALTO"),
    ("40", "Máximo 40", "NORMAL"), ("8", "Mínimo 10", "BAJO"), ("180", "< o = 200", "NORMAL"),
    ("200", "Menor o igual a 200", "NORMAL"), ("200", "Menor a 200", "ALTO"), ("40", "Igual o mayor a 40", "NORMAL"),
    ("45", "Hasta 40 U/L", "ALTO"), ("*420.0", "173.0 - 414.0", "ALTO"), ("11.2 L", "12.0 - 16.0", "BAJO"),
    ("7.20", "4.0 - 11.0 x10^3/uL", "NORMAL"), ("Reactivo", "No reactivo", "ANORMAL"),
    ("No reactivo", "No reactivo", "NORMAL"), ("Hemolizada", "70 - 99", "DESCONOCIDO"),
])
def test_formatos_reales_en_palabras(valor, rango, esperado):
    assert evaluar(valor, rango) == esperado


RANGO_SEXO = "Hombres: 13.5 - 17.5 g/dL Mujeres: 12.0 - 15.5 g/dL"


def test_rango_por_sexo_usa_el_del_paciente():
    assert evaluar("13.0", RANGO_SEXO, "F") == "NORMAL"
    assert evaluar("13.0", RANGO_SEXO, "M") == "BAJO"
    assert evaluar("13.0", "H: 13.5 - 17.5 / M: 12.0 - 15.5", "F") == "NORMAL"   # H/M: M es mujer
    assert evaluar("13.0", "M: 13.5 - 17.5 F: 12.0 - 15.5", "M") == "BAJO"      # M/F: M es masculino


def test_rango_por_sexo_sin_sexo_solo_si_ambos_coinciden():
    assert evaluar("13.0", RANGO_SEXO) == "DESCONOCIDO"   # bajo para hombre, normal para mujer
    assert evaluar("15.0", RANGO_SEXO) == "NORMAL"        # normal para los dos
    assert evaluar("18.0", RANGO_SEXO) == "ALTO"


def test_rango_con_categorias_usa_la_normal():
    assert evaluar("5.9", "Normal: < 5.7 Prediabetes: 5.7 - 6.4 Diabetes: >= 6.5") == "ALTO"
    assert evaluar("5.4", "Normal: < 5.7 Prediabetes: 5.7 - 6.4 Diabetes: >= 6.5") == "NORMAL"
    assert evaluar("110", "Deseable: < 200 Limítrofe: 200 - 239 Alto: >= 240") == "NORMAL"
    assert evaluar("57.0", "Sin riesgo: Mayor de 55 mg/dl. Riesgo moderado: 35 - 55 mg/dl. Alto Riesgo: Menor de 35") == "NORMAL"


def test_intervalo_unico_antes_de_las_categorias_manda():
    # El colesterol total del PDF real: intervalo del laboratorio y luego niveles de riesgo
    rango = "100.0 - 200.0 Sin riesgo: Menor de 200.0 mg/dl Riesgo moderado: 200.0 -239.0 mg/dl"
    assert LetraDeMedicoBackend.evaluar_con_rango("161.5", rango) == ("NORMAL", "100.0 - 200.0")
    assert evaluar("90", rango) == "BAJO"


def test_solo_categorias_de_riesgo_no_se_adivina():
    assert evaluar("110", "Limítrofe: 200 - 239 Alto: >= 240") == "DESCONOCIDO"
    assert evaluar("5.9", "Prediabetes: 5.7 - 6.4 Diabetes: >= 6.5") == "DESCONOCIDO"


def test_etiquetas_de_poblacion():
    assert evaluar("1.0", "Adultos: 0.7 - 1.3") == "NORMAL"
    assert evaluar("1.0", "Niños: 0.3 - 0.7 Adultos: 0.7 - 1.3") == "DESCONOCIDO"   # discrepan
    assert evaluar("Negativo", "Negativo < 1:40") == "NORMAL"                     # un título no es etiqueta


def test_segmentos_no_pegan_la_unidad_a_la_etiqueta():
    pre, pares = segmentos_rango(RANGO_SEXO)
    assert pre == "" and pares == [("hombres", "13.5 - 17.5 g/dl"), ("mujeres", "12.0 - 15.5 g/dl")]
    assert segmentos_rango("Sin riesgo: Mayor de 55 mg/dl. Riesgo moderado: 35 - 55")[1][1][0] == "riesgo moderado"


def test_rango_aplicado():
    assert LetraDeMedicoBackend.evaluar_con_rango("13.0", RANGO_SEXO, "F") == ("NORMAL", "12.0 - 15.5 g/dl")
    assert LetraDeMedicoBackend.evaluar_con_rango("13.0", RANGO_SEXO)[1] == ""
    assert LetraDeMedicoBackend.evaluar_con_rango("90", "70 - 99") == ("NORMAL", "70 - 99")


def test_normalizar_rango_en_palabras():
    assert normalizar_rango("Menor o igual a 5,7") == "<=5.7"
    assert normalizar_rango("Superior a 40") == ">40"
    assert normalizar_rango("Máximo 40") == "<=40"
    assert normalizar_rango("< o = 200") == "<=200"


# --- Paciente ---

def test_limpiar_paciente():
    assert limpiar_paciente({"sexo": "Masculino", "edad": "15 Años", "nombre": "X"}) == {"sexo": "M", "edad": 15}
    assert limpiar_paciente({"sexo": "F", "edad": 52}) == {"sexo": "F", "edad": 52}
    assert limpiar_paciente({"sexo": "otro", "edad": 300}) == {}
    assert limpiar_paciente(None) == {}


# --- Alcance: solo sangre ---

def test_fuera_de_alcance():
    assert not es_fuera_de_alcance({"examen": "GLUCOSA EN SUERO U OTRO FLUIDO DIFERENTE A ORINA"})
    assert not es_fuera_de_alcance({"examen": "Hemoglobina", "muestra": "sangre"})
    assert not es_fuera_de_alcance({"examen": "Creatinina", "muestra": "Suero"})
    assert es_fuera_de_alcance({"examen": "Proteínas", "muestra": "orina"})
    assert es_fuera_de_alcance({"examen": "Glucosa en orina"})
    assert es_fuera_de_alcance({"examen": "Leucocitos"}, "Uroanálisis")
    assert not es_fuera_de_alcance({"examen": "Leucocitos"}, "Hemograma y uroanálisis")
    assert es_fuera_de_alcance({"examen": "Sangre oculta", "muestra": "heces"})


class RAG:
    def recuperar(self, analitos, tipo_examen):
        self.analitos = [a["examen"] for a in analitos]
        return [{"examen": a["examen"], "titulo": "T", "texto": "Texto."} for a in analitos]


class Generador:
    def generar(self, caso, pasajes):
        self.caso = caso
        return {"explicaciones": [{"examen": a["examen"], "explicacion": "Tu valor está dentro del rango normal."}
                                  for a in caso["analitos"]], "preguntas_sugeridas": ["¿Debo repetirlo?"]}


class Guardrail:
    def revisar(self, r):
        for e in r["explicaciones"]:
            e["oraciones_marcadas"] = []
        return r


def backend():
    return LetraDeMedicoBackend(modo="propio", modelos={"rag": RAG(), "generador": Generador(), "guardrail": Guardrail()})


MIXTO = [
    {"examen": "Hemoglobina", "valor": "13.0", "unidad": "g/dL", "rango_referencia": RANGO_SEXO, "muestra": "sangre"},
    {"examen": "Proteínas", "valor": "Negativo", "unidad": "", "rango_referencia": "Negativo", "muestra": "orina"},
    {"examen": "Glucosa", "valor": "90", "unidad": "mg/dL", "rango_referencia": "70 - 99", "muestra": "sangre"},
]


def test_analitos_de_orina_no_pasan_por_el_rag_ni_el_generador():
    b = backend()
    salida = b.analizar_entidades([dict(a) for a in MIXTO], "Laboratorio", paciente={"sexo": "F", "edad": 40})
    assert b.modelo_propio("rag").analitos == ["Hemoglobina", "Glucosa"]
    exps = salida["analisis_final"]["explicaciones"]
    assert [e["examen"] for e in exps] == ["Hemoglobina", "Proteínas", "Glucosa"]   # el orden del documento
    assert exps[1]["explicacion"] == FUERA_DE_ALCANCE and exps[1]["fuera_de_alcance"]
    assert salida["datos_extraidos"][1]["estado"] == "NORMAL"                        # su estado sí se calcula
    assert any("solo para exámenes de sangre" in a for a in salida["analisis_final"]["avisos"])
    assert [p["examen"] for p in salida["pasajes_rag"]] == ["Hemoglobina", "Proteínas", "Glucosa"]


def test_el_sexo_decide_el_rango_y_llega_al_generador():
    b = backend()
    salida = b.analizar_entidades([dict(a) for a in MIXTO], "Laboratorio", paciente={"sexo": "F", "edad": 40})
    hemoglobina = salida["datos_extraidos"][0]
    assert hemoglobina["estado"] == "NORMAL" and hemoglobina["rango_aplicado"] == "12.0 - 15.5 g/dl"
    caso = b.modelo_propio("generador").caso
    assert caso["paciente"] == {"sexo": "F", "edad": 40}
    # El generador ve el rango aplicado, no el texto con las dos categorías
    assert caso["analitos"][0]["rango_referencia"] == "12.0 - 15.5 g/dl"
    assert salida["paciente"] == {"sexo": "F", "edad": 40}


def test_documento_sin_analitos_de_sangre():
    salida = backend().analizar_entidades([dict(MIXTO[1])], "Uroanálisis")
    assert "error" in salida and "sangre" in salida["error"]


def test_documento_completo_con_paciente_y_muestra():
    b = backend()
    b.step_1_and_2_multimodal_ner = lambda *a: {"tipo_de_examen": "Hemograma", "paciente": {"sexo": "Femenino", "edad": 40},
                                                "analitos": [dict(MIXTO[0])]}
    salida = b.procesar_resultado(b"", "application/pdf")
    assert salida["datos_extraidos"][0]["estado"] == "NORMAL" and salida["paciente"]["sexo"] == "F"


# --- Interfaz y ficha ---

def test_indicador_no_dibuja_rangos_con_etiquetas():
    assert ui_components.calcular_gauge_html("13.0", RANGO_SEXO, "NORMAL") is None
    assert ui_components.calcular_gauge_html("13.0", "12.0 - 15.5 g/dl", "NORMAL")


def test_ficha_muestra_el_rango_aplicado():
    analitos = [{"examen": "HDL", "valor": "57", "unidad": "mg/dl", "estado": "NORMAL",
                 "rango_referencia": "Sin riesgo: Mayor de 55. Riesgo moderado: 35 - 55", "rango_aplicado": "mayor de 55"}]
    c = ficha_consulta.contenido_ficha("Perfil", analitos, [], [])
    assert c["filas"][0]["referencia"] == "Mayor de 55"


@pytest.fixture
def ficha_larga():
    analitos = [{"examen": f"Analito {i}", "valor": "1", "unidad": "", "rango_referencia": "0 - 2", "estado": "ALTO"}
                for i in range(9)]
    explicaciones = [{"examen": a["examen"], "explicacion": "Texto largo de explicación. " * 6} for a in analitos]
    return ficha_consulta.ficha_pdf("Perfil", analitos, explicaciones, ["¿Pregunta?"] * 4)


def test_ficha_larga_se_genera(ficha_larga):
    assert ficha_larga.startswith(b"%PDF")
