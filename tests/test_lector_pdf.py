"""
Lector local de PDFs digitales (lector_pdf.py): el motor principal no necesita Gemini para leer un PDF del
laboratorio. Usa los PDFs sintéticos de evaluacion/evaluar_lector_pdf.py, de respuesta conocida.
Sin modelos ni Gemini.
"""
import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "evaluacion"))

import lector_pdf  # noqa: E402
from backend import LetraDeMedicoBackend  # noqa: E402
from evaluar_lector_pdf import DISENOS, pdf_de_caso  # noqa: E402

CASO = {
    "id": "prueba", "tipo_de_examen": "Electrolitos", "paciente": {"sexo": "F", "edad": 43},
    "analitos": [
        {"examen": "Sodio", "valor": "137", "unidad": "mmol/L", "rango_referencia": "135 - 145", "estado": "NORMAL"},
        {"examen": "K", "valor": "5,9", "unidad": "mmol/L", "rango_referencia": "3,5-5,1", "estado": "ALTO"},
        {"examen": "Nitrógeno ureico", "valor": "15", "unidad": "mg/dL", "rango_referencia": "7 a 20", "estado": "NORMAL"},
        {"examen": "Leucocitos", "valor": "3.5", "unidad": "x10^3/µL", "rango_referencia": "4.5 - 11.0", "estado": "BAJO"},
    ],
}


@pytest.mark.parametrize("diseno", DISENOS)
def test_lee_los_tres_disenos(diseno):
    leido = lector_pdf.leer_pdf(pdf_de_caso(CASO, diseno))
    assert leido["paciente"] == {"sexo": "F", "edad": 43}
    obtenido = [(a["examen"].lower(), a["valor"], a["unidad"], a["rango_referencia"]) for a in leido["analitos"]]
    assert obtenido == [(a["examen"].lower(), a["valor"], a["unidad"], a["rango_referencia"]) for a in CASO["analitos"]]
    assert all(a["muestra"] == "sangre" for a in leido["analitos"])


def test_abreviaturas_y_nombres_que_empiezan_como_ruido():
    # "K" tiene una letra y "Nitrógeno ureico" empieza por "nit" (NIT del laboratorio): ninguno es ruido
    nombres = [a["examen"] for a in lector_pdf.leer_pdf(pdf_de_caso(CASO, "tabla"))["analitos"]]
    assert "K" in nombres and "Nitrógeno ureico" in nombres
    assert lector_pdf.es_ruido("NIT: 900.123.456-7") and not lector_pdf.es_ruido("Nitritos")


def test_orina_se_marca_por_su_seccion():
    caso = {**CASO, "tipo_de_examen": "Análisis de Orina Completo",
            "analitos": [{"examen": "Proteínas", "valor": "3+", "unidad": "", "rango_referencia": "Negativo",
                          "estado": "ANORMAL"}]}
    leido = lector_pdf.leer_pdf(pdf_de_caso(caso, "tabla"))
    assert leido["analitos"][0]["valor"] == "3+" and leido["analitos"][0]["muestra"] == "orina"


def test_pdf_sin_texto_no_se_lee():
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.rect(20, 20, 100, 60, style="F")  # como un escaneo: solo imagen, sin capa de texto
    assert lector_pdf.leer_pdf(bytes(pdf.output())) is None
    assert lector_pdf.leer_pdf(b"esto no es un pdf") is None


def test_categorias_bajo_el_resultado():
    texto = (" COLESTEROL DE ALTA DENSIDAD\n"
             " Resultado:                                57.0   mg/dl\n"
             "                                                       Sin riesgo:        Mayor de 55 mg/dl.\n"
             "                                                       Riesgo moderado:  35 - 55 mg/dl.\n"
             "   METODO: COLORIMETRICO ENZIMATICO.\n")
    analitos, _, confiable = lector_pdf.analizar_texto([texto])
    assert confiable and analitos[0]["examen"] == "COLESTEROL DE ALTA DENSIDAD"
    assert analitos[0]["rango_referencia"] == "Sin riesgo: Mayor de 55 mg/dl. Riesgo moderado: 35 - 55 mg/dl"


def test_paciente_sin_nombre():
    assert lector_pdf.paciente_del_texto("Paciente: ANA PÉREZ  Edad: 15 Años  Género: Másculino") == {"edad": 15, "sexo": "M"}
    assert lector_pdf.paciente_del_texto("Sexo: F") == {"sexo": "F"}


# --- Integración con el backend ---

class Modelos:
    class RAG:
        def recuperar(self, analitos, tipo_examen):
            return [{"examen": a["examen"], "titulo": "T", "texto": "Texto."} for a in analitos]

    class Generador:
        def generar(self, caso, pasajes):
            return {"explicaciones": [{"examen": a["examen"], "explicacion": "Tu valor está dentro del rango normal."}
                                      for a in caso["analitos"]], "preguntas_sugeridas": ["¿Debo repetirlo?"]}

    class Guardrail:
        def revisar(self, r):
            for e in r["explicaciones"]:
                e["oraciones_marcadas"] = []
            return r


def backend_sin_gemini():
    b = LetraDeMedicoBackend(modo="propio", modelos={"rag": Modelos.RAG(), "generador": Modelos.Generador(),
                                                     "guardrail": Modelos.Guardrail()})

    def gemini_prohibido(*args):
        raise AssertionError("no debía llamar a Gemini")
    b.step_1_and_2_multimodal_ner = gemini_prohibido
    return b


def test_pdf_digital_se_lee_sin_gemini():
    salida = backend_sin_gemini().procesar_resultado(pdf_de_caso(CASO, "bloques"), "application/pdf")
    assert salida["lectura"] == "local"
    assert [a["estado"] for a in salida["datos_extraidos"]] == ["NORMAL", "ALTO", "NORMAL", "BAJO"]
    assert salida["paciente"] == {"sexo": "F", "edad": 43}


def test_foto_o_escaneo_va_a_gemini():
    b = backend_sin_gemini()
    llamadas = []
    b.step_1_and_2_multimodal_ner = lambda datos, mime: llamadas.append(mime) or {
        "tipo_de_examen": "X", "analitos": [{"examen": "Glucosa", "valor": "90", "rango_referencia": "70 - 99"}]}
    assert b.procesar_resultado(b"imagen", "image/png")["lectura"] == "gemini"
    assert b.procesar_resultado(b"no es pdf", "application/pdf")["lectura"] == "gemini"
    assert llamadas == ["image/png", "application/pdf"]


def test_lectura_forzada(monkeypatch):
    b = backend_sin_gemini()
    assert "error" in b.procesar_resultado(b"no es pdf", "application/pdf", lectura="local")
    llamado = []
    b.step_1_and_2_multimodal_ner = lambda *a: llamado.append(1) or {}
    b.procesar_resultado(pdf_de_caso(CASO, "tabla"), "application/pdf", lectura="gemini")
    assert llamado == [1]  # "Leer otra vez con Gemini" no pasa por el lector local


def test_pdf_real_de_prueba_si_esta_disponible():
    ruta = Path(r"C:\Users\Juan\Downloads\Juan David 2021.pdf")
    if not ruta.exists():
        pytest.skip("PDF real de prueba no disponible en este equipo")
    leido = lector_pdf.leer_pdf(ruta.read_bytes())
    assert len(leido["analitos"]) == 8 and leido["paciente"] == {"edad": 15, "sexo": "M"}
    assert leido["analitos"][7]["valor"] == "420.0" and leido["analitos"][7]["rango_referencia"] == "173.0 - 414.0"
    assert "Sin riesgo: Mayor de 55" in leido["analitos"][2]["rango_referencia"]
