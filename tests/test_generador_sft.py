"""
Pruebas de la Fase 4: prompt del generador y verificaciones automáticas (sin modelos ni Groq).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests -v
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "training" / "generador"))

from prompt import GUIA_PROFESOR, SISTEMA, mensajes_estudiante, prompt_profesor  # noqa: E402
from verificacion import es_ejemplo_limpio, frases_para_guardrails, parsear, verificar  # noqa: E402

CASO = {
    "id": "t-1", "tipo_de_examen": "Control de Diabetes", "paciente": {"sexo": "F", "edad": 52},
    "contexto_paciente": "",
    "analitos": [
        {"examen": "Glucosa", "valor": "126,4", "unidad": "mg/dL", "rango_referencia": "70 - 100", "estado": "ALTO"},
        {"examen": "Glucosa en orina", "valor": "Negativo", "unidad": "", "rango_referencia": "Negativo", "estado": "NORMAL"},
    ],
}
PASAJES = [{"examen": "Glucosa", "titulo": "Prueba de glucosa", "texto": "Mide el azúcar en la sangre."},
           {"examen": "Glucosa en orina", "titulo": "Glucosa en orina", "texto": "Mide el azúcar en la orina."}]


def _respuesta(**cambios):
    r = {"explicaciones": [
        {"examen": "Glucosa", "explicacion": "Tu glucosa es de 126.4 mg/dL, por encima del rango. Puede deberse a varias causas."},
        {"examen": "Glucosa en orina", "explicacion": "El resultado es negativo, dentro de lo esperado."},
    ], "preguntas_sugeridas": ["¿Debo repetir el examen?", "¿Necesito cambiar mi dieta?", "¿Cada cuánto me controlo?"]}
    r.update(cambios)
    return json.dumps(r, ensure_ascii=False)


def test_estudiante_no_ve_la_guia_del_profesor():
    mensajes = mensajes_estudiante(CASO, PASAJES)
    assert [m["role"] for m in mensajes] == ["system", "user"]
    assert mensajes[0]["content"] == SISTEMA
    texto = " ".join(m["content"] for m in mensajes)
    assert GUIA_PROFESOR not in texto and GUIA_PROFESOR in prompt_profesor(CASO, PASAJES)
    assert "Glucosa: 126,4 mg/dL (referencia: 70 - 100) → ALTO" in mensajes[1]["content"]
    assert "[Glucosa en orina] Glucosa en orina: Mide el azúcar en la orina." in mensajes[1]["content"]
    con_respuesta = mensajes_estudiante(CASO, PASAJES, json.loads(_respuesta()))
    assert con_respuesta[-1]["role"] == "assistant" and parsear(con_respuesta[-1]["content"])


def test_respuesta_correcta_es_ejemplo_limpio():
    v = verificar(_respuesta(), CASO)
    assert v["json_valido"] and v["cobertura"] == 1 and v["estado_consistente"] == 1
    assert v["menciona_valor"] == 1  # 126,4 aparece como 126.4; "negativo" en el cualitativo
    assert es_ejemplo_limpio(v)


def test_detecta_json_invalido_y_examen_faltante():
    assert verificar("esto no es JSON", CASO) == {"json_valido": False}
    assert verificar("```json\n" + _respuesta() + "\n```", CASO)["json_valido"]
    falta = _respuesta(explicaciones=[{"examen": "Glucosa", "explicacion": "Tu glucosa está alta."}])
    v = verificar(falta, CASO)
    assert v["cobertura"] == 0.5 and not es_ejemplo_limpio(v)


def test_detecta_estado_inconsistente():
    mal = _respuesta(explicaciones=[
        {"examen": "Glucosa", "explicacion": "Tu glucosa de 126,4 no es preocupante."},
        {"examen": "Glucosa en orina", "explicacion": "El resultado es negativo, dentro de lo esperado."},
    ])
    v = verificar(mal, CASO)
    assert v["estado_consistente"] == 0.5 and not es_ejemplo_limpio(v)


def test_frases_para_guardrails():
    assert frases_para_guardrails(_respuesta()) == [
        "Tu glucosa es de 126.4 mg/dL, por encima del rango.", "Puede deberse a varias causas.",
        "El resultado es negativo, dentro de lo esperado."]


def test_contar_silabas_en_espanol():
    from verificacion import contar_silabas
    casos = {"paciente": 3, "día": 2, "glucosa": 3, "que": 1, "aorta": 3, "hoy": 1, "guitarra": 3,
             "colesterol": 4, "creatinina": 5, "hemoglobina": 5, "riñón": 2, "país": 2}
    assert {p: contar_silabas(p) for p in casos} == casos


def test_inflesz_premia_frases_cortas_y_palabras_simples():
    from verificacion import inflesz
    facil = "Tu azúcar está un poco alta. No es grave. Habla con tu médico."
    dificil = ("La hiperglucemia observada evidencia una alteración del metabolismo de los carbohidratos "
               "posiblemente secundaria a resistencia periférica a la insulina.")
    assert inflesz(facil) > 65 > 40 > inflesz(dificil)
    assert inflesz("") is None


def test_profesor_recibe_pasajes_recortados():
    largo = [{**PASAJES[0], "texto": "palabra " * 200}, PASAJES[1]]
    prompt = prompt_profesor(CASO, largo, max_caracteres=50)
    assert "palabra " * 5 + "palabra…" in prompt and "palabra " * 7 not in prompt
    assert "palabra " * 200 in mensajes_estudiante(CASO, largo)[1]["content"]  # el estudiante ve el pasaje completo
