"""
Pruebas de robustez del pipeline y de la app: salidas incompletas o raras del generador, fallos de los
modelos, exámenes largos y texto que no se debe interpretar como HTML. Sin modelos reales ni Gemini.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests -v
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import modelos_propios  # noqa: E402
import ui_components  # noqa: E402
from backend import (LOTE_GENERADOR, NOTA_GUARDRAIL, SIN_EXPLICACION, LetraDeMedicoBackend,  # noqa: E402
                     alinear_explicaciones, limpiar_entidades, normalizar_rango)


def analitos(n):
    return [{"examen": f"Analito {i}", "valor": str(10 + i), "unidad": "mg/dL", "rango_referencia": "5 - 50"}
            for i in range(n)]


class RAGFalso:
    def recuperar(self, analitos, tipo_examen):
        return [{"examen": a["examen"], "titulo": f"Página {a['examen']}", "texto": "Texto."} for a in analitos]

    def buscar(self, consulta, examen=None):
        self.consulta, self.examen = consulta, examen
        return {"titulo": "Prueba de colesterol", "url": "https://medlineplus.gov/spanish/x.html",
                "texto": "El HDL es el colesterol bueno. Usted tiene una enfermedad grave. Se mide en sangre."}


class GuardrailFalso:
    """Marca las oraciones que contienen "enfermedad" o "confirma"."""
    def revisar(self, resultado):
        for e in resultado["explicaciones"]:
            frases = [o.strip() for o in e["explicacion"].replace(". ", ".|").split("|")]
            e["oraciones_marcadas"] = [o for o in frases if "enfermedad" in o or "confirma" in o]
        return resultado


class GeneradorQueOmite:
    """Explica todos los analitos del lote salvo los de `omitir`; registra cada lote que recibe."""
    def __init__(self, omitir=(), falla_si_contiene=None):
        self.lotes, self.omitir, self.falla_si_contiene = [], set(omitir), falla_si_contiene

    def generar(self, caso, pasajes):
        nombres = [a["examen"] for a in caso["analitos"]]
        self.lotes.append(nombres)
        assert len(pasajes) == len(nombres)  # cada lote recibe solo sus pasajes
        if self.falla_si_contiene in nombres:
            raise ValueError("contexto desbordado")
        # Solo omite en la primera llamada: en el reintento (lote distinto) sí responde
        omitir = self.omitir if len(self.lotes) == 1 else set()
        return {"explicaciones": [{"examen": n, "explicacion": f"Explicación de {n}: está dentro del rango."} for n in nombres if n not in omitir],
                "preguntas_sugeridas": ["¿Debo repetir el examen?", f"¿Qué significa {nombres[0]}?"]}


def backend_propio(generador, rag=None, guardrail=None):
    return LetraDeMedicoBackend(modo="propio", modelos={"rag": rag or RAGFalso(), "generador": generador,
                                                         "guardrail": guardrail or GuardrailFalso()})


# --- alinear_explicaciones ---

def test_alinear_empareja_nombres_renombrados_y_conserva_el_orden():
    entidades = [{"examen": "Glucosa"}, {"examen": "Colesterol HDL"}, {"examen": "Colesterol total"}]
    r = alinear_explicaciones(entidades, {"explicaciones": [
        {"examen": "Colesterol Total:", "explicacion": "total"},
        {"examen": "Glucosa en ayunas", "explicacion": "glucosa"},
        {"examen": "colesterol hdl", "explicacion": "hdl"},
    ]})
    assert [e["explicacion"] for e in r["explicaciones"]] == ["glucosa", "hdl", "total"]
    assert [e["examen"] for e in r["explicaciones"]] == ["Glucosa", "Colesterol HDL", "Colesterol total"]


def test_alinear_rellena_los_omitidos_sin_inventar_estado():
    r = alinear_explicaciones([{"examen": "Glucosa"}, {"examen": "Urea"}],
                              {"explicaciones": [{"examen": "Glucosa", "explicacion": "ok"}]})
    assert r["explicaciones"][1]["explicacion"] == SIN_EXPLICACION
    assert r["explicaciones"][1]["sin_explicacion"] is True
    assert "estado" not in r["explicaciones"][1]


def test_alinear_tolera_salidas_rotas():
    for roto in (None, [], "texto", {"explicaciones": "x"}, {"explicaciones": [None, {"examen": "A"}, 3]}):
        r = alinear_explicaciones([{"examen": "A"}], roto)
        assert len(r["explicaciones"]) == 1 and r["explicaciones"][0]["sin_explicacion"]


def test_alinear_por_posicion_solo_si_las_cantidades_coinciden():
    entidades = [{"examen": "Hb"}, {"examen": "VCM"}]
    r = alinear_explicaciones(entidades, {"explicaciones": [{"examen": "Hemoglobina", "explicacion": "1"},
                                                            {"examen": "Volumen corpuscular", "explicacion": "2"}]})
    assert [e["explicacion"] for e in r["explicaciones"]] == ["1", "2"]
    # Con una de más no se adivina por posición
    r = alinear_explicaciones(entidades, {"explicaciones": [{"examen": "X", "explicacion": "1"},
                                                            {"examen": "Y", "explicacion": "2"},
                                                            {"examen": "Z", "explicacion": "3"}]})
    assert all(e.get("sin_explicacion") for e in r["explicaciones"])


def test_alinear_deduplica_y_limita_las_preguntas():
    r = alinear_explicaciones([], {"preguntas_sugeridas": ["a", "a", " ", None, *map(str, range(10))]})
    assert r["preguntas_sugeridas"][0] == "a" and len(r["preguntas_sugeridas"]) == 6


# --- limpiar_entidades ---

def test_limpiar_entidades_convierte_a_texto_y_descarta_filas_vacias():
    limpias = limpiar_entidades([{"examen": "Glucosa", "valor": 126, "unidad": None, "rango_referencia": "70 - 99"},
                                 {"examen": "", "valor": "1"}, "basura", None, {"valor": "3"}])
    assert limpias == [{"examen": "Glucosa", "valor": "126", "unidad": "", "rango_referencia": "70 - 99"}]
    assert limpiar_entidades(None) == [] and limpiar_entidades({"a": 1}) == []


def test_extraccion_que_devuelve_lista_o_basura():
    b = LetraDeMedicoBackend(modo="gemini")

    class Respuesta:
        def __init__(self, texto):
            self.text = texto

    b._generar = lambda contents: Respuesta('[{"examen": "Glucosa", "valor": "90", "rango_referencia": "70-99"}]')
    assert b.step_1_and_2_multimodal_ner(b"", "image/png")["analitos"][0]["examen"] == "Glucosa"
    b._generar = lambda contents: Respuesta(None)
    assert b.step_1_and_2_multimodal_ner(b"", "image/png") == {}
    b._generar = lambda contents: Respuesta('"solo texto"')
    assert b.step_1_and_2_multimodal_ner(b"", "image/png") == {}


def test_documento_sin_analitos_utilizables():
    b = LetraDeMedicoBackend(modo="gemini")
    b.step_1_and_2_multimodal_ner = lambda *a: {"tipo_de_examen": "X", "analitos": [{"examen": ""}]}
    assert "error" in b.procesar_resultado(b"", "image/png")


# --- Generación por lotes ---

def test_examen_largo_se_genera_por_lotes_del_tamano_entrenado():
    gen = GeneradorQueOmite()
    salida = backend_propio(gen).analizar_entidades(analitos(19), "Hemograma completo")
    assert [len(l) for l in gen.lotes] == [8, 8, 3]
    assert max(len(l) for l in gen.lotes) <= LOTE_GENERADOR
    exps = salida["analisis_final"]["explicaciones"]
    assert len(exps) == 19 and not any(e.get("sin_explicacion") for e in exps)
    assert all(e["explicacion"].startswith(f"Explicación de {a['examen']}") for e, a in zip(exps, salida["datos_extraidos"]))
    assert len(salida["analisis_final"]["preguntas_sugeridas"]) <= 6
    assert salida["analisis_final"]["avisos"] == []


def test_analitos_omitidos_se_piden_otra_vez_en_un_lote_distinto():
    gen = GeneradorQueOmite(omitir={"Analito 2", "Analito 5"})
    salida = backend_propio(gen).analizar_entidades(analitos(6), "Perfil")
    assert gen.lotes == [[f"Analito {i}" for i in range(6)], ["Analito 2", "Analito 5"]]
    assert not any(e.get("sin_explicacion") for e in salida["analisis_final"]["explicaciones"])


def test_un_lote_que_falla_no_tumba_a_los_demas():
    gen = GeneradorQueOmite(falla_si_contiene="Analito 9")
    salida = backend_propio(gen).analizar_entidades(analitos(12), "Hemograma")
    exps = salida["analisis_final"]["explicaciones"]
    # El primer lote (0-7) sale bien; el segundo (8-11) falla y su reintento por mitades salva 10 y 11
    assert not any(exps[i].get("sin_explicacion") for i in range(8))
    assert exps[9]["sin_explicacion"] and not exps[11].get("sin_explicacion")
    assert any("sin explicación" in a for a in salida["analisis_final"]["avisos"])


def test_si_el_guardrail_falla_se_avisa_en_vez_de_caerse():
    class GuardrailRoto:
        def revisar(self, resultado):
            raise RuntimeError("sin RAM")

    salida = backend_propio(GeneradorQueOmite(), guardrail=GuardrailRoto()).analizar_entidades(analitos(2), "X")
    assert salida["analisis_final"]["guardrail"]["error"] == "sin RAM"


def test_si_el_rag_falla_el_generador_sigue_sin_pasajes():
    class RAGRoto:
        def recuperar(self, analitos, tipo_examen):
            raise OSError("índice corrupto")

    class GeneradorSinPasajes(GeneradorQueOmite):
        def generar(self, caso, pasajes):
            assert pasajes == []
            return {"explicaciones": [{"examen": a["examen"], "explicacion": "ok"} for a in caso["analitos"]]}

    salida = backend_propio(GeneradorSinPasajes(), rag=RAGRoto()).analizar_entidades(analitos(3), "X")
    assert salida["pasajes_rag"] == [] and salida["analisis_final"]["avisos"]


def test_guardrail_cuenta_las_oraciones_marcadas():
    class GeneradorDiagnostica(GeneradorQueOmite):
        def generar(self, caso, pasajes):
            return {"explicaciones": [{"examen": a["examen"], "explicacion": "Tu valor está dentro del rango. Esto confirma una anemia."}
                                      for a in caso["analitos"]]}

    salida = backend_propio(GeneradorDiagnostica()).analizar_entidades(analitos(2), "X")
    final = salida["analisis_final"]
    assert final["guardrail"] == {"motor": "mrbert-biomed-v2", "marcadas": 2, "error": None}
    assert all(e["explicacion"].endswith(NOTA_GUARDRAIL) for e in final["explicaciones"])


# --- Chat de seguimiento en modo propio ---

def test_chat_propio_responde_con_medlineplus_sin_las_oraciones_marcadas():
    rag = RAGFalso()
    b = backend_propio(GeneradorQueOmite(), rag=rag)
    datos = [{"examen": "Colesterol HDL", "valor": "36"}, {"examen": "Glucosa", "valor": "90"}]
    respuesta = b.responder_pregunta_seguimiento(datos, "Perfil lipídico", "¿Cómo subo mi colesterol HDL?")
    assert "colesterol bueno" in respuesta and "Se mide en sangre" in respuesta
    assert "enfermedad grave" not in respuesta
    assert "https://medlineplus.gov/spanish/x.html" in respuesta and "consulta médica" in respuesta
    assert rag.examen == "Colesterol HDL"  # el examen nombrado en la pregunta guía el reordenamiento


# --- GeneradorPropio: el prompt nunca desborda el contexto ---

class LlamaFalso:
    """Un token por carácter; registra los mensajes y el max_tokens con que se le llamó."""
    def tokenize(self, texto, add_bos=False):
        return list(texto)

    def create_chat_completion(self, messages, max_tokens, **kwargs):
        self.mensajes, self.max_tokens = messages, max_tokens
        return {"choices": [{"message": {"content": '{"explicaciones": [{"examen": "A", "explicacion": "ok"}]}'}}]}


def generador_con_contexto(n_ctx):
    g = modelos_propios.GeneradorPropio.__new__(modelos_propios.GeneradorPropio)
    g.llm, g.n_ctx = LlamaFalso(), n_ctx
    return g


def test_generador_acorta_los_pasajes_si_no_caben():
    caso = {"tipo_de_examen": "X", "paciente": {}, "analitos": [{"examen": "A", "valor": "1", "unidad": "",
                                                                 "rango_referencia": "0-2", "estado": "NORMAL"}]}
    pasajes = [{"examen": "A", "titulo": "T", "texto": "palabra " * 200}]
    g = generador_con_contexto(0)
    mitad = [{**pasajes[0], "texto": modelos_propios.recortar(pasajes[0]["texto"], modelos_propios.MAX_CARACTERES_PASAJE // 2)}]
    # Un contexto donde el pasaje completo (700) no deja sitio a la respuesta, pero la mitad sí
    g.n_ctx = g.tokens_prompt(modelos_propios.mensajes_estudiante(caso, mitad)) + modelos_propios.MIN_TOKENS_RESPUESTA + 10
    assert g.generar(caso, pasajes)["explicaciones"][0]["explicacion"] == "ok"
    assert g.llm.mensajes[1]["content"].endswith(mitad[0]["texto"])
    assert g.llm.max_tokens + g.tokens_prompt(g.llm.mensajes) <= g.n_ctx


def test_generador_lanza_error_claro_si_ni_sin_pasajes_cabe():
    caso = {"tipo_de_examen": "X", "paciente": {}, "analitos": []}
    try:
        generador_con_contexto(500).generar(caso, [])
    except ValueError as e:
        assert "contexto" in str(e)
    else:
        raise AssertionError("debía lanzar ValueError")


# --- Interfaz ---

def test_tarjeta_escapa_el_html_del_documento_y_del_generador():
    html = ui_components.render_tarjeta_html("<b>Glucosa</b>", "126", "mg/dL", "ALTO", "<200",
                                             "<script>alert(1)</script> está alto")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<b>Glucosa</b>" not in html


def test_tarjeta_sin_estado_no_se_pinta_de_normal():
    html = ui_components.render_tarjeta_html("Ferritina", "abc", "", "DESCONOCIDO", "30 - 400", "x")
    assert "ldm-card-desconocido" in html and "SIN EVALUAR" in html and "NORMAL" not in html
    assert "ldm-gauge-box" not in html


def test_indicador_entiende_los_mismos_rangos_que_la_regla():
    assert "Rango normal: 7 - 56" in ui_components.calcular_gauge_html("30", "7 a 56", "NORMAL")
    assert "Deseable (&lt; 200)" in ui_components.calcular_gauge_html("150", "Hasta 200", "NORMAL")
    assert "Deseable (&gt; 40)" in ui_components.calcular_gauge_html("50", "Mayor de 40", "NORMAL")
    assert "Rango normal: 4.5 - 11" in ui_components.calcular_gauge_html("6", "4,5 – 11,0", "NORMAL")
    assert normalizar_rango("Menor de 5,7") == "<5.7" and normalizar_rango("≥ 40") == ">=40"


def test_ficha_incluye_los_valores_sin_evaluar():
    ficha = ui_components.generar_ficha_medica_txt("X", [{"examen": "Ferritina", "valor": "abc", "unidad": "",
                                                          "rango_referencia": "30-400", "estado": "DESCONOCIDO"}], [])
    assert "[SIN EVALUAR] Ferritina" in ficha
