"""
Regresiones del primer PDF real probado en la app (laboratorio Compensar, 8 analitos, 2026-11-02).

Errores que salieron con ese documento y que estas pruebas impiden que vuelvan:
- "GLUCOSA EN SUERO U OTRO FLUIDO DIFERENTE A ORINA" recuperaba la página de glucosa en orina.
- La SOMATOMEDINA C (IGF-1), que no está en la base, recibía la página de creatinina, y el generador la
  explicó como creatinina y como "normal" cuando la regla la calculó ALTA.
- Una pregunta sugerida decía "resultado alto en la hormona estimulante del tiroides" con la TSH normal.
Además: el generador sin gramática JSON (con reintento) y la ficha en PDF e imagen.
Sin modelos reales ni Gemini.
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import ficha_consulta  # noqa: E402
import modelos_propios as mp  # noqa: E402
from backend import LetraDeMedicoBackend, explicacion_plantilla, revisar_coherencia  # noqa: E402

CATALOGO = json.loads((RAIZ / "training" / "sintetico" / "catalogo_analitos.json").read_text(encoding="utf-8"))
ALIAS = mp.indice_alias(CATALOGO)
ORINA = set(CATALOGO["paneles"]["Análisis de Orina Completo"])

PDF = [
    ("GLUCOSA EN SUERO U OTRO FLUIDO DIFERENTE A ORINA", "76.7", "mg/dl", "60.0 - 100.0"),
    ("COLESTEROL TOTAL", "161.5", "mg/dl", "100.0 - 200.0"),
    ("COLESTEROL DE ALTA DENSIDAD", "57.0", "mg/dl", "Mayor de 55"),
    ("COLESTEROL DE BAJA DENSIDAD LDL SEMIAUTOMATIZADO", "90.3", "mg/dl", "0.0 - 100.0"),
    ("TRIGLICERIDOS", "71.0", "mg/dl", "4.0 - 200.0"),
    ("HORMONA ESTIMULANTE DEL TIROIDES ULTRASENSIBLE", "3.260", "uUI/ml", "0.510 - 4.300"),
    ("TIROXINA LIBRE", "1.46", "ng/dl", "0.98 - 1.63"),
    ("SOMATOMEDINA C FACTOR I DE CRECIMIENTO SIMILAR A LA INSULINA O I", "420.0", "ng/ml", "173.0 - 414.0"),
]


def analitos_pdf():
    filas = [dict(zip(("examen", "valor", "unidad", "rango_referencia"), f)) for f in PDF]
    for a in filas:
        a["estado"] = LetraDeMedicoBackend.evaluar_estado_analito(a["valor"], a["rango_referencia"])
    return filas


def test_estados_del_pdf_real():
    assert [a["estado"] for a in analitos_pdf()] == ["NORMAL"] * 7 + ["ALTO"]


# --- Alias del catálogo dentro de los nombres largos de laboratorio ---

def test_alias_de_los_nombres_del_pdf():
    esperado = ["glucosa", "colesterol_total", "colesterol_hdl", "colesterol_ldl", "trigliceridos", "tsh", "t4_libre"]
    for (nombre, *_), clave in zip(PDF, esperado):
        assert mp.analitos_del_nombre(nombre, ALIAS, ORINA) == {clave}, nombre
    assert mp.analitos_del_nombre(PDF[7][0], ALIAS, ORINA) == set()  # IGF-1 no está en la base


def test_alias_glucosa_decide_por_sangre_u_orina():
    assert mp.analitos_del_nombre("Glucosa", ALIAS, ORINA, "Análisis de Orina Completo") == {"glucosa_orina"}
    assert mp.analitos_del_nombre("Glucosa", ALIAS, ORINA, "Control de Diabetes") == {"glucosa"}
    assert mp.analitos_del_nombre("Glucosa en orina", ALIAS, ORINA) == {"glucosa_orina"}


def test_alias_son_palabras_completas():
    # "CREA" es alias de creatinina, pero no debe encontrarse dentro de "Creatina quinasa"
    assert mp.analitos_del_nombre("Creatina quinasa CK", ALIAS, ORINA) == set()
    assert mp.tokens("Hormona estimulante DEL tiroides") == ["hormona", "estimulante", "tiroides"]


class RAGConAlias(mp.RecuperadorRAG):
    """El RecuperadorRAG real sin modelo: el índice devuelve un orden fijo de documentos con su similitud."""
    def __init__(self, documentos, similitudes):
        self.documentos, self.similitudes = documentos, similitudes
        self.alias, self.claves_orina, self.usa_prompt = ALIAS, ORINA, False

        class Modelo:
            def encode(self, consultas, **kwargs):
                return [[0.0]]

        class Indice:
            def search(_, vector, k):
                return [self.similitudes[:k]], [list(range(len(self.documentos)))[:k]]

        self.modelo, self.indice = Modelo(), Indice()


DOCS = [
    {"titulo": "Prueba de glucosa en la orina", "analitos": ["glucosa_orina"], "texto": "orina", "url": "u1"},
    {"titulo": "Prueba de creatinina", "analitos": ["creatinina"], "texto": "creatinina", "url": "u2"},
    {"titulo": "Prueba de glucosa en la sangre", "analitos": ["glucosa"], "texto": "sangre", "url": "u3"},
]


def test_rag_prefiere_el_analito_del_alias_aunque_no_sea_el_primero():
    rag = RAGConAlias(DOCS, [0.9, 0.8, 0.5])
    doc = rag.buscar("x", examen=PDF[0][0], tipo_examen="Resultados de Laboratorio Clínico")
    assert doc["titulo"] == "Prueba de glucosa en la sangre"


def test_rag_sin_alias_y_poca_similitud_no_devuelve_guia():
    rag = RAGConAlias(DOCS[1:], [mp.UMBRAL_SIN_ALIAS - 0.01, 0.3])
    assert rag.buscar("x", examen=PDF[7][0]) is None
    pasaje = rag.recuperar([{"examen": PDF[7][0]}], "Laboratorio")[0]
    assert pasaje["sin_cobertura"] and pasaje["texto"] == ""
    # Con similitud alta sí se acepta el primero
    assert RAGConAlias(DOCS[1:], [mp.UMBRAL_SIN_ALIAS + 0.01, 0.3]).buscar("x", examen="Examen raro")


def test_generador_no_recibe_pasajes_vacios_y_la_app_avisa():
    class RAGSinIGF:
        def recuperar(self, analitos, tipo_examen):
            return [{"examen": a["examen"], "titulo": "", "texto": "", "sin_cobertura": True}
                    if a["examen"].startswith("SOMATOMEDINA") else
                    {"examen": a["examen"], "titulo": "T", "texto": "Texto."} for a in analitos]

    class Generador:
        def generar(self, caso, pasajes):
            self.pasajes = pasajes
            return {"explicaciones": [{"examen": a["examen"], "explicacion": "Está dentro del rango normal."}
                                      for a in caso["analitos"]]}

    class Guardrail:
        def revisar(self, r):
            for e in r["explicaciones"]:
                e["oraciones_marcadas"] = []
            return r

    gen = Generador()
    b = LetraDeMedicoBackend(modo="propio", modelos={"rag": RAGSinIGF(), "generador": gen, "guardrail": Guardrail()})
    salida = b.analizar_entidades([{k: a[k] for k in ("examen", "valor", "unidad", "rango_referencia")}
                                   for a in analitos_pdf()], "Laboratorio")
    assert len(gen.pasajes) == 7 and all(p["texto"] for p in gen.pasajes)
    assert any("SOMATOMEDINA" in a for a in salida["analisis_final"]["avisos"])


# --- Revisión de coherencia ---

def test_explicacion_que_contradice_el_estado_se_reemplaza():
    analitos = analitos_pdf()
    resultado = {"explicaciones": [{"examen": a["examen"], "explicacion": "Tu resultado está dentro del rango normal."}
                                   for a in analitos],
                 "preguntas_sugeridas": []}
    # El caso real: IGF-1 ALTA explicada como creatinina normal
    resultado["explicaciones"][7]["explicacion"] = ("La prueba de creatinina mide la cantidad de creatinina en tu "
                                                    "sangre. Un resultado normal indica que tu nivel está bien.")
    avisos = revisar_coherencia(analitos, resultado)
    igf = resultado["explicaciones"][7]
    assert igf["explicacion_plantilla"] and "por encima del rango de referencia (173.0 - 414.0)" in igf["explicacion"]
    assert "creatinina" not in igf["explicacion"]
    assert not any(e.get("explicacion_plantilla") for e in resultado["explicaciones"][:7])
    assert avisos and "1 explicación" in avisos[0]


def test_preguntas_incoherentes_se_quitan():
    analitos = analitos_pdf()
    buenas = ["¿Qué significa que mi IGF-1 esté un poco elevado?",
              "¿Qué factores pueden afectar mis niveles de colesterol?",
              "¿Necesito repetir algún examen?"]
    malas = ["¿Qué significa un resultado alto en la hormona estimulante del tiroides?",  # TSH normal
             "¿Qué significa que mi nivel de creatinina esté dentro del rango normal?",   # no se la hizo
             "¿Por qué tengo el colesterol HDL bajo?"]                                     # HDL normal
    resultado = {"explicaciones": [{"examen": a["examen"], "explicacion": "dentro del rango"} for a in analitos],
                 "preguntas_sugeridas": buenas + malas}
    revisar_coherencia(analitos, resultado)
    assert resultado["preguntas_sugeridas"] == buenas


def test_preguntas_casi_repetidas_se_quitan():
    # Salida real del generador con el PDF de prueba (2026-11-02): la misma pregunta, cada vez más larga
    analitos = analitos_pdf()
    base = "¿Qué factores podrían estar influyendo en este resultado de la somatomedin C?"
    resultado = {"explicaciones": [{"examen": a["examen"], "explicacion": "dentro del rango"} for a in analitos],
                 "preguntas_sugeridas": [base, base[:-1] + " en un paciente de 15 años?",
                                         "¿Es necesario repetir el análisis o hacer algún estudio adicional?"]}
    revisar_coherencia(analitos, resultado)
    assert resultado["preguntas_sugeridas"] == [base, "¿Es necesario repetir el análisis o hacer algún estudio adicional?"]


def test_preguntas_parecidas_de_examenes_distintos_se_conservan():
    # Segundo PDF real (química, 2026-11-02): el filtro quitaba AST y ALT por "casi iguales" y dejaba una pregunta
    analitos = [
        {"examen": "COLESTEROL LDL - NO CALCULADO", "valor": "102.37", "unidad": "mg/dl", "rango_referencia": "Menor de 100 mg/dl", "estado": "ALTO"},
        {"examen": "TRIGLICERIDOS", "valor": "209.00", "unidad": "mg/dl", "rango_referencia": "Menor de 150 mg/dl", "estado": "ALTO"},
        {"examen": "ASPARTATO AMINO TRANSFERASA (ASAT - TGO)", "valor": "35.60", "unidad": "U/l", "rango_referencia": "0.00 - 32.00", "estado": "ALTO"},
        {"examen": "ALANINO AMINO TRANSFERASA (ALAT - TGP)", "valor": "28.70", "unidad": "U/l", "rango_referencia": "0.00 - 31.00", "estado": "NORMAL"},
    ]
    preguntas = ["¿Qué factores podrían estar influyendo en mis niveles de colesterol LDL y triglicéridos?",
                 "¿Qué factores podrían estar influyendo en mis niveles de AST?",
                 "¿Qué factores podrían estar influyendo en mis niveles de ALT?",
                 "¿Qué factores podrían estar influyendo en mis niveles de ALAT?"]  # ALAT es la misma ALT
    resultado = {"explicaciones": [{"examen": a["examen"], "explicacion": "Está por encima del rango normal."}
                                   for a in analitos], "preguntas_sugeridas": preguntas}
    revisar_coherencia(analitos, resultado)
    assert resultado["preguntas_sugeridas"] == preguntas[:3]
    assert mp.analitos_del_nombre(analitos[2]["examen"], ALIAS, ORINA) == {"ast"}
    assert mp.analitos_del_nombre(analitos[3]["examen"], ALIAS, ORINA) == {"alt"}


def test_si_no_queda_ninguna_pregunta_se_dan_las_de_respaldo():
    analitos = analitos_pdf()[:1]
    resultado = {"explicaciones": [{"examen": analitos[0]["examen"], "explicacion": "normal"}],
                 "preguntas_sugeridas": ["¿Por qué mi glucosa está alta?"]}
    revisar_coherencia(analitos, resultado)
    assert len(resultado["preguntas_sugeridas"]) == 2 and "glucosa" not in " ".join(resultado["preguntas_sugeridas"])


def test_plantilla_por_estado():
    a = {"examen": "TSH", "valor": "3.2", "unidad": "uUI/ml", "rango_referencia": "0.5 - 4.3", "estado": "NORMAL"}
    assert "3.2 uUI/ml, dentro del rango de referencia (0.5 - 4.3)" in explicacion_plantilla(a)
    assert "fuera de lo esperado" in explicacion_plantilla({**a, "valor": "1+", "estado": "ANORMAL"})


# --- Generador: sin gramática y con reintento ---

class LlamaDosRespuestas:
    def __init__(self, respuestas):
        self.respuestas, self.llamadas = list(respuestas), []

    def tokenize(self, texto, add_bos=False):
        return list(range(len(texto) // 4))

    def create_chat_completion(self, messages, max_tokens, **kwargs):
        self.llamadas.append({k: v for k, v in kwargs.items() if k == "response_format"})
        return {"choices": [{"message": {"content": self.respuestas.pop(0)}}]}


def generador(respuestas):
    g = mp.GeneradorPropio.__new__(mp.GeneradorPropio)
    g.llm, g.n_ctx = LlamaDosRespuestas(respuestas), 4096
    return g


CASO = {"tipo_de_examen": "X", "paciente": {}, "analitos": [
    {"examen": "A", "valor": "1", "unidad": "", "rango_referencia": "0-2", "estado": "NORMAL"}]}
VALIDO = '{"explicaciones": [{"examen": "A", "explicacion": "ok"}], "preguntas_sugeridas": []}'


def test_generador_sin_gramatica_si_el_json_es_valido():
    g = generador([VALIDO])
    assert g.generar(CASO, [])["explicaciones"][0]["explicacion"] == "ok"
    assert g.llm.llamadas == [{}]  # una sola llamada, sin response_format


def test_generador_repite_con_gramatica_si_el_json_sale_mal():
    g = generador(['{"explicaciones": [{"examen": "A", "explic', VALIDO])
    assert g.generar(CASO, [])["explicaciones"][0]["explicacion"] == "ok"
    assert g.llm.llamadas == [{}, {"response_format": {"type": "json_object"}}]


def test_capas_gpu_se_pueden_forzar(monkeypatch):
    monkeypatch.setenv("LDM_GPU_CAPAS", "0")
    assert mp.capas_gpu_por_defecto() == 0


# --- Ficha en PDF e imagen ---

def datos_ficha():
    analitos = analitos_pdf()
    analitos[2]["rango_referencia"], analitos[2]["estado"] = "", "DESCONOCIDO"
    explicaciones = [{"examen": a["examen"], "explicacion": f"Explicación {i} 🙂."} for i, a in enumerate(analitos)]
    return dict(tipo_examen="Resultados de Laboratorio Clínico", analitos=analitos, explicaciones=explicaciones,
                preguntas=["¿Qué significa mi IGF-1 elevado?"], contexto_paciente="Control.")


def test_contenido_ficha_ordena_y_limpia():
    c = ficha_consulta.contenido_ficha(**datos_ficha())
    assert [f["estado"] for f in c["filas"]][:2] == ["DESCONOCIDO", "ALTO"]  # primero lo que hay que comentar
    assert len(c["comentar"]) == 2 and c["filas"][2]["estado"] == "NORMAL"
    assert all("🙂" not in f["explicacion"] and not f["explicacion"].endswith(" .") for f in c["filas"])
    assert c["filas"][0]["referencia"] == "No indicada"


def test_ficha_pdf_es_un_pdf_con_el_texto():
    pdf = ficha_consulta.ficha_pdf(**datos_ficha())
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000


def test_ficha_png_es_una_imagen_vertical():
    import io

    from PIL import Image
    imagen = Image.open(io.BytesIO(ficha_consulta.ficha_png(**datos_ficha())))
    assert imagen.format == "PNG" and imagen.width == 1080 and imagen.height > imagen.width


def test_ficha_sin_explicaciones_alineadas_no_falla():
    datos = datos_ficha()
    datos["explicaciones"] = []
    assert ficha_consulta.ficha_pdf(**datos).startswith(b"%PDF")
    assert ficha_consulta.ficha_png(**datos)
