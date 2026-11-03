"""
Lectura local de PDFs digitales de laboratorio, sin Gemini ni internet.

Un PDF generado por el sistema del laboratorio (no escaneado) trae una capa de texto: pypdf la extrae en modo
"layout", que conserva las columnas, y aquí se reconocen los analitos con reglas. Así el motor principal (los
modelos propios) no depende de Gemini para leer el documento. Gemini queda como segunda opción para fotos,
PDFs escaneados o diseños que este lector no entiende.

Diseños que reconoce:
- Por bloques (ej. Compensar): el nombre en una o dos líneas y debajo "Resultado: 76.7  mg/dl  60.0 - 100.0",
  seguido opcionalmente de categorías ("Sin riesgo: Menor de 200.0 mg/dl").
- Por tabla: "GLUCOSA    95    mg/dL    70 - 110" en una sola fila.

Si no hay texto o la lectura no es confiable (un resultado sin nombre, ningún analito), `leer_pdf` devuelve None
y el backend pasa a Gemini. Del paciente solo se extraen el sexo y la edad, nunca el nombre ni el documento.
"""
import io
import re
import unicodedata

QUALITATIVOS = {"negativo", "positivo", "reactivo", "no reactivo", "normal", "anormal", "ausente", "presente",
                "trazas", "no se observan", "+", "++", "+++", "++++"}
# Líneas que no son analitos (encabezados, pie de página, datos del paciente, notas del método), sin tildes
RUIDO = ("metodo", "tecnica", "profesional responsable", "pagina", "paciente", "historia", "edad", "medico", "fecha",
         "servicio", "sede", "no ingreso", "no orden", "orden", "telefono", "examen", "el paciente se obliga",
         "cubiertos por", "resultados de laboratorio", "reference ranges", "intervalo biologico", "observacion",
         "nota", "valido", "validado", "firma", "documento", "direccion", "nit", "genero", "sexo", "cama",
         "impreso", "reporte", "informe", "analito", "prueba", "laboratorio clinico", "examenes procesados")
SECCIONES_ORINA = ("orina", "uroanalisis", "urianalisis", "parcial de orina", "sedimento")
SECCIONES_OTRAS = ("coprologico", "coproscopico", "heces", "materia fecal", "espermograma", "liquido")
RE_VALOR = re.compile(r"^[<>≤≥]?\s*[-+]?\d+(?:[.,]\d+)?$")
RE_VALOR_BANDERA = re.compile(r"^([<>≤≥]?\s*[-+]?\d+(?:[.,]\d+)?)\s*([*HLhl↑↓]|\(\s*[HL]\s*\))$")
RE_BANDERA = re.compile(r"^([*HLhl↑↓]|\(\s*[HL]\s*\)|alto|bajo)$", re.I)
RE_UNIDAD = re.compile(r"^(?=.*[A-Za-zµμ%])[A-Za-zµμ%/^\d.*³²⁶×x]{1,18}$")
RE_CATEGORIA = re.compile(r"^[A-Za-zÁÉÍÓÚáéíóúñÑ][A-Za-zÁÉÍÓÚáéíóúñÑ ]{1,30}:\s*\S.*\d")
PALABRAS_NO_UNIDAD = {"normal", "negativo", "positivo", "alto", "bajo", "hasta", "menor", "mayor", "de", "a",
                      "resultado", "reactivo"}


def sin_tildes(texto):
    texto = unicodedata.normalize("NFD", str(texto or "").lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def es_ruido(linea):
    """Palabras completas: "NIT: 900..." es ruido, pero "Nitritos" y "Nitrógeno ureico" no."""
    t = sin_tildes(linea).strip(" :·-")
    return any(re.match(rf"{re.escape(r)}\b", t) for r in RUIDO) or bool(re.match(r"^(pag|pagina)\s*\d", t))


def es_valor(celda):
    t = celda.strip()
    # Cruces ("1+", "3+", "++") y trazas: valores cualitativos de orina, fuera del alcance pero se muestran
    return bool(RE_VALOR.match(t) or RE_VALOR_BANDERA.match(t) or re.fullmatch(r"\d\s*\+|\++", t)) \
        or sin_tildes(t) in QUALITATIVOS


def es_unidad(celda):
    t = celda.strip()
    return bool(RE_UNIDAD.match(t)) and sin_tildes(t) not in PALABRAS_NO_UNIDAD and not RE_BANDERA.match(t)


def celdas(linea):
    """Columnas de una línea en modo layout: separadas por 2 o más espacios."""
    return [c for c in re.split(r"\s{2,}", linea.strip()) if c]


def sangria(linea):
    return len(linea) - len(linea.lstrip(" "))


def leer_resultado(partes):
    """
    Si las celdas forman un resultado, devuelve (nombre_en_la_linea, valor, unidad, rango); si no, None.
    "Resultado:" al inicio significa que el nombre está en las líneas de arriba.
    """
    if partes and sin_tildes(partes[0]).rstrip(": ") == "resultado":
        partes = partes[1:]
        prefijo = []
    else:
        prefijo = None
    # El valor y la unidad a veces vienen juntos en una celda: "76.7 mg/dl"
    expandidas = []
    for c in partes:
        m = re.match(r"^([<>≤≥]?\s*\d+(?:[.,]\d+)?)\s+(\S+)$", c)
        expandidas += [m.group(1), m.group(2)] if m and es_unidad(m.group(2)) else [c]
    partes = expandidas
    i = next((k for k, c in enumerate(partes) if es_valor(c)), None)
    if i is None:
        return None
    nombre = " ".join(partes[:i]) if prefijo is None else ""
    if prefijo is None and not re.search(r"[A-Za-zÁ-úñÑ]", nombre):
        nombre = ""  # en tablas el nombre puede venir en la línea anterior
    valor = RE_VALOR_BANDERA.sub(r"\1", partes[i].strip())
    # Las banderas ("H", "L", "*") pueden ir entre el valor y la unidad: se quitan antes de buscar la unidad
    resto = [c for c in partes[i + 1:] if not RE_BANDERA.match(c.strip())]
    unidad = ""
    if resto and es_unidad(resto[0]) and not RE_VALOR.match(resto[0]):
        unidad, resto = resto[0], resto[1:]
    resto = [c for c in resto if not RE_BANDERA.match(c.strip())]
    rango = " ".join(resto).strip()
    cualitativo = sin_tildes(valor) in QUALITATIVOS or bool(re.fullmatch(r"\d\s*\+|\++", valor))
    # Un número suelto sin unidad ni rango no es un resultado (ej. un número de orden en el encabezado)
    if not cualitativo and not unidad and not re.search(r"\d", rango):
        return None
    if cualitativo and not rango and prefijo is None and not nombre:
        return None
    return nombre, valor, unidad, rango


def muestra_de_seccion(seccion):
    s = sin_tildes(seccion)
    if any(p in s for p in SECCIONES_ORINA):
        return "orina"
    if any(p in s for p in SECCIONES_OTRAS):
        return "otra"
    return "sangre" if s else ""


def paciente_del_texto(texto):
    """Solo sexo y edad. Acepta "Edad: 15 Años", "Género: Masculino", "Sexo: F"."""
    t = sin_tildes(texto)
    paciente = {}
    m = re.search(r"\bedad\s*:?\s*(\d{1,3})\s*(?:anos|a\b|años)?", t)
    if m:
        paciente["edad"] = int(m.group(1))
    m = re.search(r"\b(?:genero|sexo)\s*:?\s*(masculino|femenino|hombre|mujer|m\b|f\b)", t)
    if m:
        paciente["sexo"] = "F" if m.group(1).startswith(("f", "mujer")) else "M"
    return paciente


def es_seccion(linea, partes):
    """Un título de sección: en mayúsculas, sin números, corto y centrado (con mucha sangría)."""
    texto = linea.strip()
    return (len(partes) == 1 and texto.isupper() and not re.search(r"\d", texto) and len(texto.split()) <= 5
            and sangria(linea) >= 15 and not es_ruido(texto))


def analizar_texto(paginas):
    """Analitos a partir del texto en modo layout de cada página. Devuelve (analitos, secciones, confiable)."""
    analitos, secciones = [], []
    nombre_pendiente, seccion, actual = [], "", None
    confiable = True
    for texto in paginas:
        for linea in texto.splitlines():
            if not linea.strip():
                continue
            partes = celdas(linea)
            if es_seccion(linea, partes):
                seccion = linea.strip()
                if seccion not in secciones:
                    secciones.append(seccion)
                nombre_pendiente, actual = [], None
                continue
            limpia = " ".join(linea.split())
            # Categorías del rango, debajo de un resultado: "Sin riesgo: Menor de 200.0 mg/dl"
            if actual is not None and RE_CATEGORIA.match(limpia) and not es_ruido(limpia):
                previo = actual["rango_referencia"].rstrip(" .")
                actual["rango_referencia"] = (previo + ". " if previo else "") + limpia.rstrip(" .")
                continue
            if es_ruido(limpia):
                nombre_pendiente, actual = [], None
                continue
            resultado = leer_resultado(partes)
            if resultado:
                nombre, valor, unidad, rango = resultado
                nombre = nombre or " ".join(nombre_pendiente)
                nombre = " ".join(nombre.split()).strip(" :.")
                # Basta una letra: los laboratorios usan abreviaturas como "K", "Na" o "Hb"
                if not re.search(r"[A-Za-zÁ-úñÑ]", nombre):
                    confiable = False  # un resultado sin nombre: mejor que lo lea Gemini
                    nombre_pendiente, actual = [], None
                    continue
                actual = {"examen": nombre, "valor": valor, "unidad": unidad, "rango_referencia": rango,
                          "muestra": muestra_de_seccion(seccion)}
                analitos.append(actual)
                nombre_pendiente = []
                continue
            # Una línea de nombre: poca sangría, con letras, sin dos puntos
            if sangria(linea) < 10 and ":" not in limpia and re.search(r"[A-Za-zÁ-úñÑ]", limpia):
                actual = None
                nombre_pendiente = (nombre_pendiente + [limpia])[-3:]
            else:
                nombre_pendiente, actual = [], None
    return analitos, secciones, confiable


def tipo_de_examen(secciones):
    if not secciones:
        return "Resultados de Laboratorio Clínico"
    nombres = [s.capitalize() for s in secciones]
    return "Resultados de Laboratorio Clínico (" + (" y ".join(nombres) if len(nombres) <= 2
                                                    else ", ".join(nombres[:-1]) + " y " + nombres[-1]) + ")"


def leer_pdf(file_bytes, minimo_caracteres=80):
    """
    {"tipo_de_examen", "paciente", "analitos"} en el mismo formato que la extracción de Gemini, o None si el PDF
    no tiene texto (escaneado), no se pudo abrir o la lectura no es confiable.
    """
    from pypdf import PdfReader
    try:
        lector = PdfReader(io.BytesIO(file_bytes))
        paginas = [p.extract_text(extraction_mode="layout") or "" for p in lector.pages]
    except Exception as e:
        print(f"No se pudo leer el PDF en local: {e}", flush=True)
        return None
    texto = "\n".join(paginas)
    if len(re.sub(r"\s", "", texto)) < minimo_caracteres:
        return None  # PDF escaneado: no tiene capa de texto
    analitos, secciones, confiable = analizar_texto(paginas)
    if not analitos or not confiable:
        return None
    return {"tipo_de_examen": tipo_de_examen(secciones), "paciente": paciente_del_texto(texto), "analitos": analitos}
