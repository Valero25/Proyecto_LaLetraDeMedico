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
# Bandera antes del valor: "* 35.60" (Colsanitas, 2011)
RE_BANDERA_VALOR = re.compile(r"^(?:[*↑↓]|\(\s*[HL]\s*\))\s*([<>≤≥]?\s*[-+]?\d+(?:[.,]\d+)?)$")
# Rango como mínimo y máximo en columnas, sin guion: "0.00   31.00" -> "0.00 - 31.00"
RE_MIN_MAX = re.compile(r"^([<>]?\d+(?:[.,]\d+)?)\s+(\d+(?:[.,]\d+)?)$")
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
    return bool(RE_VALOR.match(t) or RE_VALOR_BANDERA.match(t) or RE_BANDERA_VALOR.match(t)
                or re.fullmatch(r"\d\s*\+|\++", t)) or sin_tildes(t) in QUALITATIVOS


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
    valor = RE_BANDERA_VALOR.sub(r"\1", RE_VALOR_BANDERA.sub(r"\1", partes[i].strip()))
    # Las banderas ("H", "L", "*") pueden ir entre el valor y la unidad: se quitan antes de buscar la unidad
    resto = [c for c in partes[i + 1:] if not RE_BANDERA.match(c.strip())]
    unidad = ""
    if resto and es_unidad(resto[0]) and not RE_VALOR.match(resto[0]):
        unidad, resto = resto[0], resto[1:]
    elif resto:
        # La unidad y el rango en la misma celda: "U/l 0.00 31.00"
        m = re.match(r"^(\S+)\s+(.*\d.*)$", resto[0])
        if m and es_unidad(m.group(1)) and not RE_VALOR.match(m.group(1)):
            unidad, resto = m.group(1), [m.group(2)] + resto[1:]
    resto = [c for c in resto if not RE_BANDERA.match(c.strip())]
    rango = " ".join(resto).strip()
    rango = RE_MIN_MAX.sub(r"\1 - \2", rango)
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
    """Solo sexo y edad. Acepta "Edad: 15 Años", "Edad   Años  72", "Género: Masculino", "Sexo: F"."""
    t = sin_tildes(texto)
    paciente = {}
    m = re.search(r"\bedad\b[^\d\n]{0,25}?(\d{1,3})\b", t)
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
            # Notas sobre el rango ("** VALORES DE REFERENCIA SEGÚN LA NCEP ATP III - 2001"): se saltan sin cortar
            # el resultado actual, porque sus categorías vienen justo debajo
            if "valores de referencia" in sin_tildes(limpia) or re.match(r"^\*+\s*[A-Za-z]", limpia):
                continue
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
                    print(f"Lector local: resultado sin nombre en la línea {limpia!r}", flush=True)
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


def texto_por_posicion(pagina, ancho_caracter=5.0, tolerancia_y=2.5):
    """
    Rearma las líneas de la página con la posición (x, y) de cada fragmento de texto, en el orden en que se ven.

    Hace falta cuando el sistema del laboratorio escribe los fragmentos en otro orden: en un PDF real (Colsanitas,
    2011) el modo layout de pypdf daba "mg/dl102.37" y "U/l 0.00 31.0028.70", porque el valor se escribía después
    de la unidad y el rango. Con las posiciones sale "COLESTEROL LDL   102.37   mg/dl" y "28.70   U/l 0.00 31.00".
    Cada fragmento se coloca en la columna x / ancho_caracter, así la sangría se conserva como en el modo layout.
    """
    fragmentos = []

    def visitor(texto, cm, tm, fuente, tam):
        if texto and texto.strip():
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            fragmentos.append((y, x, " ".join(texto.split("\n")).strip()))

    pagina.extract_text(visitor_text=visitor)
    filas = []  # [(y, [(x, texto)])], de arriba a abajo
    for y, x, t in sorted(fragmentos, key=lambda f: -f[0]):
        if filas and abs(filas[-1][0] - y) < tolerancia_y:
            filas[-1][1].append((x, t))
        else:
            filas.append((y, [(x, t)]))
    lineas = []
    for _, celdas_fila in filas:
        linea = ""
        for x, t in sorted(celdas_fila):
            columna = max(0, int(x / ancho_caracter))
            linea += " " * (columna - len(linea)) if columna > len(linea) + 1 else ("   " if linea else "")
            linea += t
        lineas.append(linea)
    return "\n".join(lineas)


def leer_pdf(file_bytes, minimo_caracteres=80):
    """
    {"tipo_de_examen", "paciente", "analitos"} en el mismo formato que la extracción de Gemini, o None si el PDF
    no tiene texto (escaneado), no se pudo abrir o la lectura no es confiable.

    Primero se lee en modo layout de pypdf (validado con el PDF real de Compensar y 450 PDFs sintéticos). Si no
    sale una lectura confiable, se intenta otra vez rearmando las líneas por posición (`texto_por_posicion`).
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
        print(f"Lector local: el PDF no tiene capa de texto ({len(lector.pages)} páginas): parece escaneado", flush=True)
        return None
    analitos, secciones, confiable = analizar_texto(paginas)
    if not analitos or not confiable:
        print("Lector local: el modo layout no dio una lectura confiable; se rearman las líneas por posición", flush=True)
        try:
            por_posicion = [texto_por_posicion(p) for p in lector.pages]
        except Exception as e:
            print(f"Lector local: no se pudieron leer las posiciones del texto: {e}", flush=True)
            return None
        analitos, secciones, confiable = analizar_texto(por_posicion)
        texto = "\n".join(por_posicion)
    if not analitos:
        print("Lector local: no se reconoció ningún analito en el texto del PDF (diseño desconocido)", flush=True)
        return None
    if not confiable:
        return None
    return {"tipo_de_examen": tipo_de_examen(secciones), "paciente": paciente_del_texto(texto), "analitos": analitos}
