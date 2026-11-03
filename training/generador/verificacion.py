"""
Fase 4: verificaciones automáticas de una respuesta del generador.

Se usan dos veces:
- para filtrar las respuestas del profesor antes de entrenar (dataset.py), y
- para evaluar al modelo base, al ajustado y a Gemini sobre el test (evaluacion/evaluar_generador.py).

Todas son baratas y deterministas, salvo la de guardrails, que usa el clasificador de la Fase 3.
"""
import json
import re
import unicodedata

PALABRAS_ESTADO = {
    "ALTO": ["alto", "alta", "elevad", "por encima", "superior", "mayor"],
    "BAJO": ["bajo", "baja", "disminuid", "por debajo", "inferior", "menor", "reducid"],
    "NORMAL": ["normal", "dentro del rango", "dentro de lo esperado", "dentro de los valores", "adecuad", "esperado"],
    "ANORMAL": ["positiv", "presencia", "detect", "anormal", "fuera de lo esperado", "trazas", "cruces"],
}


def _sin_tildes(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")


def parsear(texto):
    """Devuelve el JSON de la respuesta o None. Tolera bloques ```json alrededor."""
    texto = texto.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        datos = json.loads(texto)
    except json.JSONDecodeError:
        return None
    if not isinstance(datos, dict) or not isinstance(datos.get("explicaciones"), list):
        return None
    if not all(isinstance(e, dict) and isinstance(e.get("explicacion"), str) and e.get("examen") for e in datos["explicaciones"]):
        return None
    if not isinstance(datos.get("preguntas_sugeridas", []), list):
        return None
    return datos


def dividir_oraciones(texto):
    return [o.strip() for o in re.split(r"(?<=[.!?])\s+", texto) if len(o.strip()) > 3]


def menciona_valor(explicacion, analito):
    """El número del resultado aparece tal cual, con punto o con coma decimal; en cualitativos, su palabra."""
    valor = analito["valor"].strip()
    texto = _sin_tildes(explicacion)
    if re.fullmatch(r"[\d.,]+", valor):
        return any(v in texto for v in {valor, valor.replace(",", "."), valor.replace(".", ",")})
    return _sin_tildes(valor).rstrip("+").split()[0] in texto


def estado_consistente(explicacion, analito):
    texto = _sin_tildes(explicacion)
    return any(_sin_tildes(p) in texto for p in PALABRAS_ESTADO.get(analito["estado"], []))


def verificar(texto, caso):
    """Chequeos de una respuesta frente a su caso. No incluye guardrails (ver frases_para_guardrails)."""
    datos = parsear(texto)
    if datos is None:
        return {"json_valido": False}
    por_examen = {_sin_tildes(e["examen"]): e["explicacion"] for e in datos["explicaciones"]}
    detalle = []
    for a in caso["analitos"]:
        explicacion = por_examen.get(_sin_tildes(a["examen"]))
        detalle.append({
            "examen": a["examen"], "cubierto": explicacion is not None,
            "menciona_valor": bool(explicacion) and menciona_valor(explicacion, a),
            "estado_consistente": bool(explicacion) and estado_consistente(explicacion, a),
        })
    n = len(detalle)
    preguntas = [p for p in datos.get("preguntas_sugeridas", []) if isinstance(p, str) and p.strip()]
    return {
        "json_valido": True,
        "cobertura": sum(d["cubierto"] for d in detalle) / n,
        "menciona_valor": sum(d["menciona_valor"] for d in detalle) / n,
        "estado_consistente": sum(d["estado_consistente"] for d in detalle) / n,
        "examenes_extra": len(por_examen) - sum(d["cubierto"] for d in detalle),
        "preguntas": len(preguntas),
        "detalle": detalle,
    }


def frases_para_guardrails(texto):
    """Oraciones de las explicaciones, para pasarlas por el clasificador de la Fase 3."""
    datos = parsear(texto)
    if datos is None:
        return []
    return [o for e in datos["explicaciones"] for o in dividir_oraciones(e["explicacion"])]


FUERTES = set("aeoáéóíú")  # las débiles con tilde (í, ú) rompen el diptongo: dí-a, ba-úl
VOCALES = FUERTES | set("iuü")


def contar_silabas(palabra):
    """Aproximación: un núcleo por grupo de vocales, y uno más en cada hiato entre fuertes (a-or-ta)."""
    p = palabra.lower()
    p = re.sub(r"(?<=[qg])u(?=[eiéí])", "", p)  # la u de "que", "gui" no suena
    silabas, anterior = 0, ""
    for c in p:
        if c in VOCALES and (anterior not in VOCALES or (c in FUERTES and anterior in FUERTES)):
            silabas += 1
        anterior = c
    return max(silabas, 1)


def inflesz(texto):
    """Índice de legibilidad de Szigriszt-Pazos (escala INFLESZ): 206,835 − 62,3·sílabas/palabras − palabras/frases.

    Escala: < 40 muy difícil, 40-55 algo difícil, 55-65 normal, 65-80 bastante fácil, > 80 muy fácil.
    """
    palabras = re.findall(r"[^\W\d_]+", texto)
    frases = dividir_oraciones(texto)
    if not palabras or not frases:
        return None
    silabas = sum(contar_silabas(p) for p in palabras)
    return 206.835 - 62.3 * silabas / len(palabras) - len(palabras) / len(frases)


# --- Calidad (generador v2, 2026-11-02): errores vistos con PDFs reales que las métricas de arriba no detectan ---

MARCAS_CAUSA = ("puede deberse", "podria deberse", "pueden deberse", "podrian deberse", "causas", "factores",
                "como una dieta", "puede estar relacionado", "podria estar relacionado", "puede indicar", "podria indicar")
_CATALOGO_SIGLAS = None


def _palabras(texto):
    vacias = {"de", "del", "la", "el", "los", "las", "en", "y", "a", "o", "u", "que", "tu", "es", "un", "una", "lo"}
    return [p for p in re.findall(r"[a-z0-9ñ]+", _sin_tildes(texto)) if p not in vacias]


def causas_repetidas(datos):
    """
    Oraciones de causas ("puede deberse a...") que se repiten, casi iguales, en exámenes distintos de la misma
    respuesta. Caso real (Colsanitas 2011): "una dieta alta en grasas o azúcares, falta de ejercicio..." copiado en
    LDL, triglicéridos y AST, aunque la AST es una enzima del hígado. Devuelve las oraciones repetidas.
    """
    vistas, repetidas = [], []
    for e in (datos or {}).get("explicaciones", []):
        for o in dividir_oraciones(e.get("explicacion", "")):
            if not any(m in _sin_tildes(o) for m in MARCAS_CAUSA):
                continue
            p = set(_palabras(o))
            if len(p) < 6:
                continue
            if any(examen != e.get("examen") and len(p & q) >= 0.8 * min(len(p), len(q)) for examen, q in vistas):
                repetidas.append(o)
            vistas.append((e.get("examen"), p))
    return repetidas


def _catalogo_siglas():
    """Sigla (VCM, TSH, HbA1c...) -> nombres largos del analito, con el catálogo y los alias de la app."""
    global _CATALOGO_SIGLAS
    if _CATALOGO_SIGLAS is None:
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        catalogo = json.loads((raiz / "training" / "sintetico" / "catalogo_analitos.json").read_text(encoding="utf-8"))
        nombres = {clave: list(a["nombres"]) for clave, a in catalogo["analitos"].items()}
        try:
            import sys
            sys.path.insert(0, str(raiz))
            from modelos_propios import ALIAS_EXTRA
            for clave, extra in ALIAS_EXTRA.items():
                nombres.setdefault(clave, []).extend(extra)
        except Exception:
            pass
        _CATALOGO_SIGLAS = {}
        for clave, lista in nombres.items():
            largos = [n for n in lista if len(_palabras(n)) >= 2]
            for n in lista:
                if re.fullmatch(r"[A-Za-z]{2,5}\d?[a-z]?\d?", n) and n[:2].isupper() and largos:
                    _CATALOGO_SIGLAS.setdefault(n.upper(), set()).update(largos)
    return _CATALOGO_SIGLAS


def parece_expansion(sigla, expansion):
    """
    El paréntesis desarrolla la sigla si sus iniciales la contienen casi entera, en orden: "VCM (Volumen de Célula
    Móvil)" sí (V, C, M); "HDL (colesterol bueno)" o "TGO (también conocida como AST)" no, porque son aclaraciones.
    """
    letras = [c for c in sigla.upper() if c.isalpha()]
    iniciales = [p[0].upper() for p in _palabras(expansion)]
    i = coincidencias = 0
    for letra in letras:
        while i < len(iniciales) and iniciales[i] != letra:
            i += 1
        if i < len(iniciales):
            coincidencias += 1
            i += 1
    return coincidencias >= max(2, len(letras) - 1)


def siglas_inventadas(datos):
    """
    Siglas explicadas con un significado que no es el suyo: "El VCM (Volumen de Célula Móvil)" o
    "Hidrógeno de la Célula Móvil (HCM)". La expansión se compara con los nombres largos del catálogo; es válida si
    contiene al menos el 60 % de las palabras de alguno ("Volumen corpuscular medio"). Devuelve los fragmentos.
    """
    siglas = _catalogo_siglas()
    malas = []
    for e in (datos or {}).get("explicaciones", []):
        texto = e.get("explicacion", "")
        candidatos = [(m.group(1), m.group(2)) for m in re.finditer(r"\b([A-Z][A-Za-z]{1,4}\d?[a-z]?\d?)\s*\(([^)]{6,80})\)", texto)]
        candidatos += [(m.group(2), m.group(1)) for m in
                       re.finditer(r"((?:[A-Za-zÁÉÍÓÚáéíóúñÑ]+\s+){1,6}[A-Za-zÁÉÍÓÚáéíóúñÑ]+)\s*\(([A-Z][A-Za-z]{1,4}\d?[a-z]?\d?)\)", texto)]
        for sigla, expansion in candidatos:
            largos = siglas.get(sigla.upper())
            if not largos or not re.search(r"[a-záéíóúñ]{3}", expansion) or not parece_expansion(sigla, expansion):
                continue
            palabras = set(_palabras(expansion))
            if not any(len(palabras & set(_palabras(n))) >= 0.6 * len(_palabras(n)) for n in largos):
                malas.append(f"{sigla} ({expansion.strip()})")
    return malas


def corregir_siglas(texto):
    """
    Cambia el significado inventado de una sigla por su nombre real del catálogo:
    "El VCM (Volumen de Célula Móvil)" -> "El VCM (volumen corpuscular medio)". Devuelve (texto, cambios).
    Con la forma "Expansión (SIGLA)" solo se reemplazan las palabras que forman la expansión, no las de antes.
    """
    siglas = _catalogo_siglas()
    cambios = []

    def nombre_real(sigla):
        largos = sorted(siglas.get(sigla.upper(), ()), key=lambda n: -len(_palabras(n)))
        return largos[0].lower() if largos else None

    def es_valida(sigla, expansion):
        return any(len(set(_palabras(expansion)) & set(_palabras(n))) >= 0.6 * len(_palabras(n))
                   for n in siglas.get(sigla.upper(), ()))

    def sigla_primero(m):
        sigla, expansion = m.group(1), m.group(2)
        real = nombre_real(sigla)
        if real and parece_expansion(sigla, expansion) and not es_valida(sigla, expansion):
            cambios.append((f"{sigla} ({expansion})", f"{sigla} ({real})"))
            return f"{sigla} ({real})"
        return m.group(0)

    def expansion_primero(m):
        palabras, sigla = m.group(1).split(), m.group(2)
        real = nombre_real(sigla)
        if not real:
            return m.group(0)
        # La expansión es la cola más corta de palabras cuyas iniciales forman la sigla
        for inicio in range(len(palabras) - 1, -1, -1):
            cola = " ".join(palabras[inicio:])
            if parece_expansion(sigla, cola):
                if es_valida(sigla, cola):
                    return m.group(0)
                cambios.append((f"{cola} ({sigla})", f"{real} ({sigla})"))
                return " ".join(palabras[:inicio] + [real]) + f" ({sigla})"
        return m.group(0)

    texto = re.sub(r"\b([A-Z][A-Za-z]{1,4}\d?[a-z]?\d?)\s*\(([^)]{6,80})\)", sigla_primero, texto)
    texto = re.sub(r"((?:[A-Za-zÁÉÍÓÚáéíóúñÑ]+\s+){1,6}[A-Za-zÁÉÍÓÚáéíóúñÑ]+)\s*\(([A-Z][A-Za-z]{1,4}\d?[a-z]?\d?)\)",
                   expansion_primero, texto)
    return texto, cambios


def usa_usted(datos):
    """El estilo del proyecto es tuteo cálido; "consulte a su médico" o "usted" lo rompen."""
    texto = _sin_tildes(" ".join(e.get("explicacion", "") for e in (datos or {}).get("explicaciones", [])) + " "
                        + " ".join((datos or {}).get("preguntas_sugeridas", [])))
    return bool(re.search(r"\busted(es)?\b|\bsu medico\b|\bconsulte\b|\bconverse con su\b|\ble recomend", texto))


def preguntas_a_quitar(datos, caso):
    """Preguntas que la revisión de coherencia de la app quitaría (dicen alto/bajo de un valor que no lo está,
    nombran un examen que no está en el caso o repiten otra). Devuelve (válidas, cuántas se quitan)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend import revisar_coherencia
    preguntas = [p for p in (datos or {}).get("preguntas_sugeridas", []) if isinstance(p, str) and p.strip()]
    copia = {"explicaciones": [{"examen": e.get("examen"), "explicacion": e.get("explicacion", "")}
                               for e in (datos or {}).get("explicaciones", [])],
             "preguntas_sugeridas": list(preguntas)}
    # Solo interesa el filtro de preguntas: las explicaciones van alineadas en el mismo orden que los analitos
    if len(copia["explicaciones"]) != len(caso["analitos"]):
        copia["explicaciones"] = [{"examen": a["examen"], "explicacion": "dentro del rango normal"} for a in caso["analitos"]]
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        revisar_coherencia(caso["analitos"], copia, caso.get("tipo_de_examen", ""))
    validas = [p for p in copia["preguntas_sugeridas"] if p in preguntas]
    return validas, len(preguntas) - len(validas)


def calidad(texto, caso):
    """Las cuatro verificaciones de calidad juntas, para el dataset v2 y para evaluar_generador.py."""
    datos = parsear(texto)
    if datos is None:
        return None
    validas, quitadas = preguntas_a_quitar(datos, caso)
    return {"causas_repetidas": causas_repetidas(datos), "siglas_inventadas": siglas_inventadas(datos),
            "usa_usted": usa_usted(datos), "preguntas_quitadas": quitadas, "preguntas_validas": validas}


def es_ejemplo_limpio(v):
    """Criterio para aceptar una respuesta del profesor como ejemplo de entrenamiento."""
    return (v.get("json_valido") and v["cobertura"] == 1 and v["estado_consistente"] == 1
            and v["examenes_extra"] == 0 and 2 <= v["preguntas"] <= 5)
