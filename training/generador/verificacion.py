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


def es_ejemplo_limpio(v):
    """Criterio para aceptar una respuesta del profesor como ejemplo de entrenamiento."""
    return (v.get("json_valido") and v["cobertura"] == 1 and v["estado_consistente"] == 1
            and v["examenes_extra"] == 0 and 2 <= v["preguntas"] <= 5)
