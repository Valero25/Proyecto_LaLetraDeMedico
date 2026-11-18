"""
Fase 4: formato de entrada y salida del generador de explicaciones.

El modelo ajustado ve exactamente este prompt, tanto en el entrenamiento como en la app (Fase 5).
El profesor recibe el mismo prompt más una guía de estilo (GUIA_PROFESOR). El estudiante no la ve:
debe aprender ese estilo de los ejemplos, no de las instrucciones.
"""
import json

SISTEMA = """Eres LetraDeMédico, un asistente que explica resultados de laboratorio a pacientes en español claro y cálido (tuteo).
Reglas:
- No diagnosticas ni indicas tratamientos. Si mencionas causas, hazlo como posibilidad ("puede deberse a").
- No alarmes. Los estados (NORMAL, ALTO, BAJO, ANORMAL) ya fueron calculados: no los cambies.
- Basa tus explicaciones en los pasajes médicos provistos.
- Responde solo con JSON: {"explicaciones": [{"examen": "...", "explicacion": "..."}], "preguntas_sugeridas": ["..."]}"""

GUIA_PROFESOR = """Guía de estilo (síguela con cuidado):
- Escribe una explicación por cada examen, en el mismo orden y con el mismo nombre de examen.
- Exámenes NORMALES: 1 o 2 oraciones (qué mide y que su valor está dentro del rango).
- Exámenes fuera de rango: 3 o 4 oraciones: qué mide, qué significa su valor frente al rango (menciona el valor), posibles causas generales en condicional y una invitación a conversarlo con su médico.
- Usa el contexto del paciente solo si es relevante para ese examen.
- Lenguaje sencillo: si usas un término técnico, explícalo en la misma oración.
- Entre 3 y 4 preguntas sugeridas, concretas, para hacerle al médico.
- Sin markdown, sin emojis y sin repetir el aviso de que no eres médico en cada examen."""


# v3 (2026-11-04): las causas salen del pasaje. Con la guía anterior los profesores escribían "posibles causas
# generales" de memoria, y el estudiante aprendió a inventarlas (AST alta: "una menor actividad física").
GUIA_PROFESOR_V3 = GUIA_PROFESOR.replace(
    "posibles causas generales en condicional y una invitación a conversarlo con su médico.",
    "posibles causas en condicional TOMADAS DEL PASAJE de ese examen (elige 2 o 3 de las que el pasaje nombra; "
    "no agregues causas que el pasaje no menciona) y una invitación a conversarlo con su médico. Si el pasaje no "
    "trae causas o no hay pasaje, no nombres causas concretas: di que puede tener varias explicaciones y que tu "
    "médico puede ayudarte a entenderlo.\n- No repitas la misma oración de causas en exámenes distintos.")
assert GUIA_PROFESOR_V3 != GUIA_PROFESOR


def formatear_resultado(a):
    unidad = f" {a['unidad']}" if a.get("unidad") else ""
    return f"- {a['examen']}: {a['valor']}{unidad} (referencia: {a['rango_referencia']}) → {a['estado']}"


def construir_usuario(caso, pasajes):
    """`pasajes` es una lista de {"examen", "titulo", "texto"}, uno por examen."""
    paciente = caso.get("paciente") or {}
    sexo = {"M": "hombre", "F": "mujer"}.get(paciente.get("sexo"), "sexo no indicado")
    # En la app la edad y el sexo pueden faltar: el documento no siempre los trae
    edad = f"{paciente['edad']} años" if paciente.get("edad") else "edad no indicada"
    lineas = [
        f"Tipo de examen: {caso['tipo_de_examen']}",
        f"Paciente: {sexo}, {edad}.",
        f"Contexto que escribió el paciente: {caso.get('contexto_paciente') or 'No proporcionado.'}",
        "",
        "Resultados:",
        *[formatear_resultado(a) for a in caso["analitos"]],
        "",
        "Pasajes médicos:",
        *[f"[{p['examen']}] {p['titulo']}: {p['texto']}" for p in pasajes],
    ]
    return "\n".join(lineas)


def mensajes_estudiante(caso, pasajes, respuesta=None):
    """Conversación en formato chat; con `respuesta` es un ejemplo de entrenamiento completo."""
    mensajes = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": construir_usuario(caso, pasajes)}]
    if respuesta is not None:
        mensajes.append({"role": "assistant", "content": json.dumps(respuesta, ensure_ascii=False)})
    return mensajes


def recortar(texto, limite):
    if len(texto) <= limite:
        return texto
    return texto[:limite].rsplit(" ", 1)[0] + "…"


# Groq cuenta los tokens del prompt en su límite diario: el profesor recibe pasajes más cortos que el
# estudiante (que ve hasta 700 caracteres). El inicio de cada fragmento ("¿Qué es?") basta para escribir bien.
MAX_CARACTERES_PASAJE_PROFESOR = 350


def prompt_profesor(caso, pasajes, max_caracteres=MAX_CARACTERES_PASAJE_PROFESOR, version="v1"):
    """v3: guía con causas del pasaje y el pasaje completo (700 caracteres, lo mismo que ve el estudiante): el
    profesor local no tiene cuotas, y a 350 caracteres se cortaba justo la lista de causas."""
    if version == "v3":
        return f"{SISTEMA}\n\n{GUIA_PROFESOR_V3}\n\n{construir_usuario(caso, pasajes)}"
    cortos = [{**p, "texto": recortar(p["texto"], max_caracteres)} for p in pasajes]
    return f"{SISTEMA}\n\n{GUIA_PROFESOR}\n\n{construir_usuario(caso, cortos)}"
