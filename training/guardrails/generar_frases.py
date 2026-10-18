"""
Fase 3: genera el dataset de frases para el clasificador de guardrails con un modelo profesor en Groq.

Tres clases, pensadas para lo que el asistente le escribe a un paciente:
- informativa: explica sin diagnosticar ni asustar (lo que el asistente debe decir)
- diagnostica: da por hecha una enfermedad o indica un tratamiento
- alarmista: exagera la gravedad o genera urgencia injustificada

Cada petición fija un analito con un resultado sintético (valor, rango y estado, generados con
training/sintetico/generar_examenes.py) y una clase. El estado se sortea igual para las tres
clases, para que el clasificador no aprenda el atajo "valor alto = frase peligrosa".

Se piden a propósito casos difíciles para el filtro de palabras actual de step_4_validation:
frases informativas con "tienes" o "diagnóstico" y frases diagnósticas sin esas palabras.

El script es reanudable: salta las combinaciones (analito, clase, ronda) ya generadas.

Uso:
    python training/guardrails/generar_frases.py [--rondas 2] [--modelo qwen/qwen3.8-27b]
"""
import argparse
import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "training"))
sys.path.insert(0, str(RAIZ / "training" / "sintetico"))

from groq_cliente import crear_sesion, pedir_json  # noqa: E402
from generar_examenes import _elegir_contexto, cargar_catalogo, generar_analito  # noqa: E402

SALIDA = RAIZ / "data" / "guardrails" / "frases.jsonl"
CLASES = ["informativa", "diagnostica", "alarmista"]
FRASES_POR_PETICION = 8

DEFINICIONES = """- informativa: explica qué mide el examen y qué significa el valor frente al rango; si menciona causas o enfermedades, lo hace como posibilidad ("puede deberse a", "a veces se relaciona con"); invita a conversarlo con el médico. Tono calmado y claro.
- diagnostica: da por hecho que el paciente tiene una enfermedad o condición, o descarta una con certeza, o le indica un tratamiento, medicamento, dosis o dieta concreta como si fuera su médico.
- alarmista: exagera la gravedad, genera miedo o urgencia injustificada, o presenta desenlaces graves (cáncer, muerte, daño irreversible) como probables."""

PISTAS = {
    "informativa": [
        "al menos 2 que usen palabras como \"tienes\", \"diagnóstico\" o \"enfermedad\" sin diagnosticar (ej. \"tienes el valor un poco por encima del rango\", \"solo tu médico puede hacer un diagnóstico\")",
        "al menos 1 que mencione una enfermedad concreta solo como posibilidad",
        "al menos 1 que explique qué mide el examen",
    ],
    "diagnostica": [
        "al menos 3 sin usar las palabras \"diagnóstico\", \"tienes\", \"padeces\" ni \"sufres\" (ej. \"Esto es una anemia\", \"Tu riñón no está funcionando bien\")",
        "al menos 1 que indique un medicamento, dosis o tratamiento",
        "al menos 2 sutiles, con tono amable",
    ],
    "alarmista": [
        "al menos 3 sutiles, sin signos de exclamación ni mayúsculas",
        "al menos 1 que pida ir a urgencias sin motivo claro",
        "al menos 1 que presente un desenlace grave como probable",
    ],
}

PROMPT = """Eres un generador de datos para entrenar un clasificador de seguridad. El clasificador revisa las frases que un asistente de IA escribe a pacientes de Latinoamérica para explicarles sus exámenes de laboratorio. El asistente nunca debe diagnosticar ni alarmar.

Clases:
{definiciones}

Resultado del paciente:
- Examen: {examen} (también llamado: {alias})
- Valor: {valor} {unidad}
- Rango de referencia: {rango}
- Estado: {estado}
- Contexto que escribió el paciente: {contexto}

Escribe {n} frases distintas de la clase "{clase}", tal como el asistente se las escribiría al paciente (tuteo, español neutro). Cada frase es una sola oración de máximo 35 palabras y se entiende sola.
Incluye:
{pistas}

Responde solo con JSON: {{"frases": ["...", "..."]}}"""


def construir_prompt(rng, catalogo, clave, clase):
    analito = catalogo["analitos"][clave]
    sexo = rng.choice("MF")
    edad = rng.randint(18, 80)
    resultado = generar_analito(rng, clave, analito, sexo, coma=False)
    contexto = _elegir_contexto(rng, sexo, edad) or "No proporcionado."
    prompt = PROMPT.format(
        definiciones=DEFINICIONES, examen=resultado["examen"], alias=", ".join(analito["nombres"]),
        valor=resultado["valor"], unidad=resultado["unidad"] or "", rango=resultado["rango_referencia"],
        estado=resultado["estado"], contexto=contexto, n=FRASES_POR_PETICION, clase=clase,
        pistas="\n".join(f"- {p}" for p in PISTAS[clase]))
    return prompt, resultado


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="qwen/qwen3.8-27b")
    parser.add_argument("--rondas", type=int, default=2, help="Peticiones por (analito, clase)")
    parser.add_argument("--limite", type=int, help="Hacer solo N peticiones (prueba rápida)")
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()

    catalogo = cargar_catalogo()
    sesion = crear_sesion()
    args.salida.parent.mkdir(parents=True, exist_ok=True)

    hechas, vistas = set(), set()
    if args.salida.exists():
        with open(args.salida, encoding="utf-8") as f:
            for linea in f:
                fila = json.loads(linea)
                hechas.add((fila["analito"], fila["clase"], fila["ronda"]))
                vistas.add(fila["frase"].lower())

    tareas = [(clave, clase, ronda) for ronda in range(args.rondas)
              for clave in catalogo["analitos"] for clase in CLASES]
    pendientes = [t for t in tareas if t not in hechas][:args.limite]
    print(f"{len(tareas) - len(pendientes)} peticiones ya hechas; faltan {len(pendientes)}", flush=True)

    total, repetidas = 0, 0
    with open(args.salida, "a", encoding="utf-8") as f:
        for i, (clave, clase, ronda) in enumerate(pendientes, 1):
            # Semilla por tarea: el mismo resultado sintético aunque el script se reanude
            rng = random.Random(f"{clave}-{clase}-{ronda}")
            prompt, resultado = construir_prompt(rng, catalogo, clave, clase)
            respuesta = pedir_json(sesion, args.modelo, prompt, etiqueta=f"{clave}/{clase}/{ronda}")
            for frase in respuesta.get("frases", []):
                if not isinstance(frase, str) or not frase.strip():
                    continue
                frase = frase.strip()
                if frase.lower() in vistas:
                    repetidas += 1
                    continue
                vistas.add(frase.lower())
                f.write(json.dumps({"frase": frase, "clase": clase, "analito": clave, "ronda": ronda,
                                    "estado": resultado["estado"], "profesor": args.modelo},
                                   ensure_ascii=False) + "\n")
                total += 1
            f.flush()
            if i % 10 == 0 or i == len(pendientes):
                print(f"  {i}/{len(pendientes)} peticiones, {total} frases, {repetidas} repetidas", flush=True)

    print(f"Listo: {total} frases nuevas en {args.salida}")


if __name__ == "__main__":
    main()
