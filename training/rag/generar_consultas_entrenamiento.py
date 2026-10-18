"""
Fase 2 (ajuste contrastivo): genera consultas de entrenamiento con un modelo profesor en Groq.

Por cada fragmento de la base de conocimiento, el profesor escribe consultas variadas que ese
fragmento responde: nombres como aparecen en un examen (con abreviaturas), lenguaje coloquial,
preguntas completas y errores de tipeo. Cada par (consulta, fragmento) es un ejemplo positivo.

Se descartan las consultas idénticas (ignorando mayúsculas y tildes) a alguna del set de
evaluación, para no entrenar con las respuestas del examen.

El script es reanudable: si se corta, al volver a correrlo salta los fragmentos ya procesados.

Uso:
    python training/rag/generar_consultas_entrenamiento.py [--modelo qwen/qwen3.8-27b]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "training" / "rag"))
sys.path.insert(0, str(RAIZ / "evaluacion"))

from recuperador import cargar_documentos, normalizar  # noqa: E402
from evaluar_rag import construir_consultas  # noqa: E402

SALIDA = RAIZ / "data" / "rag" / "consultas_entrenamiento.jsonl"
URL_GROQ = "https://api.groq.com/openai/v1/chat/completions"
CONSULTAS_POR_FRAGMENTO = 6
MAX_INTENTOS = 6

PROMPT = """Eres un generador de datos para entrenar un buscador médico en español.

Fragmento de una página de salud:
Título: {titulo}
Sección: {seccion}
Texto: {texto}

Nombres con que los laboratorios reportan este examen: {alias}

Escribe {n} búsquedas distintas que un paciente latinoamericano escribiría y que este fragmento responde.
Varía el estilo entre ellas:
- el nombre tal como sale en su examen, a veces abreviado (usa los nombres de arriba), solo o con "alto", "bajo", "positivo", etc.
- lenguaje coloquial, sin términos técnicos
- preguntas completas
- al menos una sin tildes o con un error de tipeo realista
Cada búsqueda tiene como máximo 12 palabras. No copies frases del texto.

Responde solo con JSON: {{"consultas": ["...", "..."]}}"""


def pedir_consultas(sesion, modelo, doc):
    cuerpo = {
        "model": modelo,
        "messages": [{"role": "user", "content": PROMPT.format(
            titulo=doc["titulo"], seccion=doc["seccion"], texto=doc["texto"][:2500],
            alias=", ".join(doc["alias"]), n=CONSULTAS_POR_FRAGMENTO)}],
        "response_format": {"type": "json_object"},
        "temperature": 0.8,
    }
    for intento in range(1, MAX_INTENTOS + 1):
        r = sesion.post(URL_GROQ, json=cuerpo, timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
            # Groq indica cuánto esperar; el límite gratuito es por minuto
            espera = float(r.headers.get("retry-after", 10 * intento))
            print(f"    {r.status_code}: esperando {espera:.0f}s (intento {intento}/{MAX_INTENTOS})", flush=True)
            time.sleep(espera)
            continue
        if r.status_code == 400 and "json_validate_failed" in r.text:
            # En modo JSON, Groq responde 400 si el modelo generó JSON inválido: basta con reintentar
            print(f"    JSON inválido del modelo, reintentando (intento {intento}/{MAX_INTENTOS})", flush=True)
            continue
        if not r.ok:
            raise RuntimeError(f"Groq respondió {r.status_code} para {doc['id']}: {r.text[:300]}")
        try:
            consultas = json.loads(r.json()["choices"][0]["message"]["content"])["consultas"]
            return [c.strip() for c in consultas if isinstance(c, str) and c.strip()]
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"    respuesta mal formada, reintentando (intento {intento}/{MAX_INTENTOS})", flush=True)
    raise RuntimeError(f"Groq no respondió tras {MAX_INTENTOS} intentos para {doc['id']}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="qwen/qwen3.8-27b")
    args = parser.parse_args()

    load_dotenv(RAIZ / ".env")
    sesion = requests.Session()
    sesion.headers.update({"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "User-Agent": "LetraDeMedico/1.0"})

    evaluacion = {normalizar(c["consulta"]) for c in construir_consultas()}
    ya_hechos = set()
    if SALIDA.exists():
        with open(SALIDA, encoding="utf-8") as f:
            ya_hechos = {json.loads(linea)["doc_id"] for linea in f}

    documentos = [d for d in cargar_documentos() if d["id"] not in ya_hechos]
    print(f"{len(ya_hechos)} fragmentos ya procesados; faltan {len(documentos)}", flush=True)

    total, descartadas = 0, 0
    with open(SALIDA, "a", encoding="utf-8") as f:
        for i, doc in enumerate(documentos, 1):
            vistas = set()
            for consulta in pedir_consultas(sesion, args.modelo, doc):
                clave = normalizar(consulta)
                if clave in evaluacion or clave in vistas:
                    descartadas += 1
                    continue
                vistas.add(clave)
                f.write(json.dumps({"consulta": consulta, "doc_id": doc["id"], "analitos": doc["analitos"],
                                    "profesor": args.modelo}, ensure_ascii=False) + "\n")
                total += 1
            f.flush()
            if i % 10 == 0 or i == len(documentos):
                print(f"  {i}/{len(documentos)} fragmentos, {total} consultas, {descartadas} descartadas", flush=True)

    print(f"Listo: {total} consultas nuevas en {SALIDA.relative_to(RAIZ)} ({descartadas} descartadas por repetidas o por estar en la evaluación)")


if __name__ == "__main__":
    main()
