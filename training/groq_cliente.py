"""
Cliente mínimo de Groq (API compatible con OpenAI) para los modelos profesor.

Reintenta ante 429/5xx (respetando retry-after) y ante JSON inválido del modelo.
"""
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
URL_GROQ = "https://api.groq.com/openai/v1/chat/completions"
MAX_INTENTOS = 6


def crear_sesion():
    load_dotenv(RAIZ / ".env")
    sesion = requests.Session()
    sesion.headers.update({"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "User-Agent": "LetraDeMedico/1.0"})
    return sesion


def pedir_json(sesion, modelo, prompt, temperatura=0.8, etiqueta="", **extra):
    """Devuelve el objeto JSON que responde el modelo. `extra` va tal cual al cuerpo (ej. reasoning_effort)."""
    cuerpo = {
        "model": modelo,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": temperatura,
        **extra,
    }
    for intento in range(1, MAX_INTENTOS + 1):
        r = sesion.post(URL_GROQ, json=cuerpo, timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
            espera = float(r.headers.get("retry-after", 10 * intento))
            print(f"    {r.status_code}: esperando {espera:.0f}s (intento {intento}/{MAX_INTENTOS})", flush=True)
            time.sleep(espera)
            continue
        if r.status_code == 400 and "json_validate_failed" in r.text:
            print(f"    JSON inválido del modelo, reintentando (intento {intento}/{MAX_INTENTOS})", flush=True)
            continue
        if not r.ok:
            raise RuntimeError(f"Groq respondió {r.status_code} ({etiqueta}): {r.text[:300]}")
        try:
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"    respuesta mal formada, reintentando (intento {intento}/{MAX_INTENTOS})", flush=True)
    raise RuntimeError(f"Groq no respondió tras {MAX_INTENTOS} intentos ({etiqueta})")
