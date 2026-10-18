"""
Fase 2: Construye la base de conocimiento del RAG.

Descarga las páginas de MedlinePlus en español listadas en fuentes.json, conserva solo las
secciones útiles para explicar resultados (qué es, para qué sirve, qué significan los resultados...)
y las une con las fichas redactadas por el equipo (fichas_equipo.json).

Salida: data/rag/documentos.jsonl, un fragmento por sección. Se versiona en git porque las
páginas de MedlinePlus cambian con el tiempo y el experimento debe ser reproducible.

Uso:
    python training/rag/construir_base_conocimiento.py
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

RAIZ = Path(__file__).resolve().parents[2]
CARPETA = Path(__file__).parent
SALIDA = RAIZ / "data" / "rag" / "documentos.jsonl"
MAX_CARACTERES = 1200       # Las secciones más largas se parten por párrafos
PAUSA_SEGUNDOS = 1.0        # Cortesía con el servidor de la NLM


def _texto_de_bloques(seccion):
    """Extrae párrafos y listas como bloques de texto; las listas se unen con punto y coma."""
    bloques = []
    for el in seccion.find_all(["p", "ul", "ol"], recursive=True):
        if el.find_parent(["ul", "ol"]):
            continue
        if el.name == "p":
            texto = el.get_text(" ", strip=True)
        else:
            texto = "; ".join(li.get_text(" ", strip=True) for li in el.find_all("li"))
        if texto:
            bloques.append(texto)
    return bloques


def _partir(bloques, max_caracteres=MAX_CARACTERES):
    """Agrupa bloques consecutivos en fragmentos que no superen max_caracteres."""
    fragmentos, actual = [], ""
    for b in bloques:
        if actual and len(actual) + len(b) + 1 > max_caracteres:
            fragmentos.append(actual)
            actual = b
        else:
            actual = f"{actual} {b}".strip()
    if actual:
        fragmentos.append(actual)
    return fragmentos


def procesar_pagina(html, slug, analitos, url, secciones_utiles):
    soup = BeautifulSoup(html, "html.parser")
    titulo = soup.h1.get_text(strip=True)
    documentos = []
    for seccion in soup.select("article section"):
        encabezado = seccion.find(["h2", "h3"])
        if not encabezado:
            continue
        nombre = encabezado.get_text(strip=True)
        if not any(nombre.startswith(prefijo) for prefijo in secciones_utiles):
            continue
        for i, texto in enumerate(_partir(_texto_de_bloques(seccion))):
            documentos.append({
                "id": f"mlp-{slug}-{len(documentos):02d}",
                "titulo": titulo,
                "seccion": nombre,
                "texto": texto,
                "analitos": analitos,
                "fuente": "MedlinePlus en español (NLM)",
                "url": url,
            })
    return documentos


def documentos_de_fichas(path):
    with open(path, encoding="utf-8") as f:
        fichas = json.load(f)["fichas"]
    documentos = []
    for ficha in fichas:
        for i, s in enumerate(ficha["secciones"]):
            documentos.append({
                "id": f"{ficha['id']}-{i:02d}",
                "titulo": ficha["titulo"],
                "seccion": s["seccion"],
                "texto": s["texto"],
                "analitos": ficha["analitos"],
                "fuente": ficha["fuente"],
                "url": None,
            })
    return documentos


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(CARPETA / "fuentes.json", encoding="utf-8") as f:
        fuentes = json.load(f)

    documentos = []
    sesion = requests.Session()
    sesion.headers["User-Agent"] = "LetraDeMedico/1.0 (proyecto academico)"
    for slug, analitos in fuentes["paginas"].items():
        url = fuentes["base_url"] + slug + "/"
        respuesta = sesion.get(url, timeout=30)
        respuesta.raise_for_status()
        # El servidor no declara charset y requests asume ISO-8859-1, lo que daña los "¿" de los títulos
        respuesta.encoding = "utf-8"
        docs = procesar_pagina(respuesta.text, slug, analitos, url, fuentes["secciones_utiles"])
        if not docs:
            raise RuntimeError(f"No se extrajo ninguna sección de {url}: ¿cambió el HTML de MedlinePlus?")
        print(f"{len(docs):>3} fragmentos  {slug}")
        documentos += docs
        time.sleep(PAUSA_SEGUNDOS)

    fichas = documentos_de_fichas(CARPETA / "fichas_equipo.json")
    print(f"{len(fichas):>3} fragmentos  fichas del equipo")
    documentos += fichas

    hoy = date.today().isoformat()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as f:
        for d in documentos:
            f.write(json.dumps({**d, "descargado": hoy}, ensure_ascii=False) + "\n")

    with open(RAIZ / "training" / "sintetico" / "catalogo_analitos.json", encoding="utf-8") as f:
        claves = set(json.load(f)["analitos"])
    cubiertas = {a for d in documentos for a in d["analitos"]}
    print(f"\n{len(documentos)} fragmentos guardados en {SALIDA.relative_to(RAIZ)}")
    print(f"Cobertura del catálogo: {len(cubiertas & claves)}/{len(claves)} analitos")
    if claves - cubiertas:
        print("Sin documentos:", sorted(claves - cubiertas))


if __name__ == "__main__":
    main()
