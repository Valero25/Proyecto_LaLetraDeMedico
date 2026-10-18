"""
Fase 2: Recuperadores para el RAG.

- RecuperadorDenso: embeddings (ALIA-MrBERT u otro modelo de sentence-transformers) + índice FAISS.
- RecuperadorBM25: baseline léxico clásico sobre la misma base de conocimiento.
- RecuperadorSubcadena: réplica de la búsqueda actual del backend sobre mock_medical_kb.json.

Construir y guardar el índice denso para la app:
    python training/rag/recuperador.py
"""
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DOCUMENTOS_PATH = RAIZ / "data" / "rag" / "documentos.jsonl"
CATALOGO_PATH = RAIZ / "training" / "sintetico" / "catalogo_analitos.json"
INDICE_DIR = RAIZ / "models" / "rag"
MODELO_ALIA = "SINAI/ALIA-MrBERT-es-biomedical-embeddings"


def cargar_documentos(path=DOCUMENTOS_PATH, con_alias=True):
    """
    con_alias agrega a cada fragmento los nombres con que los laboratorios reportan sus analitos
    (ej. "TGO" para AST): MedlinePlus no usa esas abreviaturas y sin ellas no hay cómo encontrarlas.
    """
    with open(path, encoding="utf-8") as f:
        documentos = [json.loads(linea) for linea in f]
    if con_alias:
        with open(CATALOGO_PATH, encoding="utf-8") as f:
            analitos = json.load(f)["analitos"]
        for doc in documentos:
            doc["alias"] = list(dict.fromkeys(n for clave in doc["analitos"] for n in analitos[clave]["nombres"]))
    return documentos


def texto_para_indexar(doc):
    # El título y la sección dan contexto a fragmentos que por sí solos no nombran el analito
    alias = f" También se reporta como: {', '.join(doc['alias'])}." if doc.get("alias") else ""
    return f"{doc['titulo']}.{alias} {doc['seccion']} {doc['texto']}"


def normalizar(texto):
    sin_tildes = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sin_tildes if not unicodedata.combining(c))


class RecuperadorDenso:
    def __init__(self, documentos, modelo=MODELO_ALIA, prefijo_consulta=None, prefijo_documento=""):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.documentos = documentos
        self.modelo = SentenceTransformer(modelo)
        # ALIA-MrBERT define un prompt "query"; modelos como e5 usan prefijos de texto
        self.usa_prompt = prefijo_consulta is None and "query" in (self.modelo.prompts or {})
        self.prefijo_consulta = prefijo_consulta or ""
        textos = [prefijo_documento + texto_para_indexar(d) for d in documentos]
        vectores = self.modelo.encode(textos, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
        self.indice = faiss.IndexFlatIP(vectores.shape[1])
        self.indice.add(vectores)

    def buscar(self, consulta, k=5):
        kwargs = {"prompt_name": "query"} if self.usa_prompt else {}
        vector = self.modelo.encode([self.prefijo_consulta + consulta], normalize_embeddings=True, **kwargs)
        puntajes, posiciones = self.indice.search(vector, k)
        return [(self.documentos[i], float(p)) for i, p in zip(posiciones[0], puntajes[0]) if i >= 0]

    def guardar(self, carpeta=INDICE_DIR):
        import faiss

        carpeta.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.indice, str(carpeta / "indice.faiss"))
        with open(carpeta / "documentos.json", "w", encoding="utf-8") as f:
            json.dump(self.documentos, f, ensure_ascii=False)


class RecuperadorBM25:
    def __init__(self, documentos, k1=1.5, b=0.75):
        self.documentos = documentos
        self.k1, self.b = k1, b
        self.tokens = [self._tokenizar(texto_para_indexar(d)) for d in documentos]
        self.largo_medio = sum(map(len, self.tokens)) / len(self.tokens)
        df = Counter(t for toks in self.tokens for t in set(toks))
        n = len(documentos)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.frecuencias = [Counter(toks) for toks in self.tokens]

    @staticmethod
    def _tokenizar(texto):
        return re.findall(r"[a-z0-9]+", normalizar(texto))

    def buscar(self, consulta, k=5):
        terminos = self._tokenizar(consulta)
        puntajes = []
        for i, frec in enumerate(self.frecuencias):
            largo = len(self.tokens[i])
            p = sum(
                self.idf.get(t, 0) * frec[t] * (self.k1 + 1)
                / (frec[t] + self.k1 * (1 - self.b + self.b * largo / self.largo_medio))
                for t in terminos if t in frec
            )
            puntajes.append(p)
        mejores = sorted(range(len(puntajes)), key=lambda i: -puntajes[i])[:k]
        return [(self.documentos[i], puntajes[i]) for i in mejores if puntajes[i] > 0]


class RecuperadorSubcadena:
    """Misma lógica que LetraDeMedicoBackend.recuperar_conocimiento_rag (primera coincidencia parcial)."""

    # Claves del catálogo que explica cada entrada de mock_medical_kb.json
    MAPEO = {
        "glucosa": ["glucosa"],
        "hemoglobina a1c": ["hemoglobina_a1c"],
        "colesterol total": ["colesterol_total"],
        "trigliceridos": ["trigliceridos"],
        "creatinina": ["creatinina"],
    }

    def __init__(self, path=RAIZ / "mock_medical_kb.json"):
        with open(path, encoding="utf-8") as f:
            self.kb = json.load(f)

    def buscar(self, consulta, k=5):
        nombre = consulta.lower()
        for clave, texto in self.kb.items():
            if clave in nombre or nombre in clave:
                return [({"id": f"mock-{clave}", "analitos": self.MAPEO.get(clave, []), "texto": texto}, 1.0)]
        return []


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    documentos = cargar_documentos()
    recuperador = RecuperadorDenso(documentos)
    recuperador.guardar()
    print(f"Índice FAISS con {len(documentos)} fragmentos guardado en {INDICE_DIR.relative_to(RAIZ)}")
    for consulta in ["TGO alta", "colesterol bueno bajo", "nitritos positivos en la orina"]:
        doc, puntaje = recuperador.buscar(consulta, k=1)[0]
        print(f"  {consulta!r} -> {doc['titulo']} / {doc['seccion']} ({puntaje:.3f})")


if __name__ == "__main__":
    main()
