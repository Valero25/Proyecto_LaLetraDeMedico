"""
Modelos propios de LetraDeMédico (MODO=propio), entrenados en las Fases 2 a 4.

- RecuperadorRAG: ALIA-MrBERT ajustado + FAISS sobre la base de MedlinePlus (Fase 2).
- GeneradorPropio: Qwen2.5-3B-Instruct ajustado con QLoRA, exportado a GGUF y servido con llama.cpp (Fase 4).
- Guardrail: MrBERT-biomed v2, clasificador de frases informativa / diagnóstica / alarmista (Fase 3).

Cada clase carga su modelo una sola vez (la app las guarda en caché). Las rutas por defecto
apuntan a models/; se pueden cambiar con variables de entorno (ver RUTAS).
"""
import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
for carpeta in ("training/rag", "training/generador"):
    sys.path.insert(0, str(RAIZ / carpeta))

from prompt import mensajes_estudiante, recortar  # noqa: E402
from verificacion import dividir_oraciones, parsear  # noqa: E402

RUTAS = {
    "rag": Path(os.environ.get("LDM_MODELO_RAG", RAIZ / "models" / "rag" / "alia-ajustado-completo")),
    "indice": Path(os.environ.get("LDM_INDICE_RAG", RAIZ / "models" / "rag" / "indice-alia-ajustado")),
    "guardrail": Path(os.environ.get("LDM_MODELO_GUARDRAIL", RAIZ / "models" / "guardrails" / "mrbert-biomed-v2")),
    "generador": Path(os.environ.get("LDM_MODELO_GENERADOR", RAIZ / "models" / "generador" / "letrademedico-qwen2.5-3b-q4_k_m.gguf")),
}
MAX_CARACTERES_PASAJE = 700  # igual que en el entrenamiento (training/generador/preparar_casos.py)
UMBRAL_RIESGO = 0.5


def faltantes():
    """Modelos que no están en disco; la app usa esto para ofrecer o no el modo propio."""
    revisar = {"rag": RUTAS["rag"] / "model.safetensors", "guardrail": RUTAS["guardrail"] / "model.safetensors",
               "generador": RUTAS["generador"]}
    return [nombre for nombre, ruta in revisar.items() if not ruta.exists()]


class RecuperadorRAG:
    """Primer fragmento de MedlinePlus para "{examen} ({tipo de examen})", como en el entrenamiento del generador."""

    def __init__(self):
        import faiss
        from recuperador import RecuperadorDenso, cargar_documentos

        if (RUTAS["indice"] / "indice.faiss").exists():
            from sentence_transformers import SentenceTransformer
            self.modelo = SentenceTransformer(str(RUTAS["rag"]))
            self.indice = faiss.read_index(str(RUTAS["indice"] / "indice.faiss"))
            self.documentos = json.loads((RUTAS["indice"] / "documentos.json").read_text(encoding="utf-8"))
            self.usa_prompt = "query" in (self.modelo.prompts or {})
        else:
            # Primera vez: indexa los 208 fragmentos (unos minutos en CPU) y guarda el índice para la próxima
            r = RecuperadorDenso(cargar_documentos(con_alias=True), str(RUTAS["rag"]))
            r.guardar(RUTAS["indice"])
            self.modelo, self.indice, self.documentos, self.usa_prompt = r.modelo, r.indice, r.documentos, r.usa_prompt

        # Nombre del examen (sin tildes ni mayúsculas) -> analitos del catálogo, para el reordenamiento
        from recuperador import CATALOGO_PATH, normalizar
        self.normalizar = normalizar
        self.analitos_por_nombre = {}
        for clave, a in json.loads(Path(CATALOGO_PATH).read_text(encoding="utf-8"))["analitos"].items():
            for nombre in a["nombres"]:
                self.analitos_por_nombre.setdefault(normalizar(nombre), set()).add(clave)

    def buscar(self, consulta, examen=None, k=5):
        """
        Búsqueda densa y reordenamiento por alias: entre los k primeros, se prefiere el fragmento que el
        catálogo asocia al nombre exacto del examen. Así "Glucosa" no termina en la página de HbA1c.
        Medido en las 182 consultas del set del generador: 97,8 % -> 99,5 % de acierto.
        """
        kwargs = {"prompt_name": "query"} if self.usa_prompt else {}
        vector = self.modelo.encode([consulta], normalize_embeddings=True, **kwargs)
        _, posiciones = self.indice.search(vector, k)
        candidatos = [self.documentos[i] for i in posiciones[0] if i >= 0]
        validos = self.analitos_por_nombre.get(self.normalizar(examen or ""), set())
        return next((d for d in candidatos if validos & set(d["analitos"])), candidatos[0])

    def recuperar(self, analitos, tipo_examen):
        pasajes = []
        for a in analitos:
            doc = self.buscar(f"{a['examen']} ({tipo_examen})", examen=a["examen"])
            pasajes.append({"examen": a["examen"], "titulo": doc["titulo"], "fuente": doc.get("url") or doc.get("fuente"),
                            "texto": recortar(doc["texto"], MAX_CARACTERES_PASAJE)})
        return pasajes


class GeneradorPropio:
    """Qwen2.5-3B + LoRA en GGUF (llama.cpp). La salida se restringe a JSON válido con una gramática."""

    def __init__(self, hilos=None):
        from llama_cpp import Llama
        self.llm = Llama(model_path=str(RUTAS["generador"]), n_ctx=4096, n_threads=hilos or os.cpu_count(),
                         verbose=False)

    def generar(self, caso, pasajes, max_tokens=1200):
        respuesta = self.llm.create_chat_completion(
            messages=mensajes_estudiante(caso, pasajes), temperature=0, max_tokens=max_tokens,
            response_format={"type": "json_object"})
        texto = respuesta["choices"][0]["message"]["content"]
        return parsear(texto) or {}


class Guardrail:
    """Marca las oraciones con P(diagnóstica) + P(alarmista) >= UMBRAL_RIESGO."""

    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizador = AutoTokenizer.from_pretrained(RUTAS["guardrail"])
        self.modelo = AutoModelForSequenceClassification.from_pretrained(RUTAS["guardrail"]).eval()
        self.clases = [self.modelo.config.id2label[i] for i in range(self.modelo.config.num_labels)]

    def riesgo(self, oraciones):
        import torch
        if not oraciones:
            return []
        with torch.no_grad():
            lote = self.tokenizador(oraciones, truncation=True, max_length=96, padding=True, return_tensors="pt")
            probs = self.modelo(**lote).logits.softmax(-1)
        return (1 - probs[:, self.clases.index("informativa")]).tolist()

    def revisar(self, resultado):
        """Agrega a cada explicación la lista de oraciones marcadas (vacía si no hay)."""
        explicaciones = resultado.get("explicaciones", [])
        oraciones = [dividir_oraciones(e.get("explicacion", "")) for e in explicaciones]
        riesgos = iter(self.riesgo([o for frases in oraciones for o in frases]))
        for e, frases in zip(explicaciones, oraciones):
            e["oraciones_marcadas"] = [o for o in frases if next(riesgos) >= UMBRAL_RIESGO]
        return resultado
