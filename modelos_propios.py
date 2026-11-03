"""
Modelos propios de LetraDeMédico (MODO=propio), entrenados en las Fases 2 a 4.

- RecuperadorRAG: ALIA-MrBERT ajustado + FAISS sobre la base de MedlinePlus (Fase 2).
- GeneradorPropio: Qwen2.5-3B-Instruct ajustado con QLoRA, exportado a GGUF y servido con llama.cpp (Fase 4).
- Guardrail: MrBERT-biomed v2, clasificador de frases informativa / diagnóstica / alarmista (Fase 3).

Cada clase carga su modelo una sola vez (la app las guarda en caché). Las rutas por defecto
apuntan a models/; se pueden cambiar con variables de entorno (ver RUTAS).
"""
import glob
import json
import os
import re
import sys
import unicodedata
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
N_CTX_GENERADOR = 4096
MIN_TOKENS_RESPUESTA = 900  # un lote de 8 analitos ocupa unos 700 tokens de respuesta (medido con el PDF de prueba)
# Sin alias en el nombre, un pasaje se acepta solo con similitud coseno >= 0,55. Con 37 exámenes que no están en
# la base (IGF-1, ferritina, PSA, CK...) y 2 tipos de examen, la similitud máxima fue 0,54 (CK -> creatinina).
UMBRAL_SIN_ALIAS = 0.55


PALABRAS_VACIAS = {"de", "del", "la", "el", "los", "las", "en", "a", "u", "o", "y", "e", "por", "para", "con"}
PALABRAS_SANGRE = {"suero", "sangre", "serico", "serica", "plasma", "plasmatico", "plasmatica", "hemograma"}
PALABRAS_ORINA = {"orina", "urinario", "urinaria", "uroanalisis", "sedimento", "urianalisis"}


def tokens(texto):
    """Palabras sin tildes, mayúsculas, signos ni palabras vacías: "Hormona estimulante DEL tiroides" ->
    [hormona, estimulante, tiroides]."""
    texto = unicodedata.normalize("NFD", str(texto or "").lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return [t for t in re.split(r"[^a-z0-9]+", texto) if t and t not in PALABRAS_VACIAS]


# Nombres largos con que los laboratorios reales reportan algunos analitos (ej. el PDF de Compensar de prueba).
# Van aquí y no en el catálogo para no cambiar el dataset sintético de la Fase 1, que sortea entre sus nombres.
ALIAS_EXTRA = {
    "colesterol_hdl": ["Colesterol de alta densidad", "HDL colesterol"],
    "colesterol_ldl": ["Colesterol de baja densidad", "LDL colesterol"],
    "tsh": ["Hormona estimulante del tiroides", "Tirotropina"],
    "t4_libre": ["Tiroxina libre", "T4L"],
    "glucosa": ["Glicemia", "Glucemia"],
    "hemoglobina_a1c": ["Hemoglobina glicosilada", "Hemoglobina glucosilada", "HbA1c"],
    "bun": ["Nitrógeno ureico"],
    # "ALANINO AMINO TRANSFERASA (ALAT - TGP)" y "ASPARTATO AMINO TRANSFERASA (ASAT - TGO)": PDF real del 2026-11-02
    "alt": ["Transaminasa glutámico pirúvica", "TGP", "ALAT", "Alanino amino transferasa", "Alanina amino transferasa"],
    "ast": ["Transaminasa glutámico oxalacética", "TGO", "ASAT", "Aspartato amino transferasa"],
}


def indice_alias(catalogo):
    """[(tokens del alias, clave del analito)] con los nombres del catálogo y ALIAS_EXTRA."""
    nombres = [(n, clave) for clave, a in catalogo["analitos"].items() for n in a["nombres"]]
    nombres += [(n, clave) for clave, extra in ALIAS_EXTRA.items() if clave in catalogo["analitos"] for n in extra]
    return list(dict.fromkeys((tuple(tokens(n)), clave) for n, clave in nombres if tokens(n)))


def contiene(secuencia, sub):
    n = len(sub)
    return any(tuple(secuencia[i:i + n]) == sub for i in range(len(secuencia) - n + 1))


def analitos_del_nombre(examen, alias, claves_orina=(), tipo_examen=""):
    """
    Analitos del catálogo que nombra el examen: los del alias más largo que aparece como palabras
    completas y seguidas dentro del nombre. Si hay empate entre un analito de sangre y uno de orina
    ("Glucosa"; o "colesterol" frente a "densidad" en "COLESTEROL DE ALTA DENSIDAD"), gana el de orina solo
    si el nombre o el tipo de examen hablan de orina sin hablar de suero o sangre. La sangre se mira primero
    porque los laboratorios escriben "GLUCOSA EN SUERO U OTRO FLUIDO DIFERENTE A ORINA".
    """
    palabras = tokens(examen)
    coincidencias = [(len(t), clave) for t, clave in alias if contiene(palabras, t)]
    if not coincidencias:
        return set()
    largo = max(n for n, _ in coincidencias)
    claves = {clave for n, clave in coincidencias if n == largo}
    orina = claves & set(claves_orina)
    sangre = claves - orina
    if orina and sangre:
        contexto = set(palabras) | set(tokens(tipo_examen))
        es_orina = bool(contexto & PALABRAS_ORINA) and not contexto & PALABRAS_SANGRE
        claves = orina if es_orina else sangre
    return claves


def faltantes():
    """Modelos que no están en disco; la app usa esto para ofrecer o no el modo propio."""
    revisar = {"rag": RUTAS["rag"] / "model.safetensors", "guardrail": RUTAS["guardrail"] / "model.safetensors",
               "generador": RUTAS["generador"]}
    return [nombre for nombre, ruta in revisar.items() if not ruta.exists()]


class RecuperadorRAG:
    """Primer fragmento de MedlinePlus para "{examen} ({tipo de examen})", como en el entrenamiento del generador."""

    def __init__(self):
        import faiss
        from recuperador import CATALOGO_PATH, RecuperadorDenso, cargar_documentos

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

        catalogo = json.loads(Path(CATALOGO_PATH).read_text(encoding="utf-8"))
        self.alias = indice_alias(catalogo)
        self.claves_orina = set(catalogo["paneles"].get("Análisis de Orina Completo", []))

    def analitos_del_nombre(self, examen, tipo_examen=""):
        return analitos_del_nombre(examen, self.alias, self.claves_orina, tipo_examen)

    def buscar(self, consulta, examen=None, tipo_examen="", k=None):
        """
        Búsqueda densa con dos protecciones (medidas con el PDF real de prueba, 2026-11-02):

        1. Alias del catálogo como palabras completas dentro del nombre del laboratorio
           ("GLUCOSA EN SUERO U OTRO FLUIDO..." contiene "glucosa"): se devuelve el mejor fragmento de ese
           analito, aunque no esté entre los primeros. Antes solo se usaba el nombre exacto, y ese examen
           terminaba en la página de glucosa en orina.
        2. Sin alias, el primer resultado solo se acepta si su similitud supera UMBRAL_SIN_ALIAS. Si no,
           devuelve None: es mejor explicar sin guía que con la de otro examen (IGF-1 recibía la de creatinina).
        """
        kwargs = {"prompt_name": "query"} if self.usa_prompt else {}
        vector = self.modelo.encode([consulta], normalize_embeddings=True, **kwargs)
        # Son 208 fragmentos: ordenarlos todos cuesta lo mismo que los 5 primeros
        similitudes, posiciones = self.indice.search(vector, k or len(self.documentos))
        candidatos = [(float(s), self.documentos[i]) for s, i in zip(similitudes[0], posiciones[0]) if i >= 0]
        validos = self.analitos_del_nombre(examen, tipo_examen) if examen else set()
        if validos:
            con_alias = next((d for _, d in candidatos if validos & set(d["analitos"])), None)
            if con_alias is not None:
                return con_alias
        if not candidatos or candidatos[0][0] < UMBRAL_SIN_ALIAS:
            return None
        return candidatos[0][1]

    def recuperar(self, analitos, tipo_examen):
        """Un pasaje por analito y en su orden. Los que no tienen guía llevan texto vacío y sin_cobertura."""
        pasajes = []
        for a in analitos:
            doc = self.buscar(f"{a['examen']} ({tipo_examen})", examen=a["examen"], tipo_examen=tipo_examen)
            if doc is None:
                pasajes.append({"examen": a["examen"], "titulo": "", "fuente": None, "texto": "", "sin_cobertura": True})
                continue
            pasajes.append({"examen": a["examen"], "titulo": doc["titulo"], "fuente": doc.get("url") or doc.get("fuente"),
                            "texto": recortar(doc["texto"], MAX_CARACTERES_PASAJE)})
        return pasajes


def registrar_dlls_cuda():
    """
    La rueda CUDA de llama-cpp-python no trae cudart ni cublas. Con los paquetes de pip
    nvidia-cuda-runtime-cu12 y nvidia-cublas-cu12 (requirements-gpu.txt) quedan en site-packages/nvidia/*/bin:
    se registran antes de importar llama_cpp para no tener que instalar el CUDA Toolkit.
    """
    if os.name != "nt":
        return
    for base in sys.path:
        for carpeta in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(carpeta)
            except OSError:
                continue
            if carpeta not in os.environ.get("PATH", ""):
                os.environ["PATH"] = carpeta + os.pathsep + os.environ.get("PATH", "")


def capas_gpu_por_defecto():
    """-1 (todas las capas en la GPU) si llama.cpp se compiló con CUDA; 0 si es la versión de CPU.
    LDM_GPU_CAPAS lo fuerza (0 = solo CPU)."""
    if os.environ.get("LDM_GPU_CAPAS"):
        return int(os.environ["LDM_GPU_CAPAS"])
    try:
        import llama_cpp
        return -1 if llama_cpp.llama_supports_gpu_offload() else 0
    except Exception:
        return 0


class GeneradorPropio:
    """
    Qwen2.5-3B + LoRA en GGUF (llama.cpp), con decodificación voraz.

    Velocidad (examen real de 8 analitos, medido el 2026-11-02 en un i5-13420H y una RTX 4050):
    la gramática JSON de llama.cpp se evalúa en CPU sobre los ~151 mil tokens del vocabulario de Qwen en cada
    paso, y era el cuello de botella (CPU 159 s -> 85 s sin ella; GPU 175 s -> 28 s). El modelo ajustado ya
    escribe JSON válido por sí solo (99,3 % en el test de la Fase 4) y en CPU la salida es idéntica con y sin
    gramática. Por eso se genera sin gramática y solo si el JSON sale inválido se repite con ella.
    """

    def __init__(self, hilos=None, capas_gpu=None):
        registrar_dlls_cuda()
        from llama_cpp import Llama
        if hilos is None:
            # llama.cpp rinde mejor con los núcleos físicos que con todos los hilos lógicos (69 s frente a 75 s por caso)
            try:
                import psutil
                hilos = psutil.cpu_count(logical=False)
            except ImportError:
                hilos = None
            hilos = hilos or os.cpu_count()
        self.n_ctx = N_CTX_GENERADOR
        self.capas_gpu = capas_gpu_por_defecto() if capas_gpu is None else capas_gpu
        try:
            self.llm = Llama(model_path=str(RUTAS["generador"]), n_ctx=self.n_ctx, n_threads=hilos,
                             n_gpu_layers=self.capas_gpu, verbose=False)
        except Exception as e:
            if not self.capas_gpu:
                raise
            # GPU sin memoria libre o driver incompatible: se sigue en CPU en vez de no arrancar
            print(f"No se pudo cargar el generador en la GPU ({e}); se usa la CPU", flush=True)
            self.capas_gpu = 0
            self.llm = Llama(model_path=str(RUTAS["generador"]), n_ctx=self.n_ctx, n_threads=hilos,
                             n_gpu_layers=0, verbose=False)
        self.dispositivo = "GPU" if self.capas_gpu else "CPU"

    def tokens_prompt(self, mensajes):
        # Cuenta el contenido más un margen por las marcas de la plantilla de chat (<|im_start|>, roles...)
        texto = "\n".join(m["content"] for m in mensajes)
        return len(self.llm.tokenize(texto.encode("utf-8"), add_bos=False)) + 16 * len(mensajes)

    def generar(self, caso, pasajes, max_tokens=1200):
        """
        Si el prompt más la respuesta no caben en el contexto, se acortan los pasajes (primero a la mitad,
        luego sin pasajes) en vez de dejar que llama.cpp falle o corte el JSON a la mitad.
        """
        mensajes = None
        for limite in (MAX_CARACTERES_PASAJE, MAX_CARACTERES_PASAJE // 2, 0):
            cortos = [{**p, "texto": recortar(p["texto"], limite)} for p in pasajes] if limite else []
            mensajes = mensajes_estudiante(caso, cortos)
            disponibles = self.n_ctx - self.tokens_prompt(mensajes)
            if disponibles >= MIN_TOKENS_RESPUESTA:
                break
        else:
            raise ValueError(f"El caso no cabe en el contexto de {self.n_ctx} tokens ni siquiera sin pasajes")
        limite_tokens = min(max_tokens, disponibles)
        datos = self._completar(mensajes, limite_tokens, gramatica=False)
        if datos is None:
            print("JSON inválido sin gramática; se repite el lote con gramática", flush=True)
            datos = self._completar(mensajes, limite_tokens, gramatica=True)
        return datos or {}

    def _completar(self, mensajes, max_tokens, gramatica):
        formato = {"response_format": {"type": "json_object"}} if gramatica else {}
        respuesta = self.llm.create_chat_completion(messages=mensajes, temperature=0, max_tokens=max_tokens, **formato)
        return parsear(respuesta["choices"][0]["message"]["content"] or "")


class Guardrail:
    """Marca las oraciones con P(diagnóstica) + P(alarmista) >= UMBRAL_RIESGO."""

    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizador = AutoTokenizer.from_pretrained(RUTAS["guardrail"])
        self.modelo = AutoModelForSequenceClassification.from_pretrained(RUTAS["guardrail"]).eval()
        self.clases = [self.modelo.config.id2label[i] for i in range(self.modelo.config.num_labels)]

    def riesgo(self, oraciones, lote_max=32):
        """P(riesgosa) por oración. En lotes de `lote_max` para que un examen largo no dispare la RAM."""
        import torch
        riesgos = []
        with torch.no_grad():
            for i in range(0, len(oraciones), lote_max):
                lote = self.tokenizador(oraciones[i:i + lote_max], truncation=True, max_length=96,
                                        padding=True, return_tensors="pt")
                probs = self.modelo(**lote).logits.softmax(-1)
                riesgos += (1 - probs[:, self.clases.index("informativa")]).tolist()
        return riesgos

    def revisar(self, resultado):
        """Agrega a cada explicación la lista de oraciones marcadas (vacía si no hay)."""
        explicaciones = resultado.get("explicaciones", [])
        oraciones = [dividir_oraciones(e.get("explicacion", "")) for e in explicaciones]
        riesgos = iter(self.riesgo([o for frases in oraciones for o in frases]))
        for e, frases in zip(explicaciones, oraciones):
            e["oraciones_marcadas"] = [o for o in frases if next(riesgos) >= UMBRAL_RIESGO]
        return resultado
