"""
Fase 2 (ajuste contrastivo): fine-tuning de un modelo de embeddings con las consultas generadas.

Cada ejemplo es (consulta, fragmento positivo, fragmento negativo difícil). La pérdida
MultipleNegativesRankingLoss acerca la consulta a su fragmento y la aleja del negativo y de los
demás fragmentos del batch. Los negativos difíciles salen de las confusiones medidas en la
evaluación (ej. glucosa en sangre frente a glucosa en orina) o, si no hay, del mismo panel.

Variantes:
- split: entrena sin los analitos de test de split_analitos.json, para medir si el modelo
  generaliza a analitos que nunca vio (el experimento honesto).
- completo: entrena con todos los analitos (el modelo que usaría la app).

Uso:
    python training/rag/ajustar_embeddings.py --modelo alia --variante split
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "training" / "rag"))

from recuperador import MODELO_ALIA, cargar_documentos, texto_para_indexar  # noqa: E402

CONSULTAS_PATH = RAIZ / "data" / "rag" / "consultas_entrenamiento.jsonl"
SPLIT_PATH = Path(__file__).with_name("split_analitos.json")
CATALOGO_PATH = RAIZ / "training" / "sintetico" / "catalogo_analitos.json"
SALIDA_DIR = RAIZ / "models" / "rag"

MODELOS = {
    "alia": {"nombre": MODELO_ALIA, "prefijo_consulta": "", "prefijo_documento": ""},
    "e5-small": {"nombre": "intfloat/multilingual-e5-small", "prefijo_consulta": "query: ", "prefijo_documento": "passage: "},
}

# Pares que los modelos base confunden (ver fallos en evaluacion/resultados/fase2_rag.json)
CONFUSIONES = [
    ("glucosa", "glucosa_orina"), ("glucosa", "hemoglobina_a1c"),
    ("hemoglobina", "hemoglobina_a1c"), ("hemoglobina", "sangre_orina"),
    ("leucocitos", "leucocitos_sedimento"), ("leucocitos", "neutrofilos"),
    ("eritrocitos", "eritrocitos_sedimento"), ("eritrocitos", "hematocrito"),
    ("bilirrubina_total", "bilirrubina_orina"), ("alt", "ast"), ("alt", "albumina"),
    ("bun", "creatinina"), ("tsh", "t4_libre"), ("colesterol_total", "trigliceridos"),
]


def grupos_de_analitos(documentos):
    """Analitos que comparten algún fragmento forman un grupo (ej. HDL, LDL y colesterol total)."""
    grupo = {}
    for doc in documentos:
        actuales = {grupo.get(a, frozenset([a])) for a in doc["analitos"]}
        unido = frozenset().union(*actuales, doc["analitos"])
        for a in unido:
            grupo[a] = unido
    return sorted({g for g in grupo.values()}, key=sorted)


def crear_split(documentos, proporcion_test=0.3, seed=13):
    """Separa grupos completos para que ningún fragmento de test aparezca en entrenamiento."""
    grupos = grupos_de_analitos(documentos)
    rng = random.Random(seed)
    rng.shuffle(grupos)
    total = sum(len(g) for g in grupos)
    test = []
    for g in grupos:
        if sum(map(len, test)) + len(g) <= proporcion_test * total:
            test.append(g)
    analitos_test = sorted(a for g in test for a in g)
    analitos_train = sorted(a for g in grupos if g not in test for a in g)
    return {"seed": seed, "train": analitos_train, "test": analitos_test}


def cargar_split(documentos):
    if not SPLIT_PATH.exists():
        with open(SPLIT_PATH, "w", encoding="utf-8") as f:
            json.dump(crear_split(documentos), f, indent=2, ensure_ascii=False)
    with open(SPLIT_PATH, encoding="utf-8") as f:
        return json.load(f)


def elegir_negativo(rng, doc, documentos_por_analito, paneles_por_analito, permitidos):
    propios = set(doc["analitos"])
    candidatos = [b for a, b in CONFUSIONES if a in propios] + [a for a, b in CONFUSIONES if b in propios]
    if not candidatos:
        candidatos = sorted({o for a in propios for p in paneles_por_analito.get(a, []) for o in p})
    candidatos = [c for c in candidatos if c not in propios and c in permitidos]
    if not candidatos:
        candidatos = sorted(permitidos - propios)
    analito = rng.choice(candidatos)
    return rng.choice(documentos_por_analito[analito])


def construir_ejemplos(documentos, consultas, permitidos, prefijos, seed=13):
    rng = random.Random(seed)
    por_id = {d["id"]: d for d in documentos}
    documentos_por_analito = {}
    for d in documentos:
        for a in d["analitos"]:
            documentos_por_analito.setdefault(a, []).append(d)
    with open(CATALOGO_PATH, encoding="utf-8") as f:
        paneles = json.load(f)["paneles"].values()
    paneles_por_analito = {}
    for claves in paneles:
        for a in claves:
            paneles_por_analito.setdefault(a, []).append(claves)

    anclas, positivos, negativos = [], [], []
    for c in consultas:
        doc = por_id[c["doc_id"]]
        if not set(doc["analitos"]) <= permitidos:
            continue
        negativo = elegir_negativo(rng, doc, documentos_por_analito, paneles_por_analito, permitidos)
        anclas.append(prefijos["prefijo_consulta"] + c["consulta"])
        positivos.append(prefijos["prefijo_documento"] + texto_para_indexar(doc))
        negativos.append(prefijos["prefijo_documento"] + texto_para_indexar(negativo))
    return anclas, positivos, negativos


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", choices=MODELOS, required=True)
    parser.add_argument("--variante", choices=["split", "completo"], required=True)
    parser.add_argument("--epocas", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-seq", type=int, default=384, help="Truncado durante el entrenamiento, por velocidad")
    args = parser.parse_args()

    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    documentos = cargar_documentos()
    with open(CONSULTAS_PATH, encoding="utf-8") as f:
        consultas = [json.loads(linea) for linea in f]
    todos = {a for d in documentos for a in d["analitos"]}
    split = cargar_split(documentos)
    permitidos = set(split["train"]) if args.variante == "split" else todos

    config = MODELOS[args.modelo]
    anclas, positivos, negativos = construir_ejemplos(documentos, consultas, permitidos, config)
    print(f"{len(anclas)} ejemplos de entrenamiento ({len(permitidos)} analitos permitidos)", flush=True)

    modelo = SentenceTransformer(config["nombre"])
    max_seq_original = modelo.max_seq_length
    modelo.max_seq_length = min(args.max_seq, max_seq_original)

    salida = SALIDA_DIR / f"{args.modelo}-ajustado-{args.variante}"
    argumentos = SentenceTransformerTrainingArguments(
        output_dir=str(salida / "checkpoints"),
        num_train_epochs=args.epocas,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_steps=0.1,  # En transformers 5 un float es la proporción de pasos de calentamiento
        # Sin duplicados en un batch: dos consultas del mismo fragmento serían falsos negativos entre sí
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=13,
    )
    entrenador = SentenceTransformerTrainer(
        model=modelo,
        args=argumentos,
        train_dataset=Dataset.from_dict({"ancla": anclas, "positivo": positivos, "negativo": negativos}),
        loss=losses.MultipleNegativesRankingLoss(modelo),
    )
    inicio = time.time()
    entrenador.train()
    minutos = (time.time() - inicio) / 60

    # Para indexar se usa el texto completo, igual que con el modelo base
    modelo.max_seq_length = max_seq_original
    modelo.save(str(salida))
    with open(salida / "entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump({"modelo_base": config["nombre"], "variante": args.variante, "ejemplos": len(anclas),
                   "analitos_entrenamiento": sorted(permitidos), "epocas": args.epocas, "lr": args.lr,
                   "batch": args.batch, "max_seq_entrenamiento": args.max_seq, "minutos": round(minutos, 1),
                   "perdida_por_paso": [h for h in entrenador.state.log_history if "loss" in h]},
                  f, indent=2, ensure_ascii=False)
    print(f"Modelo guardado en {salida.relative_to(RAIZ)} ({minutos:.1f} min)")


if __name__ == "__main__":
    main()
