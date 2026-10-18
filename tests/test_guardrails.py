"""
Pruebas de la Fase 3: armado del dataset de guardrails y baseline de palabras (sin llamar a Groq).

Ejecutar desde la raíz del proyecto:
    python -m pytest tests -v
"""
import csv
import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "training" / "guardrails"))
sys.path.insert(0, str(RAIZ / "evaluacion"))

from dataset import cargar_dataset  # noqa: E402
from evaluar_guardrails import predecir_palabras  # noqa: E402
from generar_frases import construir_prompt  # noqa: E402
from verificar_etiquetas import kappa_cohen  # noqa: E402

SPLIT = json.loads((RAIZ / "training" / "rag" / "split_analitos.json").read_text(encoding="utf-8"))
CATALOGO = json.loads((RAIZ / "training" / "sintetico" / "catalogo_analitos.json").read_text(encoding="utf-8"))


def _escribir(tmp_path, frases, juez, revision=None):
    rutas = {n: tmp_path / n for n in ("frases.jsonl", "juez.jsonl", "revisar.csv")}
    rutas["frases.jsonl"].write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in frases), encoding="utf-8")
    rutas["juez.jsonl"].write_text("".join(json.dumps({"frase": f, "clase_juez": c}) + "\n" for f, c in juez.items()), encoding="utf-8")
    if revision is not None:
        with open(rutas["revisar.csv"], "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frase", "clase_profesor", "clase_juez", "clase_equipo"])
            w.writerows(revision)
    return dict(frases_path=rutas["frases.jsonl"], juez_path=rutas["juez.jsonl"], revisar_path=rutas["revisar.csv"])


def test_etiqueta_final_prioriza_equipo_y_exige_acuerdo(tmp_path):
    analito = SPLIT["train"][0]
    frases = [{"frase": f"f{i}", "clase": c, "analito": analito}
              for i, c in enumerate(["informativa", "diagnostica", "alarmista", "diagnostica", "informativa"])]
    juez = {"f0": "informativa", "f1": "informativa", "f2": "alarmista", "f3": "alarmista"}  # f4 sin juez
    revision = [["f1", "diagnostica", "informativa", "informativa"], ["f3", "diagnostica", "alarmista", "descartar"]]
    dataset, motivos = cargar_dataset(**_escribir(tmp_path, frases, juez, revision))
    etiquetas = {d["frase"]: (d["clase"], d["origen"]) for d in dataset}
    assert etiquetas == {"f0": ("informativa", "acuerdo"), "f1": ("informativa", "equipo"), "f2": ("alarmista", "acuerdo")}
    assert motivos == {"descartada por el equipo": 1, "sin juez": 1}


def test_split_por_analito_manda_analitos_de_test_a_test(tmp_path):
    frases = [{"frase": f"{a}-{i}", "clase": "informativa", "analito": a}
              for a in CATALOGO["analitos"] for i in range(5)]
    juez = {d["frase"]: "informativa" for d in frases}
    dataset, _ = cargar_dataset(**_escribir(tmp_path, frases, juez))
    for d in dataset:
        assert (d["split"] == "test") == (d["analito"] in SPLIT["test"]), d
    assert {d["split"] for d in dataset} == {"train", "val", "test"}


def test_filtro_de_palabras_replica_el_backend():
    """Las debilidades que motivan el clasificador: falso positivo con 'tienes' y falso negativo sin palabras clave."""
    frases = ["Tienes el valor un poco por encima del rango.", "Esto es una anemia.", "Padeces diabetes."]
    assert predecir_palabras(frases) == [1, 0, 1]


def test_prompt_de_generacion_incluye_resultado_y_pistas():
    prompt, resultado = construir_prompt(random.Random(0), CATALOGO, "glucosa", "diagnostica")
    assert resultado["examen_canonico"] == "glucosa"
    assert resultado["estado"] in prompt and "\"diagnostica\"" in prompt and "sin usar las palabras" in prompt


def test_kappa_cohen():
    assert kappa_cohen(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1
    assert abs(kappa_cohen(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9
