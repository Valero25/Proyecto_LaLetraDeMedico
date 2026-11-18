"""
Fase 2, teoría (fundamentos.md, sección 1): ¿crecen los pesos con el ajuste contrastivo del RAG?

La regla de Hebb pura no tiene límite: los pesos de las conexiones que se activan juntas crecen sin parar. El
decaimiento de pesos (weight decay) es una de las formas de frenarlo. Este script compara, capa por capa, los pesos
del modelo base de ALIA con los de los modelos ajustados y mide:

- la norma L2 de los pesos antes y después (¿crecieron?);
- el cambio relativo ||W_ajustado - W_base|| / ||W_base|| (¿cuánto se alejaron del punto de partida?);
- cuánto habría encogido los pesos un weight decay de AdamW: en cada paso, w <- w - lr_t * wd * w, así que el
  factor total es prod(1 - lr_t * wd), con el calendario de tasas de aprendizaje que registró el entrenamiento.

Solo lee archivos safetensors: no necesita GPU.

    python evaluacion/normas_pesos_rag.py [--modelos alia-ajustado-split alia-ajustado-completo]
"""
import argparse
import json
import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "evaluacion" / "resultados" / "fase2_normas_pesos.json"
MODELO_BASE = "SINAI/ALIA-MrBERT-es-biomedical-embeddings"


def cargar_pesos(ruta):
    from safetensors.torch import load_file
    pesos = load_file(str(ruta))
    # Doble precisión: en float32, sumar 39 millones de cuadrados (los embeddings) da error de redondeo.
    # Sentence-Transformers a veces antepone "0." o "model." al nombre de cada tensor
    return {k.removeprefix("0.").removeprefix("model."): v.double() for k, v in pesos.items()}


def ruta_base():
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MODELO_BASE, "model.safetensors")


def factor_decaimiento(entrenamiento, wd):
    """prod(1 - lr_t * wd) sobre los pasos. Interpola el calendario lineal entre los registros de la pérdida."""
    # El registro es cada 10 pasos: el total sale de los pasos por época del último registro
    ultimo = entrenamiento["perdida_por_paso"][-1]
    pasos_totales = round(ultimo["step"] / ultimo["epoch"] * entrenamiento["epocas"])
    lr_max = entrenamiento["lr"]
    calentamiento = max(1, round(0.1 * pasos_totales))
    log = 0.0
    for paso in range(1, pasos_totales + 1):
        # Calendario lineal de transformers: sube hasta lr_max en el calentamiento y baja a 0 al final
        lr = lr_max * paso / calentamiento if paso <= calentamiento else \
            lr_max * max(0.0, (pasos_totales - paso) / max(1, pasos_totales - calentamiento))
        log += math.log(1 - lr * wd)
    return math.exp(log), pasos_totales


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelos", nargs="+", default=["alia-ajustado-split", "alia-ajustado-completo"])
    parser.add_argument("--wd", type=float, nargs="+", default=[0.01, 0.1])
    args = parser.parse_args()

    base = cargar_pesos(ruta_base())
    norma_base = math.sqrt(sum(float((v ** 2).sum()) for v in base.values()))
    resultados = {"modelo_base": MODELO_BASE, "parametros": sum(v.numel() for v in base.values()),
                  "norma_base": norma_base, "modelos": {}}
    print(f"Base: {resultados['parametros']:,} parámetros, norma L2 total {norma_base:.2f}")

    for nombre in args.modelos:
        carpeta = RAIZ / "models" / "rag" / nombre
        ajustado = cargar_pesos(carpeta / "model.safetensors")
        comunes = sorted(set(base) & set(ajustado))
        faltan = sorted(set(base) ^ set(ajustado))
        norma2_aj, delta2 = 0.0, 0.0
        por_tensor = []
        for k in comunes:
            w0, w1 = base[k], ajustado[k]
            n0, n1 = float(w0.norm()), float(w1.norm())
            d = float((w1 - w0).norm())
            norma2_aj += n1 ** 2
            delta2 += d ** 2
            por_tensor.append({"tensor": k, "norma_base": n0, "norma_ajustado": n1,
                               "cambio_norma": (n1 - n0) / n0 if n0 else 0.0, "cambio_relativo": d / n0 if n0 else 0.0})
        norma_aj, delta = math.sqrt(norma2_aj), math.sqrt(delta2)
        crecen = sum(t["norma_ajustado"] > t["norma_base"] for t in por_tensor)
        entrenamiento = json.loads((carpeta / "entrenamiento.json").read_text(encoding="utf-8"))
        decaimiento = {}
        for wd in args.wd:
            factor, pasos = factor_decaimiento(entrenamiento, wd)
            # Si el decaimiento actuara solo, encogería cada peso en (1 - factor): ||W|| * (1 - factor)
            decaimiento[str(wd)] = {"factor": factor, "reduccion_relativa": 1 - factor,
                                    "reduccion_norma": norma_base * (1 - factor), "pasos": pasos}
        r = {
            "tensores_comparados": len(comunes), "tensores_distintos": faltan,
            "norma_ajustado": norma_aj, "cambio_norma_total": (norma_aj - norma_base) / norma_base,
            "cambio_relativo_total": delta / norma_base, "norma_del_cambio": delta,
            "tensores_que_crecen": crecen,
            "mayores_cambios_relativos": sorted(por_tensor, key=lambda t: -t["cambio_relativo"])[:5],
            "mayores_crecimientos": sorted(por_tensor, key=lambda t: -t["cambio_norma"])[:5],
            "weight_decay_hipotetico": decaimiento,
        }
        resultados["modelos"][nombre] = r
        print(f"\n== {nombre} ({entrenamiento['epocas']} épocas, lr {entrenamiento['lr']}, "
              f"weight decay {entrenamiento.get('weight_decay', 0.0)})")
        print(f"   norma L2: {norma_base:.2f} -> {norma_aj:.2f} ({r['cambio_norma_total']:+.4%})")
        print(f"   ||W_ajustado - W_base|| = {delta:.3f} ({r['cambio_relativo_total']:.4%} de ||W_base||)")
        print(f"   tensores cuya norma crece: {crecen} de {len(comunes)}")
        for wd, dd in decaimiento.items():
            print(f"   weight decay {wd}: encogería cada peso un {dd['reduccion_relativa']:.5%} "
                  f"(norma {dd['reduccion_norma']:.3f}) en {dd['pasos']} pasos")
        if faltan:
            print(f"   tensores que no coinciden: {faltan[:5]}")

    SALIDA.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
