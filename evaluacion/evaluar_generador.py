"""
Fase 4: compara los generadores (modelo base, modelo ajustado, Gemini, profesor) sobre los casos de test.

Lee cada archivo de data/generador/salidas/*.jsonl y mide, con training/generador/verificacion.py:
- JSON válido: % de respuestas que se pueden usar en la app.
- Cobertura: % de analitos con explicación.
- Menciona el valor y estado consistente: fidelidad a los datos del examen.
- INFLESZ: legibilidad en español (más alto = más fácil; 55-65 es "normal", 65-80 "bastante fácil").
- Guardrails: % de oraciones y de respuestas con al menos una oración marcada como riesgosa por el
  clasificador de la Fase 3 (por defecto MrBERT-biomed v2, P(riesgosa) >= 0,5).

Uso:
    python evaluacion/evaluar_generador.py [--sin-guardrails]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "training" / "generador"))
sys.path.insert(0, str(RAIZ / "training" / "guardrails"))
sys.path.insert(0, str(RAIZ / "evaluacion"))

from verificacion import calidad, frases_para_guardrails, inflesz, parsear, verificar  # noqa: E402

CASOS = RAIZ / "data" / "generador" / "casos.jsonl"
SALIDAS = RAIZ / "data" / "generador" / "salidas"
RESULTADOS = RAIZ / "evaluacion" / "resultados" / "fase4_generador.json"
UMBRAL_RIESGO = 0.5


def media(valores):
    valores = [v for v in valores if v is not None]
    return statistics.mean(valores) if valores else None


def evaluar_archivo(filas, casos, clasificar):
    verificaciones = [verificar(r["respuesta"], casos[r["id"]]) for r in filas]
    validas = [(r, v) for r, v in zip(filas, verificaciones) if v["json_valido"]]
    textos = [" ".join(e["explicacion"] for e in parsear(r["respuesta"])["explicaciones"]) for r, _ in validas]
    resultado = {
        "n": len(filas),
        "json_valido": len(validas) / len(filas),
        # Las métricas de contenido se calculan sobre las respuestas válidas; una inválida cuenta como 0 en cobertura
        "cobertura": sum(v["cobertura"] for _, v in validas) / len(filas),
        "menciona_valor": media(v["menciona_valor"] for _, v in validas),
        "estado_consistente": media(v["estado_consistente"] for _, v in validas),
        "inflesz": media(inflesz(t) for t in textos),
        "palabras_por_respuesta": media(len(t.split()) for t in textos),
        "segundos_por_caso": media(r.get("segundos") for r in filas),
    }
    # Calidad (generador v2): errores vistos con PDFs reales que las métricas de arriba no detectan.
    # Todas son "menos es mejor", salvo menciona_valor_alterados.
    calidades = [calidad(r["respuesta"], casos[r["id"]]) for r, _ in validas]
    alterados = [d["menciona_valor"] for r, v in validas
                 for d, a in zip(v["detalle"], casos[r["id"]]["analitos"]) if a["estado"] != "NORMAL"]
    n = len(validas) or 1
    resultado.update({
        "menciona_valor_alterados": media(alterados),
        "causas_repetidas": sum(bool(q["causas_repetidas"]) for q in calidades) / n,
        "siglas_inventadas": sum(bool(q["siglas_inventadas"]) for q in calidades) / n,
        "usa_usted": sum(q["usa_usted"] for q in calidades) / n,
        "respuestas_con_pregunta_quitada": sum(q["preguntas_quitadas"] > 0 for q in calidades) / n,
        "ejemplos_siglas_inventadas": [s for q in calidades for s in q["siglas_inventadas"]][:10],
        "ejemplos_causas_repetidas": [s for q in calidades for s in q["causas_repetidas"]][:10],
        # v3: oraciones de causas (de exámenes alterados) cuyas causas concretas no están en el pasaje del RAG
        "causas_sin_respaldo": (sum(len(q["causas_sin_respaldo"]) for q in calidades)
                                / max(sum(q["causas_evaluadas"] for q in calidades), 1)),
        "ejemplos_causas_sin_respaldo": [f"{e}: {o}" for q in calidades for e, o in q["causas_sin_respaldo"]][:10],
    })
    if clasificar:
        frases = [frases_para_guardrails(r["respuesta"]) for r, _ in validas]
        riesgo = iter(clasificar([o for fs in frases for o in fs]))
        marcadas = [[o for o in fs if next(riesgo) >= UMBRAL_RIESGO] for fs in frases]
        total = sum(map(len, frases))
        resultado["oraciones_marcadas"] = sum(map(len, marcadas)) / total if total else None
        resultado["respuestas_con_marca"] = sum(bool(m) for m in marcadas) / len(validas) if validas else None
        resultado["ejemplos_marcados"] = [o for m in marcadas for o in m][:20]
    return resultado


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sin-guardrails", action="store_true")
    parser.add_argument("--clasificador", default="mrbert-biomed-v2",
                        help="Guardrail de la Fase 3 (v2: entrenado también con texto real)")
    parser.add_argument("--version", default="v1", choices=["v1", "v3"],
                        help="v3: casos_v3.jsonl (pasajes de la app) y salidas_v3/ -> fase4_generador_v3.json")
    args = parser.parse_args()
    casos_ruta, salidas, destino = CASOS, SALIDAS, RESULTADOS
    if args.version == "v3":
        casos_ruta = CASOS.with_name("casos_v3.jsonl")
        salidas = SALIDAS.with_name("salidas_v3")
        destino = RESULTADOS.with_name("fase4_generador_v3.json")

    with open(casos_ruta, encoding="utf-8") as f:
        casos = {c["id"]: c for c in map(json.loads, f)}
    clasificar = None
    if not args.sin_guardrails:
        from evaluar_guardrails import probabilidades_modelo
        clasificar = lambda frases: [1 - p[0] for p in probabilidades_modelo(args.clasificador, frases)]  # noqa: E731

    resultados = {}
    for archivo in sorted(salidas.glob("*.jsonl")):
        with open(archivo, encoding="utf-8") as f:
            filas = [json.loads(linea) for linea in f]
        if filas:
            resultados[archivo.stem] = evaluar_archivo(filas, casos, clasificar)

    columnas = ["n", "json_valido", "cobertura", "menciona_valor", "menciona_valor_alterados", "estado_consistente",
                "inflesz", "oraciones_marcadas", "respuestas_con_marca", "causas_repetidas", "siglas_inventadas",
                "usa_usted", "respuestas_con_pregunta_quitada", "causas_sin_respaldo", "palabras_por_respuesta",
                "segundos_por_caso"]
    print(f"{'modelo':<34}" + "".join(f"{c[:12]:>13}" for c in columnas))
    for nombre, r in resultados.items():
        celdas = []
        for c in columnas:
            v = r.get(c)
            celdas.append("—" if v is None else f"{v:.0f}" if c in ("n", "palabras_por_respuesta", "segundos_por_caso")
                          else f"{v:.1f}" if c == "inflesz" else f"{v:.1%}")
        print(f"{nombre:<34}" + "".join(f"{x:>13}" for x in celdas))

    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump({"clasificador": None if args.sin_guardrails else args.clasificador, "casos": casos_ruta.name,
                   "resultados": resultados}, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
