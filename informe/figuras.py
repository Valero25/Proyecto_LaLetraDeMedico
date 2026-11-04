"""
Fase 6: figuras del informe a partir de los resultados medidos (evaluacion/resultados/*.json).

    python informe/figuras.py

Escribe los PNG en informe/figuras/. Las curvas de entrenamiento viven en models/*/entrenamiento.json, que no se
versiona: la primera vez se copian a evaluacion/resultados/curvas.json, y desde ahí se pueden regenerar las
figuras sin los modelos.

Colores: los tres primeros de la paleta categórica validada (azul, naranja, aqua; pasan todas las parejas para
daltonismo) y gris para las líneas base. El aqua tiene menos de 3:1 de contraste con el fondo, por eso todas las
barras llevan su valor escrito.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "evaluacion" / "resultados"
SALIDA = RAIZ / "informe" / "figuras"

AZUL, NARANJA, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#b4b2ab"
FONDO, TEXTO, TEXTO_2, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXTO,
    "axes.edgecolor": REJILLA, "axes.labelcolor": TEXTO_2, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 10, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": TEXTO_2, "ytick.color": TEXTO_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 9.5,
})


def leer(nombre):
    return json.loads((RESULTADOS / nombre).read_text(encoding="utf-8"))


def guardar(fig, nombre, fuente):
    # Debajo de todo (bbox_inches="tight" lo incluye): así nunca pisa el título del eje x
    fig.text(0.01, -0.04, f"Fuente: {fuente}", fontsize=8, color=TEXTO_2)
    fig.savefig(SALIDA / nombre, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ", SALIDA / nombre)


def rejilla(ax, eje="x"):
    ax.grid(axis=eje, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def barras_agrupadas(ax, etiquetas, series, formato="{:.0%}", limite=1.0):
    """Barras horizontales agrupadas. `series` = [(nombre, color, valores)]. Un hueco entre barras y el valor escrito."""
    n = len(series)
    alto = 0.8 / n
    for k, (nombre, color, valores) in enumerate(series):
        posiciones = [i + (k - (n - 1) / 2) * alto for i in range(len(etiquetas))]
        barras = ax.barh(posiciones, valores, height=alto * 0.86, color=color, label=nombre)
        for barra, valor in zip(barras, valores):
            ax.text(barra.get_width() + limite * 0.01, barra.get_y() + barra.get_height() / 2, formato.format(valor),
                    va="center", fontsize=8.5, color=TEXTO)
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    ax.invert_yaxis()
    ax.set_xlim(0, limite * 1.12)
    rejilla(ax)


# --- 1. RAG ---------------------------------------------------------------------------------------------------
def figura_rag():
    r = leer("fase2_rag.json")["resultados"]
    filas = [
        ("subcadena", "Subcadena (sistema original)"),
        ("bm25+alias", "BM25 + alias"),
        ("e5-small+alias", "e5-small + alias"),
        ("alia+alias", "ALIA-MrBERT + alias"),
        ("e5-small-ajustado-split+alias", "e5-small ajustado (split)"),
        ("alia-ajustado-split+alias", "ALIA ajustado (split)"),
        ("alia-ajustado-completo+alias", "ALIA ajustado (completo, en la app)"),
    ]
    filas = [(k, e) for k, e in filas if k in r]
    total = [r[k]["metricas"]["total"]["hit@1"] for k, _ in filas]
    manual = [r[k]["metricas"]["manual"]["hit@1"] for k, _ in filas]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    barras_agrupadas(ax, [e for _, e in filas], [("Todas las consultas (505)", AZUL, total),
                                                ("Consultas manuales, coloquiales (40)", NARANJA, manual)])
    ax.set_title("RAG: acierto en el primer resultado (hit@1)", pad=30)
    ax.set_xlabel("hit@1")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    # Entre el título y el gráfico: dentro tapaba las barras del modelo final y debajo chocaba con la fuente
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2, handlelength=1.2)
    guardar(fig, "fig1_rag.png", "evaluacion/resultados/fase2_rag.json")


# --- 2. Guardrail -----------------------------------------------------------------------------------------------
def figura_guardrail():
    sint = leer("fase3_guardrails.json")["resultados"]
    real = leer("fase3_guardrails_real.json")["resultados"]
    modelos = [("palabras", "Filtro de palabras (original)"), ("tfidf", "TF-IDF + regresión logística"),
               ("bilstm", "BiLSTM propia (desde cero)"), ("mrbert-es", "MrBERT-es ajustado"),
               ("mrbert-biomed", "MrBERT-biomed ajustado (v1)"), ("mrbert-biomed-v2", "MrBERT-biomed v2 (+ texto real)")]
    modelos = [(k, e) for k, e in modelos if k in sint]
    fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    barras_agrupadas(izq, [e for _, e in modelos], [("F1 binaria (¿riesgosa?)", AZUL,
                                                     [sint[k]["binaria"]["f1"] for k, _ in modelos])], "{:.2f}")
    izq.set_title("Frases sintéticas de analitos no vistos (506)")
    izq.set_xlabel("F1 binaria")
    reales = [(k, e) for k, e in [("palabras", "Filtro de palabras"), ("tfidf", "TF-IDF"),
                                  ("mrbert-biomed", "MrBERT-biomed v1"), ("mrbert-biomed-v2", "MrBERT-biomed v2")]
              if k in real]
    n_inf = real[reales[0][0]].get("tres_clases", {}).get("informativa", {}).get("support", 166)
    # Solo con falsos positivos el filtro de palabras parecería el mejor, pero no detecta ninguna riesgosa:
    # cada etiqueta dice cuántas de las riesgosas reales detecta
    con_clases = next(real[k]["tres_clases"] for k, _ in reales if "tres_clases" in real[k])
    n_riesgosas = int(con_clases["diagnostica"]["support"] + con_clases["alarmista"]["support"])
    reales = [(k, e + "\n" + f"(detecta {n_riesgosas - real[k]['binaria']['falsos_negativos']} de {n_riesgosas} riesgosas)")
              for k, e in reales]
    barras_agrupadas(der, [e for _, e in reales], [("Falsos positivos", NARANJA,
                                                    [real[k]["binaria"]["falsos_positivos"] for k, _ in reales])],
                     "{:.0f}", limite=max(real[k]["binaria"]["falsos_positivos"] for k, _ in reales))
    der.set_title(f"Texto real de los generadores ({int(n_inf)} informativas)")
    der.set_xlabel("Oraciones informativas marcadas por error (menos es mejor)")
    fig.suptitle("Guardrail: casi perfecto en sintético, pero solo el v2 sirve en texto real",
                 x=0.01, ha="left", fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    guardar(fig, "fig2_guardrail.png", "evaluacion/resultados/fase3_guardrails.json y fase3_guardrails_real.json")


# --- 3. Generador -----------------------------------------------------------------------------------------------
def figura_generador():
    r = leer("fase4_generador.json")["resultados"]
    base, ajustado = r["qwen2.5-3b-base"], r["qwen2.5-3b-ajustado"]
    metricas = [("json_valido", "JSON válido"), ("cobertura", "Un examen por analito"),
                ("estado_consistente", "Respeta el estado calculado"), ("menciona_valor", "Menciona el valor del paciente"),
                ("respuestas_con_marca", "Respuestas con una oración riesgosa\n(guardrail v2; menos es mejor)")]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    barras_agrupadas(ax, [e for _, e in metricas], [("Qwen2.5-3B base", GRIS, [base[k] for k, _ in metricas]),
                                                    ("Qwen2.5-3B ajustado con QLoRA", AZUL, [ajustado[k] for k, _ in metricas])])
    ax.set_title(f"Generador: base frente a ajustado ({ajustado['n']} casos de test)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower right")
    guardar(fig, "fig3_generador.png", "evaluacion/resultados/fase4_generador.json")


def figura_generador_v2():
    """Base, v1 y v2 sobre los mismos 150 casos de test, en las métricas que el v2 busca mejorar."""
    r = leer("fase4_generador.json")["resultados"]
    if "qwen2.5-3b-ajustado-v2" not in r:
        print("   (falta data/generador/salidas/qwen2.5-3b-ajustado-v2.jsonl evaluado: se omite fig6)")
        return
    modelos = [("qwen2.5-3b-base", "Qwen2.5-3B base", GRIS), ("qwen2.5-3b-ajustado", "Ajustado v1 (555 ejemplos)", AZUL),
               ("qwen2.5-3b-ajustado-v2", "Ajustado v2 (421 ejemplos limpios)", NARANJA)]
    metricas = [("estado_consistente", "Respeta el estado calculado"),
                ("menciona_valor_alterados", "Menciona el valor en los\nexámenes fuera de rango"),
                ("causas_repetidas", "Respuestas con causas repetidas\n(menos es mejor)"),
                ("siglas_inventadas", "Respuestas con una sigla inventada\n(menos es mejor)"),
                ("respuestas_con_pregunta_quitada", "Respuestas con una pregunta incoherente\n(menos es mejor)"),
                ("respuestas_con_marca", "Respuestas con una oración riesgosa\n(guardrail v2; menos es mejor)")]
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    barras_agrupadas(ax, [e for _, e in metricas],
                     [(nombre, color, [r[k].get(m) or 0 for m, _ in metricas]) for k, nombre, color in modelos])
    ax.set_title(f"Generador v2: mismos hiperparámetros, datos más limpios ({r['qwen2.5-3b-ajustado-v2']['n']} casos de test)",
                 pad=30)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2, handlelength=1.2, fontsize=8.5)
    guardar(fig, "fig6_generador_v2.png", "evaluacion/resultados/fase4_generador.json")


# --- 4. Curvas de entrenamiento -----------------------------------------------------------------------------------
def curvas():
    """Copia las curvas de models/ a evaluacion/resultados/curvas.json (versionado) y las devuelve."""
    destino = RESULTADOS / "curvas.json"
    fuentes = {"qlora": RAIZ / "models" / "generador" / "qwen2.5-3b-instruct-lora" / "entrenamiento.json",
               "qlora_v2": RAIZ / "models" / "generador" / "qwen2.5-3b-instruct-lora-v2" / "entrenamiento.json",
               "bilstm": RAIZ / "models" / "guardrails" / "bilstm" / "entrenamiento.json"}
    datos = json.loads(destino.read_text(encoding="utf-8")) if destino.exists() else {}
    for nombre, ruta in fuentes.items():
        if ruta.exists():
            e = json.loads(ruta.read_text(encoding="utf-8"))
            datos[nombre] = ({"perdida_train": e["perdida_train"], "perdida_val": e["perdida_val"], "epocas": e["epocas"],
                              "lr": e["lr"], "ejemplos_train": e["ejemplos_train"]} if nombre.startswith("qlora") else
                             {"historial": e["historial"], "parametros": e["parametros"], "lr": e["lr"]})
    if datos:
        destino.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    return datos


def figura_curvas():
    datos = curvas()
    if not {"qlora", "bilstm"} <= set(datos):
        print("   (faltan curvas: se omite fig4)")
        return
    q, b = datos["qlora"], datos["bilstm"]["historial"]
    fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 4.0))
    izq.plot([p["paso"] for p in q["perdida_train"]], [p["perdida"] for p in q["perdida_train"]], color=AZUL,
             linewidth=2, label="Train")
    izq.plot([p["paso"] for p in q["perdida_val"]], [p["perdida"] for p in q["perdida_val"]], color=NARANJA,
             linewidth=2, marker="o", markersize=7, label="Validación")
    ultimo = q["perdida_val"][-1]
    izq.annotate(f"{ultimo['perdida']:.3f}", (ultimo["paso"], ultimo["perdida"]), textcoords="offset points",
                 xytext=(6, 6), fontsize=8.5)
    izq.set_title(f"QLoRA del generador ({q['ejemplos_train']} ejemplos, {q['epocas']} épocas)")
    izq.set_xlabel("Paso de optimización")
    izq.set_ylabel("Pérdida (entropía cruzada, solo la respuesta)")
    izq.legend()
    rejilla(izq, "y")

    epocas = [h["epoca"] for h in b]
    der.plot(epocas, [h["f1_val"] for h in b], color=AZUL, linewidth=2, marker="o", markersize=6)
    mejor = max(b, key=lambda h: h["f1_val"])
    der.annotate(f"mejor: {mejor['f1_val']:.3f} (época {mejor['epoca']})", (mejor["epoca"], mejor["f1_val"]),
                 textcoords="offset points", xytext=(-70, 12), fontsize=8.5)
    der.set_ylim(top=1.0)  # aire para la anotación, sin chocar con el título
    der.set_title("BiLSTM propia: F1 macro en validación")
    der.set_xlabel("Época (parada temprana, paciencia 5)")
    der.set_ylabel("F1 macro")
    rejilla(der, "y")
    fig.suptitle("Curvas de aprendizaje", x=0.01, ha="left", fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    guardar(fig, "fig4_curvas.png", "evaluacion/resultados/curvas.json (copiado de models/*/entrenamiento.json)")


# --- 5. Velocidad ---------------------------------------------------------------------------------------------
def figura_velocidad():
    # Medido el 2026-11-02 con el examen real de 8 analitos (PLAN.md, Fase 5); i5-13420H y RTX 4050 Laptop 6 GB
    filas = [("CPU con gramática JSON (antes)", 159), ("GPU con gramática JSON", 175),
             ("CPU sin gramática", 85), ("GPU sin gramática (ahora)", 28)]
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    colores = [AZUL if "ahora" in e else GRIS for e, _ in filas]
    barras = ax.barh(range(len(filas)), [s for _, s in filas], height=0.62, color=colores)
    for barra, (_, s) in zip(barras, filas):
        ax.text(barra.get_width() + 2, barra.get_y() + barra.get_height() / 2, f"{s} s", va="center", fontsize=9)
    ax.set_yticks(range(len(filas)), [e for e, _ in filas])
    ax.invert_yaxis()
    ax.set_xlim(0, 200)
    rejilla(ax)
    ax.set_title("Generador propio: segundos por examen de 8 analitos")
    ax.set_xlabel("Segundos (menos es mejor)")
    guardar(fig, "fig5_velocidad.png", "PLAN.md, Fase 5 (medido el 2026-11-02)")


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    print("Figuras:")
    for f in (figura_rag, figura_guardrail, figura_generador, figura_curvas, figura_velocidad, figura_generador_v2):
        f()
