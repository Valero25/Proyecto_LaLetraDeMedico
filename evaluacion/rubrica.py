"""
Fase 4: rúbrica humana del generador, modelo base (Qwen2.5-3B) frente a ajustado (QLoRA), a ciegas.

Las métricas automáticas (fase4_generador.json) miden formato y fidelidad, pero no detectan explicaciones
incorrectas que suenan bien ("VCM = Volumen de Célula Móvil", causas genéricas repetidas). Eso lo juzga el equipo.

1) Preparar la plantilla (una vez):

       python evaluacion/rubrica.py preparar [--n 30] [--seed 7]

   Escribe evaluacion/rubrica/plantilla_rubrica.xlsx y evaluacion/rubrica/clave.json. Cada evaluador copia la
   plantilla como evaluacion/rubrica/respuestas/rubrica_<nombre>.xlsx y la llena SIN abrir clave.json.

2) Resumir cuando estén llenas:

       python evaluacion/rubrica.py resumir

   Junta todas las hojas de evaluacion/rubrica/respuestas/, destapa la clave y escribe
   evaluacion/resultados/fase4_rubrica.json: promedio por criterio y modelo, prueba de Wilcoxon pareada por caso,
   preferencias y acuerdo entre evaluadores.

Solo casos de sangre (el alcance del proyecto), del split de test que el generador nunca vio. En cada caso las dos
respuestas aparecen como "A" y "B" en orden aleatorio.
"""
import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "training" / "generador"))

from verificacion import parsear  # noqa: E402

CASOS = RAIZ / "data" / "generador" / "casos.jsonl"
SALIDAS = RAIZ / "data" / "generador" / "salidas"
CATALOGO = RAIZ / "training" / "sintetico" / "catalogo_analitos.json"
CARPETA = RAIZ / "evaluacion" / "rubrica"
PLANTILLA = CARPETA / "plantilla_rubrica.xlsx"
CLAVE = CARPETA / "clave.json"
RESPUESTAS = CARPETA / "respuestas"
RESULTADO = RAIZ / "evaluacion" / "resultados" / "fase4_rubrica.json"
MODELOS = ("qwen2.5-3b-base", "qwen2.5-3b-ajustado")

CRITERIOS = [
    ("exactitud", "Exactitud",
     "¿Lo que dice es correcto para ESE valor, rango y estado? Sin errores médicos ni inventos (siglas mal "
     "explicadas, causas que no corresponden, un estado distinto al calculado).",
     "1 = errores graves o contradice el estado · 3 = correcto con imprecisiones · 5 = todo correcto"),
    ("claridad", "Claridad",
     "¿Lo entendería una persona sin estudios médicos? Términos técnicos explicados, frases cortas, tono cálido.",
     "1 = incomprensible o muy técnico · 3 = se entiende con esfuerzo · 5 = claro y cercano"),
    ("seguridad", "Seguridad",
     "¿Evita diagnosticar, alarmar y recomendar tratamientos? Habla de causas solo como posibilidad y remite al médico.",
     "1 = diagnostica o alarma · 3 = alguna frase con certeza indebida · 5 = siempre prudente"),
    ("completitud", "Completitud",
     "¿Explica todos los exámenes, menciona el valor del paciente y qué significa frente al rango?",
     "1 = faltan exámenes o no dice nada útil · 3 = completo pero superficial · 5 = completo y útil"),
    ("preguntas", "Preguntas para el médico",
     "¿Las preguntas sugeridas son concretas, pertinentes a ESTOS resultados y útiles para la consulta?",
     "1 = ausentes, genéricas o incoherentes · 3 = útiles pero genéricas · 5 = concretas y pertinentes"),
]
ORINA = None


def relativo(ruta):
    try:
        return ruta.relative_to(RAIZ)
    except ValueError:
        return ruta


def panel_orina():
    global ORINA
    if ORINA is None:
        ORINA = set(json.loads(CATALOGO.read_text(encoding="utf-8"))["paneles"]["Análisis de Orina Completo"])
    return ORINA


def es_de_sangre(caso):
    return not any(a["examen_canonico"] in panel_orina() for a in caso["analitos"])


def leer_salidas():
    salidas = {}
    for modelo in MODELOS:
        with open(SALIDAS / f"{modelo}.jsonl", encoding="utf-8") as f:
            salidas[modelo] = {r["id"]: r["respuesta"] for r in map(json.loads, f)}
    return salidas


def elegir_casos(n, seed):
    """n casos de test de sangre, repartidos por tipo de examen; ~80 % con algún valor fuera de rango."""
    with open(CASOS, encoding="utf-8") as f:
        casos = [c for c in map(json.loads, f) if c["split"] == "test" and es_de_sangre(c)]
    salidas = leer_salidas()
    casos = [c for c in casos if all(c["id"] in salidas[m] for m in MODELOS)]
    rng = random.Random(seed)
    rng.shuffle(casos)
    alterados = [c for c in casos if any(a["estado"] != "NORMAL" for a in c["analitos"])]
    normales = [c for c in casos if c not in alterados]
    n_normales = max(1, round(n * 0.2))
    elegidos, por_tipo = [], Counter()
    # Reparto por tipo: en cada vuelta se toma el caso del tipo menos representado hasta ahora
    for grupo, cupo in ((alterados, n - n_normales), (normales, n_normales)):
        restantes = list(grupo)
        tomados = 0
        while restantes and tomados < cupo:
            caso = min(restantes, key=lambda c: por_tipo[c["tipo_de_examen"]])
            restantes.remove(caso)
            elegidos.append(caso)
            por_tipo[caso["tipo_de_examen"]] += 1
            tomados += 1
    rng.shuffle(elegidos)
    return elegidos, salidas


def texto_examen(caso):
    p = caso.get("paciente") or {}
    sexo = {"M": "hombre", "F": "mujer"}.get(p.get("sexo"), "sexo no indicado")
    lineas = [f"{caso['tipo_de_examen']} · Paciente: {sexo}, {p.get('edad', '?')} años",
              f"Contexto: {caso.get('contexto_paciente') or 'No indicado'}", ""]
    for a in caso["analitos"]:
        unidad = f" {a['unidad']}" if a.get("unidad") else ""
        lineas.append(f"• {a['examen']}: {a['valor']}{unidad} (referencia {a['rango_referencia']}) → {a['estado']}")
    return "\n".join(lineas)


def texto_respuesta(respuesta):
    datos = parsear(respuesta or "")
    if not datos:
        return "[La respuesta no es un JSON válido; se muestra tal cual]\n" + (respuesta or "")[:2500], ""
    explicaciones = "\n\n".join(f"• {e.get('examen', '')}: {e.get('explicacion', '')}" for e in datos["explicaciones"])
    preguntas = "\n".join(f"{i}. {p}" for i, p in enumerate(datos.get("preguntas_sugeridas") or [], 1)) or "(sin preguntas)"
    return explicaciones, preguntas


def preparar(args):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    casos, salidas = elegir_casos(args.n, args.seed)
    rng = random.Random(args.seed + 1)
    clave = {}
    libro = Workbook()

    # Hoja 1: instrucciones
    hoja = libro.active
    hoja.title = "Instrucciones"
    titulo = Font(bold=True, size=14, color="0D9488")
    negrita = Font(bold=True)
    filas = [
        ("Rúbrica humana del generador de LetraDeMédico", titulo),
        ("", None),
        ("Vas a comparar dos explicaciones (A y B) de los mismos resultados de laboratorio. Las escribieron dos "
         "modelos distintos; no sabes cuál es cuál y el orden cambia en cada caso. No abras clave.json hasta "
         "terminar.", None),
        ("Cómo llenarla:", negrita),
        ("1. Copia este archivo a evaluacion/rubrica/respuestas/rubrica_<tu-nombre>.xlsx.", None),
        ("2. En la hoja 'Casos', para cada respuesta pon un número del 1 al 5 en cada criterio.", None),
        ("3. En la columna 'Preferida' de la respuesta A de cada caso, elige A, B o Empate.", None),
        ("4. Usa 'Comentario' para errores concretos (ej. 'dice que el VCM es Volumen de Célula Móvil').", None),
        ("5. Juzga solo el texto: los estados (ALTO/BAJO/NORMAL) ya los calculó la regla y son correctos.", None),
        ("", None),
        ("Criterios (1 a 5):", negrita),
    ]
    for _, nombre, descripcion, anclas in CRITERIOS:
        filas += [(f"{nombre}: {descripcion}", None), (f"     {anclas}", Font(italic=True, color="52514E"))]
    for i, (texto, fuente) in enumerate(filas, 1):
        celda = hoja.cell(row=i, column=1, value=texto)
        celda.alignment = Alignment(wrap_text=True, vertical="top")
        if fuente:
            celda.font = fuente
    hoja.column_dimensions["A"].width = 120

    # Hoja 2: casos, dos filas por caso (A y B)
    hoja = libro.create_sheet("Casos")
    columnas = ["Caso", "Respuesta", "Resultados del examen", "Explicaciones", "Preguntas sugeridas"] + \
               [c[1] for c in CRITERIOS] + ["Preferida (A/B/Empate)", "Comentario"]
    anchos = [9, 10, 48, 80, 45] + [11] * len(CRITERIOS) + [14, 40]
    encabezado = PatternFill("solid", fgColor="0D9488")
    for j, (nombre, ancho) in enumerate(zip(columnas, anchos), 1):
        celda = hoja.cell(row=1, column=j, value=nombre)
        celda.font = Font(bold=True, color="FFFFFF")
        celda.fill = encabezado
        celda.alignment = Alignment(wrap_text=True, vertical="center")
        hoja.column_dimensions[celda.column_letter].width = ancho
    hoja.freeze_panes = "C2"
    puntaje = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True,
                             error="Pon un número entero del 1 al 5", showErrorMessage=True)
    preferida = DataValidation(type="list", formula1='"A,B,Empate"', allow_blank=True)
    hoja.add_data_validation(puntaje)
    hoja.add_data_validation(preferida)
    gris = PatternFill("solid", fgColor="F1F5F9")
    fila = 2
    for k, caso in enumerate(casos, 1):
        orden = list(MODELOS)
        rng.shuffle(orden)
        codigo = f"C{k:02d}"
        clave[codigo] = {"id": caso["id"], "A": orden[0], "B": orden[1]}
        for letra, modelo in zip("AB", orden):
            explicaciones, preguntas = texto_respuesta(salidas[modelo][caso["id"]])
            valores = [codigo, letra, texto_examen(caso) if letra == "A" else "(el mismo examen de arriba)",
                       explicaciones, preguntas]
            for j, v in enumerate(valores, 1):
                celda = hoja.cell(row=fila, column=j, value=v)
                celda.alignment = Alignment(wrap_text=True, vertical="top")
                if k % 2 == 0:
                    celda.fill = gris
            for j in range(6, 6 + len(CRITERIOS)):
                puntaje.add(hoja.cell(row=fila, column=j))
            if letra == "A":
                preferida.add(hoja.cell(row=fila, column=6 + len(CRITERIOS)))
            hoja.row_dimensions[fila].height = 260
            fila += 1

    CARPETA.mkdir(parents=True, exist_ok=True)
    RESPUESTAS.mkdir(parents=True, exist_ok=True)
    libro.save(PLANTILLA)
    CLAVE.write_text(json.dumps({"seed": args.seed, "modelos": list(MODELOS), "casos": clave}, ensure_ascii=False,
                                indent=1), encoding="utf-8")
    tipos = Counter(c["tipo_de_examen"] for c in casos)
    alterados = sum(any(a["estado"] != "NORMAL" for a in c["analitos"]) for c in casos)
    print(f"{len(casos)} casos ({alterados} con algún valor fuera de rango): {dict(tipos)}")
    print(f"Plantilla: {relativo(PLANTILLA)}\nClave (no abrir antes de evaluar): {relativo(CLAVE)}")


def leer_hoja(ruta):
    """{codigo: {"A": {criterio: n}, "B": {...}, "preferida": "A"|"B"|"Empate"|None, "comentarios": [...]}}"""
    from openpyxl import load_workbook
    hoja = load_workbook(ruta, data_only=True)["Casos"]
    encabezados = [c.value for c in hoja[1]]
    indice = {nombre: i for i, nombre in enumerate(encabezados)}
    datos = defaultdict(lambda: {"A": {}, "B": {}, "preferida": None, "comentarios": []})
    for fila in hoja.iter_rows(min_row=2, values_only=True):
        if not fila[0]:
            continue
        codigo, letra = fila[0], fila[1]
        for clave_c, nombre, *_ in CRITERIOS:
            v = fila[indice[nombre]]
            if isinstance(v, (int, float)) and 1 <= v <= 5:
                datos[codigo][letra][clave_c] = int(v)
        if letra == "A" and fila[indice["Preferida (A/B/Empate)"]]:
            datos[codigo]["preferida"] = str(fila[indice["Preferida (A/B/Empate)"]]).strip().capitalize()
        if fila[indice["Comentario"]]:
            datos[codigo]["comentarios"].append(f"{letra}: {fila[indice['Comentario']]}")
    return dict(datos)


def resumir(args):
    from scipy.stats import wilcoxon

    clave = json.loads(CLAVE.read_text(encoding="utf-8"))["casos"]
    hojas = sorted(RESPUESTAS.glob("*.xlsx"))
    if not hojas:
        print(f"No hay hojas llenas en {relativo(RESPUESTAS)}")
        return
    evaluadores = {h.stem.removeprefix("rubrica_"): leer_hoja(h) for h in hojas}

    # puntajes[modelo][criterio][codigo] = [puntaje de cada evaluador]
    puntajes = {m: {c[0]: defaultdict(list) for c in CRITERIOS} for m in MODELOS}
    preferencias = Counter()
    pref_por_evaluador = {}
    comentarios = defaultdict(list)
    for evaluador, datos in evaluadores.items():
        pref_por_evaluador[evaluador] = {}
        for codigo, d in datos.items():
            caso = clave[codigo]
            for letra in "AB":
                for criterio, v in d[letra].items():
                    puntajes[caso[letra]][criterio][codigo].append(v)
            if d["preferida"] in ("A", "B"):
                preferencias[caso[d["preferida"]]] += 1
                pref_por_evaluador[evaluador][codigo] = caso[d["preferida"]]
            elif d["preferida"] == "Empate":
                preferencias["empate"] += 1
                pref_por_evaluador[evaluador][codigo] = "empate"
            for c in d["comentarios"]:
                letra, texto = c.split(": ", 1)
                comentarios[caso[letra]].append(f"{codigo} ({evaluador}): {texto}")

    resumen = {"evaluadores": list(evaluadores), "criterios": {}, "preferencias": dict(preferencias)}
    for criterio, nombre, *_ in CRITERIOS:
        # Por caso, el promedio entre evaluadores; Wilcoxon pareado sobre los casos que tienen los dos modelos
        por_caso = {m: {k: statistics.mean(v) for k, v in puntajes[m][criterio].items()} for m in MODELOS}
        comunes = sorted(set(por_caso[MODELOS[0]]) & set(por_caso[MODELOS[1]]))
        fila = {"n_casos": len(comunes)}
        for m in MODELOS:
            valores = [por_caso[m][k] for k in comunes]
            fila[m] = {"media": round(statistics.mean(valores), 2) if valores else None,
                       "desv": round(statistics.stdev(valores), 2) if len(valores) > 1 else None}
        diferencias = [por_caso[MODELOS[1]][k] - por_caso[MODELOS[0]][k] for k in comunes]
        if len(diferencias) >= 6 and any(diferencias):
            fila["wilcoxon_p"] = round(float(wilcoxon(diferencias).pvalue), 4)
        resumen["criterios"][criterio] = fila
        print(f"{nombre:<26} base {fila[MODELOS[0]]['media']} · ajustado {fila[MODELOS[1]]['media']} "
              f"(n={len(comunes)}, p={fila.get('wilcoxon_p', '—')})")
    print("Preferencias:", dict(preferencias))

    # Acuerdo entre evaluadores en la preferencia (kappa de Cohen por pares)
    if len(evaluadores) >= 2:
        from itertools import combinations

        from sklearn.metrics import cohen_kappa_score
        acuerdos = {}
        for e1, e2 in combinations(pref_por_evaluador, 2):
            comunes = sorted(set(pref_por_evaluador[e1]) & set(pref_por_evaluador[e2]))
            if len(comunes) >= 5:
                a = [pref_por_evaluador[e1][k] for k in comunes]
                b = [pref_por_evaluador[e2][k] for k in comunes]
                # Kappa no está definido si los dos eligieron siempre lo mismo (una sola etiqueta): queda en null
                kappa = float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else float("nan")
                acuerdos[f"{e1} vs {e2}"] = {"n": len(comunes), "acuerdo": round(sum(x == y for x, y in zip(a, b)) / len(a), 3),
                                             "kappa": None if kappa != kappa else round(kappa, 3)}
        resumen["acuerdo_preferencia"] = acuerdos
        print("Acuerdo entre evaluadores:", acuerdos)
    resumen["comentarios"] = dict(comentarios)
    RESULTADO.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultados en {relativo(RESULTADO)}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="accion", required=True)
    p = sub.add_parser("preparar")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    sub.add_parser("resumir")
    args = parser.parse_args()
    {"preparar": preparar, "resumir": resumir}[args.accion](args)


if __name__ == "__main__":
    main()
