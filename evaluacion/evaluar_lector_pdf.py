"""
Fase 5: mide el lector local de PDFs (lector_pdf.py) con PDFs digitales sintéticos de respuesta conocida.

Cada caso de test de la Fase 1 se imprime como PDF en tres diseños de laboratorio, y se compara lo que lee el
lector con los datos originales:
- tabla:          "Examen | Resultado | Unidades | Valores de referencia" en una fila por analito.
- bloques:        el nombre en una línea y debajo "Resultado: valor  unidad  rango" (como el PDF real de prueba).
- tabla_bandera:  como tabla, con una columna "*" o "H"/"L" para los valores fuera de rango.

    python evaluacion/evaluar_lector_pdf.py [--n 60]

Un analito cuenta como exacto si coinciden nombre, valor, unidad y rango. También se mide si se reconoce la
muestra (sangre u orina, por la sección) y el sexo y la edad del paciente.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import lector_pdf  # noqa: E402

DATOS = RAIZ / "data" / "sintetico" / "examenes.jsonl"
SALIDA = RAIZ / "evaluacion" / "resultados" / "fase5_lector_pdf.json"
FUENTES = RAIZ / "assets" / "fuentes"
DISENOS = ("tabla", "bloques", "tabla_bandera")
PANEL_ORINA = "Análisis de Orina Completo"


def pdf_de_caso(caso, diseno):
    """Bytes de un PDF digital con los analitos del caso, en el diseño pedido."""
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.add_font("DejaVu", "", str(FUENTES / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FUENTES / "DejaVuSans-Bold.ttf"))
    pdf.set_auto_page_break(True, margin=20)
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 13)
    pdf.cell(0, 8, "LABORATORIO CLÍNICO DE PRUEBA", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", size=9)
    sexo = {"M": "Masculino", "F": "Femenino"}[caso["paciente"]["sexo"]]
    pdf.cell(0, 5, "Paciente: PERSONA DE PRUEBA        Documento: 000000", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"Edad: {caso['paciente']['edad']} años        Sexo: {sexo}        Fecha: 2026-11-02",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("DejaVu", "B", 10)
    pdf.cell(0, 6, caso["tipo_de_examen"].upper(), align="C", new_x="LMARGIN", new_y="NEXT")
    # Columnas con el espacio de un informe real: la unidad más larga ("x10^3/µL") no toca la columna del rango
    columnas = (15, 95, 128, 155)
    if diseno != "bloques":
        pdf.set_font("DejaVu", "B", 9)
        y = pdf.get_y()
        for x, titulo in zip(columnas, ("Examen", "Resultado", "Unidades", "Valores de referencia")):
            pdf.set_xy(x, y)
            pdf.cell(30, 6, titulo)
        pdf.ln(7)
    pdf.set_font("DejaVu", size=9)
    for a in caso["analitos"]:
        y = pdf.get_y()
        if diseno == "bloques":
            pdf.set_x(15)
            pdf.cell(0, 5, a["examen"].upper(), new_x="LMARGIN", new_y="NEXT")
            y = pdf.get_y()
            celdas = [(15, "Resultado:"), (95, a["valor"]), (128, a["unidad"]), (155, a["rango_referencia"])]
        else:
            celdas = [(columnas[0], a["examen"]), (columnas[1], a["valor"]), (columnas[2], a["unidad"]),
                      (columnas[3], a["rango_referencia"])]
            if diseno == "tabla_bandera" and a["estado"] != "NORMAL":
                celdas.append((115, {"ALTO": "H", "BAJO": "L"}.get(a["estado"], "*")))
        for x, texto in celdas:
            if texto:
                pdf.set_xy(x, y)
                pdf.cell(40, 5, str(texto))
        pdf.set_y(y + 6)
        if diseno == "bloques":
            pdf.set_x(18)
            pdf.set_font("DejaVu", size=7)
            pdf.cell(0, 4, "METODO: ENZIMÁTICO", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("DejaVu", size=9)
    pdf.ln(6)
    pdf.set_font("DejaVu", size=7)
    pdf.cell(0, 4, "Profesional responsable: BACTERIÓLOGA DE PRUEBA", align="C", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def normalizar(t):
    return " ".join(str(t or "").split()).lower()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="Casos de test por diseño")
    args = parser.parse_args()
    with open(DATOS, encoding="utf-8") as f:
        casos = [c for c in map(json.loads, f) if c["split"] == "test"][:args.n]

    resultados, errores = {}, []
    for diseno in DISENOS:
        c = Counter()
        for caso in casos:
            leido = lector_pdf.leer_pdf(pdf_de_caso(caso, diseno))
            c["documentos"] += 1
            if leido is None:
                c["no_leidos"] += 1
                c["analitos"] += len(caso["analitos"])
                errores.append({"diseno": diseno, "caso": caso["id"], "error": "documento no leído"})
                continue
            c["paciente_ok"] += leido["paciente"] == {"sexo": caso["paciente"]["sexo"], "edad": caso["paciente"]["edad"]}
            por_nombre = {normalizar(a["examen"]): a for a in leido["analitos"]}
            c["analitos_de_mas"] += max(0, len(leido["analitos"]) - len(caso["analitos"]))
            esperada = "orina" if caso["tipo_de_examen"] == PANEL_ORINA else "sangre"
            for a in caso["analitos"]:
                c["analitos"] += 1
                l = por_nombre.get(normalizar(a["examen"]))
                if l is None:
                    errores.append({"diseno": diseno, "caso": caso["id"], "esperado": a["examen"],
                                    "leidos": [x["examen"] for x in leido["analitos"]]})
                    continue
                campos = {k: normalizar(l[k]) == normalizar(a[k]) for k in ("valor", "unidad", "rango_referencia")}
                for k, ok in campos.items():
                    c[k] += ok
                c["nombre"] += 1
                c["muestra"] += l["muestra"] == esperada
                c["exactos"] += all(campos.values())
                if not all(campos.values()):
                    errores.append({"diseno": diseno, "caso": caso["id"], "examen": a["examen"],
                                    "esperado": {k: a[k] for k in campos}, "leido": {k: l[k] for k in campos}})
        n = c["analitos"]
        resultados[diseno] = {
            "documentos": c["documentos"], "documentos_leidos": 1 - c["no_leidos"] / c["documentos"],
            "analitos": n, "exactos": c["exactos"] / n, "nombre": c["nombre"] / n, "valor": c["valor"] / n,
            "unidad": c["unidad"] / n, "rango": c["rango_referencia"] / n, "muestra": c["muestra"] / n,
            "paciente": c["paciente_ok"] / (c["documentos"] - c["no_leidos"] or 1), "analitos_de_mas": c["analitos_de_mas"],
        }
        r = resultados[diseno]
        print(f"{diseno:<14} documentos leídos {r['documentos_leidos']:.0%} | analitos exactos {r['exactos']:.1%} "
              f"(nombre {r['nombre']:.1%}, valor {r['valor']:.1%}, unidad {r['unidad']:.1%}, rango {r['rango']:.1%}) "
              f"| muestra {r['muestra']:.1%} | paciente {r['paciente']:.0%} | de más {r['analitos_de_mas']}")
    SALIDA.write_text(json.dumps({"casos_por_diseno": len(casos), "resultados": resultados, "errores": errores[:80]},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(errores)} errores; detalle en {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
