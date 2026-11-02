"""
Ficha resumen para llevar a la consulta médica, en PDF (para imprimir) o PNG (para compartir por el celular).

Las dos salidas se dibujan desde el mismo contenido (`contenido_ficha`), así que nunca dicen cosas distintas:
- tabla con todos los resultados, primero los que conviene comentar;
- la explicación de los valores fuera de rango o sin evaluar;
- las preguntas que eligió el paciente, con casillas para marcar;
- espacio para notas de la consulta y el aviso de que no es un diagnóstico.

Usa las fuentes DejaVu de assets/fuentes (licencia libre, ver LICENSE_DEJAVU.txt): cubren µ, ³, ≤, • y las
tildes, y hacen que el resultado sea igual en cualquier computador.
"""
import io
import re
from datetime import datetime
from pathlib import Path

FUENTES = Path(__file__).resolve().parent / "assets" / "fuentes"
FUENTE = FUENTES / "DejaVuSans.ttf"
FUENTE_NEGRITA = FUENTES / "DejaVuSans-Bold.ttf"

# Estado -> (etiqueta, color de texto, color de fondo), los mismos tonos de la app
ESTADOS = {
    "ALTO": ("ALTO", (153, 27, 27), (254, 226, 226)),
    "ELEVADO": ("ALTO", (153, 27, 27), (254, 226, 226)),
    "BAJO": ("BAJO", (146, 64, 14), (254, 243, 199)),
    "ANORMAL": ("FUERA DE RANGO", (91, 33, 182), (237, 233, 254)),
    "NORMAL": ("NORMAL", (6, 95, 70), (209, 250, 229)),
}
SIN_EVALUAR = ("SIN EVALUAR", (71, 85, 105), (241, 245, 249))
PRIMARIO = (13, 148, 136)
TEXTO = (30, 41, 59)
SUAVE = (100, 116, 139)
AVISO = ("Este resumen fue preparado con la asistencia de LetraDeMédico para facilitar la conversación con tu "
         "médico. Es informativo: NO es un diagnóstico ni reemplaza la valoración clínica.")
NOTA_GUARDRAIL_RE = re.compile(r"\s*\(Nota del sistema:.*?\)\.?\s*$", re.S)
# Emojis y pictogramas: DejaVu no los tiene y quedarían como cuadros
NO_IMPRIMIBLE = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")


def limpiar(texto):
    texto = NO_IMPRIMIBLE.sub("", str(texto or ""))
    # Sin el emoji puede quedar "rango (173 - 414) ." : se juntan espacios y se pega la puntuación
    texto = re.sub(r"[ \t]+", " ", texto)
    return re.sub(r" +([.,;:!?])", r"\1", texto).strip()


def etiqueta_estado(estado):
    return ESTADOS.get(str(estado or "").upper(), SIN_EVALUAR)


def es_para_comentar(estado):
    return str(estado or "").upper() != "NORMAL"


def contenido_ficha(tipo_examen, analitos, explicaciones, preguntas, contexto_paciente="", fecha=None):
    """
    Contenido común de las dos salidas. `explicaciones` va alineada con `analitos` (una por analito, como la
    deja backend.alinear_explicaciones); si falta o no coincide, la ficha sale sin explicaciones.
    """
    explicaciones = explicaciones if explicaciones and len(explicaciones) == len(analitos) else [{}] * len(analitos)
    filas = []
    for a, e in zip(analitos, explicaciones):
        texto = "" if e.get("sin_explicacion") else NOTA_GUARDRAIL_RE.sub("", e.get("explicacion", ""))
        filas.append({
            "examen": limpiar(a.get("examen")),
            "resultado": limpiar(f"{a.get('valor', '')} {a.get('unidad', '')}") or "No reportado",
            "referencia": limpiar(a.get("rango_referencia")) or "No indicada",
            "estado": str(a.get("estado") or "").upper(),
            "explicacion": limpiar(texto),
        })
    # Primero lo que conviene comentar; dentro de cada grupo se respeta el orden del documento
    filas.sort(key=lambda f: not es_para_comentar(f["estado"]))
    return {
        "titulo": "Guía para la consulta médica",
        "fecha": (fecha or datetime.now()).strftime("%d/%m/%Y"),
        "tipo_examen": limpiar(tipo_examen) or "Examen de laboratorio",
        "contexto": limpiar(contexto_paciente),
        "filas": filas,
        "comentar": [f for f in filas if es_para_comentar(f["estado"])],
        "preguntas": [limpiar(p) for p in preguntas if limpiar(p)],
    }


# =====================================================================================================
# PDF (fpdf2)
# =====================================================================================================

def ficha_pdf(tipo_examen, analitos, explicaciones, preguntas, contexto_paciente="", fecha=None):
    """Bytes de un PDF A4 con texto seleccionable."""
    from fpdf import FPDF
    from fpdf.fonts import FontFace

    c = contenido_ficha(tipo_examen, analitos, explicaciones, preguntas, contexto_paciente, fecha)

    class Ficha(FPDF):
        def footer(self):
            self.set_y(-12)
            self.set_font("DejaVu", size=7.5)
            self.set_text_color(*SUAVE)
            self.cell(0, 5, f"LetraDeMédico · No es un diagnóstico médico · Página {self.page_no()}/{{nb}}", align="C")

    pdf = Ficha(format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_font("DejaVu", "", str(FUENTE))
    pdf.add_font("DejaVu", "B", str(FUENTE_NEGRITA))
    pdf.set_title(f"{c['titulo']} - {c['tipo_examen']}")
    pdf.set_creator("LetraDeMédico")
    pdf.add_page()
    ancho = pdf.epw

    # Encabezado
    pdf.set_fill_color(*PRIMARIO)
    pdf.rect(0, 0, pdf.w, 30, style="F")
    pdf.set_xy(15, 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("DejaVu", "B", 18)
    pdf.cell(ancho, 9, c["titulo"])
    pdf.set_xy(15, 18)
    pdf.set_font("DejaVu", size=10)
    pdf.cell(ancho, 6, f"LetraDeMédico · {c['fecha']}")
    pdf.set_y(36)

    def subtitulo(texto):
        pdf.ln(3)
        pdf.set_font("DejaVu", "B", 12.5)
        pdf.set_text_color(*PRIMARIO)
        pdf.cell(ancho, 7, texto, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*PRIMARIO)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + ancho, pdf.get_y())
        pdf.ln(2.5)
        pdf.set_text_color(*TEXTO)

    def dato(etiqueta, valor):
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(*TEXTO)
        pdf.write(5.5, f"{etiqueta}: ")
        pdf.set_font("DejaVu", size=10)
        pdf.write(5.5, valor)
        pdf.ln(6.5)

    dato("Tipo de examen", c["tipo_examen"])
    if c["contexto"]:
        dato("Motivo o contexto", c["contexto"])
    n_comentar = len(c["comentar"])
    dato("Resumen", f"{len(c['filas'])} resultados; {n_comentar} para comentar con tu médico"
         if n_comentar else f"{len(c['filas'])} resultados, todos dentro del rango de referencia")

    # Tabla de resultados
    subtitulo("1. Tus resultados")
    pdf.set_font("DejaVu", size=9)
    # Las celdas heredan el color de relleno vigente (el verde del encabezado): se vuelve a blanco
    pdf.set_fill_color(255, 255, 255)
    encabezado = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=PRIMARIO)
    with pdf.table(col_widths=(44, 20, 22, 14), headings_style=encabezado, line_height=5.2,
                   text_align=("LEFT", "LEFT", "LEFT", "CENTER"), borders_layout="HORIZONTAL_LINES",
                   padding=1.6) as tabla:
        fila = tabla.row()
        for titulo in ("Examen", "Resultado", "Referencia", "Estado"):
            fila.cell(titulo)
        for f in c["filas"]:
            etiqueta, color, fondo = etiqueta_estado(f["estado"])
            fila = tabla.row()
            fila.cell(f["examen"])
            fila.cell(f["resultado"])
            fila.cell(f["referencia"])
            fila.cell(etiqueta, style=FontFace(emphasis="BOLD", color=color, fill_color=fondo, size_pt=8))

    # Explicaciones de lo que conviene comentar
    if c["comentar"]:
        subtitulo("2. Para comentar con tu médico")
        for f in c["comentar"]:
            etiqueta, color, _ = etiqueta_estado(f["estado"])
            pdf.set_font("DejaVu", "B", 10)
            pdf.set_text_color(*color)
            pdf.multi_cell(ancho, 5.5, f"{f['examen']} · {etiqueta} · {f['resultado']} (ref. {f['referencia']})",
                           new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("DejaVu", size=9.5)
            pdf.set_text_color(*TEXTO)
            texto = f["explicacion"] or ("No hay una explicación automática para este examen: pídele a tu médico "
                                         "que te explique qué mide y qué significa tu valor.")
            pdf.multi_cell(ancho, 5, texto, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2.5)

    # Preguntas con casillas
    subtitulo(f"{3 if c['comentar'] else 2}. Mis preguntas para el médico")
    pdf.set_font("DejaVu", size=10)
    if c["preguntas"]:
        for p in c["preguntas"]:
            y = pdf.get_y()
            pdf.set_draw_color(*SUAVE)
            pdf.rect(pdf.l_margin, y + 1, 3.6, 3.6)
            pdf.set_x(pdf.l_margin + 6)
            pdf.multi_cell(ancho - 6, 5.5, p, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
    else:
        pdf.set_text_color(*SUAVE)
        pdf.multi_cell(ancho, 5.5, "No seleccionaste preguntas. Puedes escribirlas en las notas.",
                       new_x="LMARGIN", new_y="NEXT")

    # Aviso (antes de las notas, para que nunca quede solo en otra página)
    pdf.ln(4)
    pdf.set_fill_color(255, 251, 235)
    pdf.set_draw_color(245, 158, 11)
    pdf.set_text_color(120, 53, 15)
    pdf.set_font("DejaVu", "B", 9)
    pdf.multi_cell(ancho, 5, "Aviso importante", border="LTR", fill=True, padding=(2, 3, 0, 3),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", size=9)
    pdf.multi_cell(ancho, 5, AVISO, border="LBR", fill=True, padding=(0, 3, 2, 3), new_x="LMARGIN", new_y="NEXT")

    # Notas de la consulta: las líneas que quepan sobre el pie de página (mínimo 3; si no caben, otra página)
    alto_linea, espacio_fijo = 8, 16
    libres = int((pdf.h - pdf.b_margin - pdf.get_y() - espacio_fijo) // alto_linea)
    if libres < 3:
        pdf.add_page()
        libres = 6
    subtitulo("Notas de la consulta")
    pdf.set_draw_color(203, 213, 225)
    for _ in range(min(libres, 6)):
        pdf.ln(alto_linea)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + ancho, pdf.get_y())
    return bytes(pdf.output())


# =====================================================================================================
# Imagen PNG (Pillow): una sola imagen vertical, cómoda para compartir por el celular
# =====================================================================================================

def ficha_png(tipo_examen, analitos, explicaciones, preguntas, contexto_paciente="", fecha=None, ancho=1080):
    """Bytes de un PNG vertical de `ancho` píxeles; el alto depende del contenido."""
    from PIL import Image, ImageDraw, ImageFont

    c = contenido_ficha(tipo_examen, analitos, explicaciones, preguntas, contexto_paciente, fecha)

    def fuente(tam, negrita=False):
        return ImageFont.truetype(str(FUENTE_NEGRITA if negrita else FUENTE), tam)

    f_titulo, f_sub, f_h2 = fuente(46, True), fuente(26), fuente(32, True)
    f_txt, f_txt_b, f_peq, f_peq_b = fuente(26), fuente(26, True), fuente(22), fuente(21, True)
    margen = 56
    util = ancho - 2 * margen
    medidor = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def envolver(texto, f, ancho_max):
        lineas = []
        for parrafo in str(texto).split("\n"):
            actual = ""
            for palabra in parrafo.split():
                prueba = f"{actual} {palabra}".strip()
                if medidor.textlength(prueba, font=f) <= ancho_max or not actual:
                    actual = prueba
                else:
                    lineas.append(actual)
                    actual = palabra
            lineas.append(actual)
        return lineas

    def alto_linea(f):
        return int(f.size * 1.42)

    # Se dibuja dos veces: la primera solo mide el alto total, la segunda dibuja en una imagen de ese alto
    def dibujar(d):
        y = 0
        if d:
            d.rectangle([0, 0, ancho, 190], fill=PRIMARIO)
            d.text((margen, 46), c["titulo"], font=f_titulo, fill="white")
            d.text((margen, 118), f"LetraDeMédico · {c['fecha']}", font=f_sub, fill=(204, 251, 241))
        y = 230

        def parrafo(texto, f, color=TEXTO, x=margen, ancho_max=util, espacio=0):
            nonlocal y
            for linea in envolver(texto, f, ancho_max):
                if d:
                    d.text((x, y), linea, font=f, fill=color)
                y += alto_linea(f)
            y += espacio

        def subtitulo(texto):
            nonlocal y
            y += 26
            if d:
                d.text((margen, y), texto, font=f_h2, fill=PRIMARIO)
            y += alto_linea(f_h2) + 4
            if d:
                d.line([margen, y, ancho - margen, y], fill=PRIMARIO, width=3)
            y += 18

        parrafo(f"Tipo de examen: {c['tipo_examen']}", f_txt_b, espacio=4)
        if c["contexto"]:
            parrafo(f"Motivo o contexto: {c['contexto']}", f_txt, color=SUAVE, espacio=4)
        n_comentar = len(c["comentar"])
        parrafo(f"{len(c['filas'])} resultados · {n_comentar} para comentar con tu médico" if n_comentar
                else f"{len(c['filas'])} resultados, todos dentro del rango", f_txt, color=SUAVE)

        subtitulo("1. Tus resultados")
        ancho_pastilla = 250
        for f in c["filas"]:
            etiqueta, color, fondo = etiqueta_estado(f["estado"])
            inicio = y
            parrafo(f["examen"], f_txt_b, ancho_max=util - ancho_pastilla - 20)
            parrafo(f"{f['resultado']}   ·   referencia: {f['referencia']}", f_peq, color=SUAVE,
                    ancho_max=util - ancho_pastilla - 20)
            if d:
                ancho_texto = d.textlength(etiqueta, font=f_peq_b)
                x1 = ancho - margen
                x0 = x1 - max(ancho_texto + 40, 150)
                d.rounded_rectangle([x0, inicio + 2, x1, inicio + 46], radius=22, fill=fondo)
                d.text(((x0 + x1 - ancho_texto) / 2, inicio + 11), etiqueta, font=f_peq_b, fill=color)
            y = max(y, inicio + 52) + 14
            if d:
                d.line([margen, y - 7, ancho - margen, y - 7], fill=(226, 232, 240), width=2)

        if c["comentar"]:
            subtitulo("2. Para comentar con tu médico")
            for f in c["comentar"]:
                etiqueta, color, _ = etiqueta_estado(f["estado"])
                parrafo(f"{f['examen']} · {etiqueta}", f_txt_b, color=color)
                parrafo(f["explicacion"] or "Sin explicación automática: pídele a tu médico que te la explique.",
                        f_peq, espacio=16)

        subtitulo(f"{3 if c['comentar'] else 2}. Mis preguntas para el médico")
        for p in c["preguntas"] or ["No seleccionaste preguntas."]:
            if d and c["preguntas"]:
                d.rectangle([margen, y + 6, margen + 24, y + 30], outline=SUAVE, width=3)
            parrafo(p, f_txt, x=margen + 42, ancho_max=util - 42, espacio=10)

        y += 20
        alto_aviso = (len(envolver(AVISO, f_peq, util - 48)) + 1) * alto_linea(f_peq) + 40
        if d:
            d.rounded_rectangle([margen, y, ancho - margen, y + alto_aviso], radius=16, fill=(255, 251, 235),
                                outline=(245, 158, 11), width=3)
        y += 20
        parrafo("Aviso importante", f_peq_b, color=(120, 53, 15), x=margen + 24, ancho_max=util - 48)
        parrafo(AVISO, f_peq, color=(120, 53, 15), x=margen + 24, ancho_max=util - 48)
        return y + 60

    alto = dibujar(None)
    imagen = Image.new("RGB", (ancho, alto), "white")
    dibujar(ImageDraw.Draw(imagen))
    salida = io.BytesIO()
    imagen.save(salida, format="PNG", optimize=True)
    return salida.getvalue()
