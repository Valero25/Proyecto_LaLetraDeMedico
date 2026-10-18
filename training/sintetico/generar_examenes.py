"""
Fase 1: Generador de exámenes de laboratorio sintéticos.

Produce casos con la misma estructura que devuelve la extracción multimodal del backend
({"tipo_de_examen", "analitos": [{examen, valor, unidad, rango_referencia}]}), pero con la
etiqueta correcta de cada analito ya calculada ("estado" y "severidad"). Esos casos alimentan
el dataset del clasificador de guardrails (Fase 3) y el dataset SFT del generador (Fase 4).

Los rangos salen de catalogo_analitos.json. El generador mezcla formatos que la regla
determinista del backend ya entiende (formato_canonico=True) con variantes reales que
todavía no entiende (ej. "70 a 99", "Hasta 200", "Trazas"), para poder medir esa brecha.

Uso:
    python training/sintetico/generar_examenes.py --n 2000 --seed 42
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
CATALOGO_PATH = Path(__file__).with_name("catalogo_analitos.json")
SALIDA_DEFAULT = RAIZ / "data" / "sintetico" / "examenes.jsonl"

PROB_NORMAL = 0.6
PROB_MARCADA = 0.4          # Entre los anormales, proporción de desviaciones marcadas (el resto son leves)
MARGEN_LEVE = 0.2           # Una desviación leve queda a menos del 20% del ancho del rango (o del umbral)
PROB_COMA_DECIMAL = 0.3     # Algunos laboratorios escriben "4,5" en vez de "4.5"
PROB_FORMATO_VARIANTE = 0.15
PROB_QUITAR_ANALITO = 0.1
PANEL_MIXTO = "Chequeo General"

# (nombre del formato, plantilla, ¿la regla del backend lo entiende?)
FORMATOS_RANGO = {
    "rango": [
        ("rango_guion_espacios", "{lo} - {hi}", True),
        ("rango_guion", "{lo}-{hi}", True),
        ("rango_raya", "{lo} – {hi}", True),
        ("rango_a", "{lo} a {hi}", False),
    ],
    "max": [
        ("max_simbolo", "< {u}", True),
        ("max_simbolo_junto", "<{u}", True),
        ("max_texto", "Menor de {u}", False),
        ("max_hasta", "Hasta {u}", False),
    ],
    "min": [
        ("min_simbolo", "> {u}", True),
        ("min_simbolo_junto", ">{u}", True),
        ("min_texto", "Mayor de {u}", False),
    ],
}

# (valor, severidad, ¿la regla del backend lo entiende?)
CUALITATIVOS_ANORMALES = [
    ("Positivo", "indeterminada", True),
    ("Presente", "indeterminada", True),
    ("Trazas", "leve", False),
    ("1+", "leve", False),
    ("2+", "marcada", False),
    ("3+", "marcada", False),
]

# (texto, restricción): el contexto opcional que el paciente escribe en la app
CONTEXTOS_PACIENTE = [
    ("Me hice el examen de control anual.", None),
    ("Estaba en ayunas de 12 horas.", None),
    ("No estaba en ayunas cuando me tomaron la muestra.", None),
    ("Tengo diabetes tipo 2 y tomo metformina.", None),
    ("Tengo hipertensión y tomo losartán.", None),
    ("Últimamente me siento con mucho cansancio.", None),
    ("Estoy embarazada de 20 semanas.", "embarazo"),
    ("Hago ejercicio intenso varias veces por semana.", None),
    ("Tomo suplementos de hierro.", None),
    ("Mi médico me lo pidió porque en mi familia hay antecedentes de colesterol alto.", None),
]
PROB_SIN_CONTEXTO = 0.35


def cargar_catalogo(path=CATALOGO_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _por_sexo(valor, sexo):
    """Los rangos pueden ser únicos o depender del sexo ({"M": ..., "F": ...})."""
    return valor[sexo] if isinstance(valor, dict) else valor


def formatear_numero(x, decimales, coma=False):
    texto = f"{x:.{decimales}f}"
    return texto.replace(".", ",") if coma else texto


def estado_verdadero(x, analito, sexo):
    """
    Etiqueta correcta de un valor numérico. Usa la misma convención que la regla del backend:
    rango inclusivo en ambos extremos, "< u" es ALTO desde u y "> u" es BAJO hasta u.
    """
    if analito["tipo"] == "rango":
        lo, hi = _por_sexo(analito["rango"], sexo)
        if x < lo:
            return "BAJO"
        if x > hi:
            return "ALTO"
        return "NORMAL"
    u = _por_sexo(analito["umbral"], sexo)
    if analito["tipo"] == "max":
        return "ALTO" if x >= u else "NORMAL"
    return "BAJO" if x <= u else "NORMAL"


def estados_posibles(analito, sexo):
    """Un analito con rango 0 - 5 no puede salir BAJO; con umbral máximo solo puede salir ALTO."""
    if analito["tipo"] == "max":
        return ["ALTO"]
    if analito["tipo"] == "min":
        return ["BAJO"]
    lo, hi = _por_sexo(analito["rango"], sexo)
    pmin, pmax = analito["limites_fisiologicos"]
    paso = 10 ** -analito["decimales"]
    posibles = []
    if lo - paso >= pmin:
        posibles.append("BAJO")
    if hi + paso <= pmax:
        posibles.append("ALTO")
    return posibles


def _intervalo_muestreo(analito, sexo, estado, severidad):
    """Devuelve el intervalo [a, b] de donde sacar el valor para el estado y severidad pedidos."""
    pmin, pmax = analito["limites_fisiologicos"]
    paso = 10 ** -analito["decimales"]

    if analito["tipo"] == "rango":
        lo, hi = _por_sexo(analito["rango"], sexo)
        margen = MARGEN_LEVE * (hi - lo)
        if estado == "NORMAL":
            return lo, hi
        if estado == "ALTO":
            leve = (hi + paso, min(pmax, hi + margen))
            marcada = (hi + margen, pmax)
        else:
            leve = (max(pmin, lo - margen), lo - paso)
            marcada = (pmin, lo - margen)
    else:
        u = _por_sexo(analito["umbral"], sexo)
        margen = MARGEN_LEVE * u
        if analito["tipo"] == "max":
            if estado == "NORMAL":
                return pmin, u - paso
            leve = (u, min(pmax, u + margen))
            marcada = (u + margen, pmax)
        else:
            if estado == "NORMAL":
                return u + paso, min(pmax, 2 * u)
            leve = (max(pmin, u - margen), u)
            marcada = (pmin, u - margen)

    # Si el rango está pegado al límite fisiológico no hay espacio para una desviación marcada
    if severidad == "marcada" and marcada[0] < marcada[1]:
        return marcada
    return leve


def _ajustar_al_estado(x, analito, sexo, estado):
    """El redondeo puede empujar el valor al borde del rango; lo corregimos un paso."""
    paso = 10 ** -analito["decimales"]
    for _ in range(3):
        if estado_verdadero(x, analito, sexo) == estado:
            return x
        if estado == "ALTO":
            x = round(x + paso, analito["decimales"])
        elif estado == "BAJO":
            x = round(x - paso, analito["decimales"])
        else:
            break
    if estado_verdadero(x, analito, sexo) != estado:
        raise ValueError(f"No se pudo generar un valor {estado} para {analito['nombres'][0]}")
    return x


def generar_analito(rng, clave, analito, sexo, coma):
    nombre = rng.choice(analito["nombres"])

    if analito["tipo"] == "cualitativo":
        if rng.random() < PROB_NORMAL:
            valor, estado, severidad, canonico, formato = "Negativo", "NORMAL", "normal", True, "cualitativo"
        else:
            valor, severidad, canonico = rng.choice(CUALITATIVOS_ANORMALES)[:3]
            estado = "ANORMAL"
            formato = "cualitativo" if canonico else "cualitativo_cruces"
        return {
            "examen": nombre,
            "examen_canonico": clave,
            "valor": valor,
            "unidad": analito["unidad"],
            "rango_referencia": "Negativo",
            "estado": estado,
            "severidad": severidad,
            "formato": formato,
            "formato_canonico": canonico,
        }

    posibles = estados_posibles(analito, sexo)
    if not posibles or rng.random() < PROB_NORMAL:
        estado, severidad = "NORMAL", "normal"
    else:
        estado = rng.choice(posibles)
        severidad = "marcada" if rng.random() < PROB_MARCADA else "leve"

    a, b = _intervalo_muestreo(analito, sexo, estado, severidad)
    decimales = analito["decimales"]
    x = round(rng.uniform(a, b), decimales)
    x = _ajustar_al_estado(x, analito, sexo, estado)

    canonicos = [f for f in FORMATOS_RANGO[analito["tipo"]] if f[2]]
    variantes = [f for f in FORMATOS_RANGO[analito["tipo"]] if not f[2]]
    if variantes and rng.random() < PROB_FORMATO_VARIANTE:
        formato, plantilla, canonico = rng.choice(variantes)
    else:
        formato, plantilla, canonico = rng.choice(canonicos)

    fmt = lambda v: formatear_numero(v, decimales, coma)
    if analito["tipo"] == "rango":
        lo, hi = _por_sexo(analito["rango"], sexo)
        rango_texto = plantilla.format(lo=fmt(lo), hi=fmt(hi))
    else:
        rango_texto = plantilla.format(u=fmt(_por_sexo(analito["umbral"], sexo)))

    return {
        "examen": nombre,
        "examen_canonico": clave,
        "valor": fmt(x),
        "unidad": analito["unidad"],
        "rango_referencia": rango_texto,
        "estado": estado,
        "severidad": severidad,
        "formato": formato,
        "formato_canonico": canonico,
    }


def _elegir_contexto(rng, sexo, edad):
    if rng.random() < PROB_SIN_CONTEXTO:
        return ""
    opciones = [
        texto for texto, restriccion in CONTEXTOS_PACIENTE
        if restriccion != "embarazo" or (sexo == "F" and edad <= 45)
    ]
    return rng.choice(opciones)


def _elegir_claves(rng, catalogo, panel):
    paneles = catalogo["paneles"]
    if panel == PANEL_MIXTO:
        # Un chequeo general mezcla unos pocos analitos de varios paneles de sangre
        candidatos = [p for p in paneles if p != "Análisis de Orina Completo"]
        claves = []
        for p in rng.sample(candidatos, k=rng.randint(2, 3)):
            claves += rng.sample(paneles[p], k=min(len(paneles[p]), rng.randint(2, 4)))
        return list(dict.fromkeys(claves))

    claves = [c for c in paneles[panel] if rng.random() >= PROB_QUITAR_ANALITO]
    return claves if len(claves) >= 2 else list(paneles[panel])


def generar_caso(rng, catalogo, indice, seed):
    sexo = rng.choice(["M", "F"])
    edad = rng.randint(18, 85)
    panel = rng.choice(list(catalogo["paneles"]) + [PANEL_MIXTO])
    coma = rng.random() < PROB_COMA_DECIMAL

    analitos = [
        generar_analito(rng, clave, catalogo["analitos"][clave], sexo, coma)
        for clave in _elegir_claves(rng, catalogo, panel)
    ]

    sorteo_split = rng.random()
    split = "train" if sorteo_split < 0.7 else "val" if sorteo_split < 0.85 else "test"

    return {
        "id": f"sint-{seed}-{indice:05d}",
        "split": split,
        "tipo_de_examen": panel,
        "paciente": {"sexo": sexo, "edad": edad},
        "contexto_paciente": _elegir_contexto(rng, sexo, edad),
        "analitos": analitos,
    }


def generar_dataset(n, seed=42, catalogo=None):
    catalogo = catalogo or cargar_catalogo()
    rng = random.Random(seed)
    return [generar_caso(rng, catalogo, i, seed) for i in range(n)]


def resumen(casos):
    analitos = [a for c in casos for a in c["analitos"]]
    return {
        "casos": len(casos),
        "analitos": len(analitos),
        "split": dict(Counter(c["split"] for c in casos)),
        "paneles": dict(Counter(c["tipo_de_examen"] for c in casos)),
        "estados": dict(Counter(a["estado"] for a in analitos)),
        "severidad": dict(Counter(a["severidad"] for a in analitos)),
        "formato_canonico": dict(Counter(a["formato_canonico"] for a in analitos)),
    }


def main():
    # La consola de Windows usa cp1252 por defecto y daña las tildes
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Genera exámenes de laboratorio sintéticos etiquetados.")
    parser.add_argument("--n", type=int, default=2000, help="Número de casos a generar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--salida", type=Path, default=SALIDA_DEFAULT)
    args = parser.parse_args()

    casos = generar_dataset(args.n, args.seed)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        for caso in casos:
            f.write(json.dumps(caso, ensure_ascii=False) + "\n")

    print(f"Generados {len(casos)} casos en {args.salida}")
    print(json.dumps(resumen(casos), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
