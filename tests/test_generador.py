"""
Pruebas de la Fase 1: generador de exámenes sintéticos.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests -v
"""
import re
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "training" / "sintetico"))

from backend import LetraDeMedicoBackend  # noqa: E402
from generar_examenes import cargar_catalogo, estado_verdadero, generar_dataset  # noqa: E402

CATALOGO = cargar_catalogo()


@pytest.fixture(scope="module")
def casos():
    return generar_dataset(1500, seed=7, catalogo=CATALOGO)


def _analitos(casos):
    return [(c, a) for c in casos for a in c["analitos"]]


def _a_numero(texto):
    return float(texto.replace(",", "."))


def test_es_reproducible_con_la_misma_semilla():
    assert generar_dataset(50, seed=1) == generar_dataset(50, seed=1)
    assert generar_dataset(50, seed=1) != generar_dataset(50, seed=2)


def test_paneles_referencian_analitos_del_catalogo():
    for panel, claves in CATALOGO["paneles"].items():
        faltantes = set(claves) - set(CATALOGO["analitos"])
        assert not faltantes, f"{panel} referencia analitos inexistentes: {faltantes}"


def test_estructura_compatible_con_el_backend(casos):
    for caso in casos:
        assert caso["split"] in {"train", "val", "test"}
        assert caso["tipo_de_examen"]
        assert len(caso["analitos"]) >= 2
        for a in caso["analitos"]:
            # Las cuatro claves que produce la extracción multimodal del backend
            for clave in ("examen", "valor", "unidad", "rango_referencia"):
                assert isinstance(a[clave], str)
            assert a["estado"] in {"NORMAL", "ALTO", "BAJO", "ANORMAL"}


def test_valores_dentro_de_limites_fisiologicos(casos):
    for caso, a in _analitos(casos):
        analito = CATALOGO["analitos"][a["examen_canonico"]]
        if analito["tipo"] == "cualitativo":
            continue
        pmin, pmax = analito["limites_fisiologicos"]
        assert pmin <= _a_numero(a["valor"]) <= pmax, a


def test_etiqueta_coincide_con_el_valor(casos):
    """Verificación independiente de que el estado guardado corresponde al número generado."""
    for caso, a in _analitos(casos):
        analito = CATALOGO["analitos"][a["examen_canonico"]]
        if analito["tipo"] == "cualitativo":
            assert (a["valor"] == "Negativo") == (a["estado"] == "NORMAL")
        else:
            assert estado_verdadero(_a_numero(a["valor"]), analito, caso["paciente"]["sexo"]) == a["estado"], a


def test_hay_variedad_de_estados_y_severidades(casos):
    estados = {a["estado"] for _, a in _analitos(casos)}
    severidades = {a["severidad"] for _, a in _analitos(casos)}
    assert estados == {"NORMAL", "ALTO", "BAJO", "ANORMAL"}
    assert {"normal", "leve", "marcada"} <= severidades


def test_proporciones_de_split(casos):
    n = len(casos)
    train = sum(c["split"] == "train" for c in casos) / n
    test = sum(c["split"] == "test" for c in casos) / n
    assert 0.65 < train < 0.75
    assert 0.11 < test < 0.19


def test_embarazo_solo_en_pacientes_compatibles(casos):
    for caso in casos:
        if "embarazada" in caso["contexto_paciente"]:
            assert caso["paciente"]["sexo"] == "F" and caso["paciente"]["edad"] <= 45


def test_regla_del_backend_acierta_en_formatos_canonicos(casos):
    """La regla determinista debe acertar el 100% de los formatos que dice soportar."""
    fallos = [
        (a["valor"], a["rango_referencia"], a["estado"])
        for _, a in _analitos(casos)
        if a["formato_canonico"]
        and LetraDeMedicoBackend.evaluar_estado_analito(a["valor"], a["rango_referencia"]) != a["estado"]
    ]
    assert not fallos, f"{len(fallos)} fallos, ej.: {fallos[:5]}"


def test_regla_del_backend_nunca_inventa_un_estado_en_formatos_variantes(casos):
    """
    En formatos que la regla no soporta ("70 a 99", "Trazas") se acepta DESCONOCIDO,
    pero nunca un estado equivocado: eso llegaría al paciente como un dato falso.
    """
    equivocados = []
    for _, a in _analitos(casos):
        if a["formato_canonico"]:
            continue
        prediccion = LetraDeMedicoBackend.evaluar_estado_analito(a["valor"], a["rango_referencia"])
        if prediccion not in (a["estado"], "DESCONOCIDO"):
            equivocados.append((a["valor"], a["rango_referencia"], a["estado"], prediccion))
    assert not equivocados, f"{len(equivocados)} estados equivocados, ej.: {equivocados[:5]}"


@pytest.mark.xfail(reason="Brecha conocida de la regla: no entiende 'a', 'Hasta', 'Menor de' ni cruces. Se corrige en la Fase 5.", strict=True)
def test_regla_del_backend_entiende_formatos_variantes(casos):
    for _, a in _analitos(casos):
        if not a["formato_canonico"]:
            assert LetraDeMedicoBackend.evaluar_estado_analito(a["valor"], a["rango_referencia"]) == a["estado"]


def test_comas_decimales_consistentes_en_cada_caso(casos):
    for caso in casos:
        numericos = [a for a in caso["analitos"] if re.fullmatch(r"\d+[.,]\d+", a["valor"])]
        separadores = {"," if "," in a["valor"] else "." for a in numericos}
        assert len(separadores) <= 1, caso["id"]
