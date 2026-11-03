"""
Rúbrica humana (evaluacion/rubrica.py): la plantilla es ciega y balanceada, y el resumen destapa bien la clave.
Usa una carpeta temporal: no toca la plantilla ni las respuestas reales del equipo.
"""
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "evaluacion"))

openpyxl = pytest.importorskip("openpyxl")
pytest.importorskip("scipy")
import rubrica  # noqa: E402


@pytest.fixture
def carpeta(tmp_path, monkeypatch):
    monkeypatch.setattr(rubrica, "CARPETA", tmp_path)
    monkeypatch.setattr(rubrica, "PLANTILLA", tmp_path / "plantilla.xlsx")
    monkeypatch.setattr(rubrica, "CLAVE", tmp_path / "clave.json")
    monkeypatch.setattr(rubrica, "RESPUESTAS", tmp_path / "respuestas")
    monkeypatch.setattr(rubrica, "RESULTADO", tmp_path / "resultado.json")
    rubrica.preparar(Namespace(n=10, seed=3))
    return tmp_path


def test_plantilla_ciega_y_solo_sangre(carpeta):
    clave = json.loads((carpeta / "clave.json").read_text(encoding="utf-8"))["casos"]
    assert len(clave) == 10
    assert {v["A"] for v in clave.values()} == set(rubrica.MODELOS)  # el orden cambia entre casos
    casos = {c["id"]: c for c in map(json.loads, rubrica.CASOS.open(encoding="utf-8"))}
    assert all(rubrica.es_de_sangre(casos[v["id"]]) for v in clave.values())
    hoja = openpyxl.load_workbook(carpeta / "plantilla.xlsx")["Casos"]
    assert hoja.max_row - 1 == 20  # A y B por caso
    textos = " ".join(str(c.value) for fila in hoja.iter_rows() for c in fila if c.value)
    assert "ajustado" not in textos and "base" not in textos.lower().split()  # nada delata al modelo


def llenar(carpeta, nombre, puntaje_por_modelo, preferido):
    clave = json.loads((carpeta / "clave.json").read_text(encoding="utf-8"))["casos"]
    (carpeta / "respuestas").mkdir(exist_ok=True)
    destino = carpeta / "respuestas" / f"rubrica_{nombre}.xlsx"
    shutil.copy(carpeta / "plantilla.xlsx", destino)
    libro = openpyxl.load_workbook(destino)
    hoja = libro["Casos"]
    n_crit = len(rubrica.CRITERIOS)
    for fila in range(2, hoja.max_row + 1):
        codigo, letra = hoja.cell(fila, 1).value, hoja.cell(fila, 2).value
        modelo = clave[codigo][letra]
        for j in range(6, 6 + n_crit):
            hoja.cell(fila, j).value = puntaje_por_modelo[modelo]
        if letra == "A":
            hoja.cell(fila, 6 + n_crit).value = "A" if clave[codigo]["A"] == preferido else "B"
            hoja.cell(fila, 7 + n_crit).value = "Inventa una sigla"
    libro.save(destino)


def test_resumen_destapa_la_clave(carpeta):
    base, ajustado = rubrica.MODELOS
    llenar(carpeta, "ana", {base: 2, ajustado: 4}, ajustado)
    llenar(carpeta, "luis", {base: 3, ajustado: 5}, ajustado)
    rubrica.resumir(Namespace())
    r = json.loads((carpeta / "resultado.json").read_text(encoding="utf-8"))
    assert r["evaluadores"] == ["ana", "luis"]
    exactitud = r["criterios"]["exactitud"]
    assert exactitud[base]["media"] == 2.5 and exactitud[ajustado]["media"] == 4.5 and exactitud["n_casos"] == 10
    assert exactitud["wilcoxon_p"] < 0.01
    assert r["preferencias"] == {ajustado: 20}
    assert r["acuerdo_preferencia"]["ana vs luis"]["acuerdo"] == 1.0
    # Los comentarios van en la fila A: se atribuyen al modelo que era A en cada caso, uno por evaluador
    clave = json.loads((carpeta / "clave.json").read_text(encoding="utf-8"))["casos"]
    base_como_a = sum(v["A"] == base for v in clave.values())
    assert len(r["comentarios"].get(base, [])) == 2 * base_como_a
    assert len(r["comentarios"].get(ajustado, [])) == 2 * (10 - base_como_a)
    # Si los dos eligen siempre lo mismo, kappa no está definido: se guarda como null, no como NaN
    assert r["acuerdo_preferencia"]["ana vs luis"]["kappa"] is None
