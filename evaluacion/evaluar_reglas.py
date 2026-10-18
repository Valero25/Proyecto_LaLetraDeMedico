"""
Fase 1: Evalúa la regla determinista del backend (evaluar_estado_analito) contra las etiquetas
correctas de los exámenes sintéticos, separando por formato del rango/valor.

Uso:
    python evaluacion/evaluar_reglas.py [--datos data/sintetico/examenes.jsonl]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from backend import LetraDeMedicoBackend  # noqa: E402

DATOS_DEFAULT = RAIZ / "data" / "sintetico" / "examenes.jsonl"
SALIDA = RAIZ / "evaluacion" / "resultados" / "fase1_reglas.json"


def evaluar(casos):
    por_formato = defaultdict(Counter)
    errores = []
    for caso in casos:
        for a in caso["analitos"]:
            prediccion = LetraDeMedicoBackend.evaluar_estado_analito(a["valor"], a["rango_referencia"])
            if prediccion == a["estado"]:
                resultado = "correcto"
            elif prediccion == "DESCONOCIDO":
                resultado = "desconocido"
            else:
                resultado = "incorrecto"
                errores.append({**a, "prediccion": prediccion})
            por_formato[a["formato"]][resultado] += 1
            por_formato[a["formato"]]["canonico" if a["formato_canonico"] else "variante"] += 1
    return por_formato, errores


def main():
    # La consola de Windows usa cp1252 por defecto y daña las tildes
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--datos", type=Path, default=DATOS_DEFAULT)
    args = parser.parse_args()

    with open(args.datos, encoding="utf-8") as f:
        casos = [json.loads(linea) for linea in f]

    por_formato, errores = evaluar(casos)

    print(f"{'formato':<24}{'tipo':<11}{'total':>7}{'correcto':>10}{'descon.':>9}{'incorr.':>9}{'exactitud':>11}")
    filas = {}
    for formato in sorted(por_formato, key=lambda f: (-por_formato[f]["canonico"], f)):
        c = por_formato[formato]
        total = c["correcto"] + c["desconocido"] + c["incorrecto"]
        exactitud = c["correcto"] / total
        tipo = "canónico" if c["canonico"] else "variante"
        print(f"{formato:<24}{tipo:<11}{total:>7}{c['correcto']:>10}{c['desconocido']:>9}{c['incorrecto']:>9}{exactitud:>10.1%}")
        filas[formato] = {"tipo": tipo, "total": total, "correcto": c["correcto"],
                          "desconocido": c["desconocido"], "incorrecto": c["incorrecto"], "exactitud": exactitud}

    total = sum(f["total"] for f in filas.values())
    correctos = sum(f["correcto"] for f in filas.values())
    print(f"\nExactitud global: {correctos / total:.1%} ({correctos}/{total})")
    print(f"Etiquetas incorrectas (no 'DESCONOCIDO', sino un estado equivocado): {len(errores)}")
    for e in errores[:10]:
        print(f"  {e['examen']}: valor={e['valor']!r} rango={e['rango_referencia']!r} -> {e['prediccion']} (esperado {e['estado']})")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump({"datos": str(args.datos.relative_to(RAIZ)), "exactitud_global": correctos / total,
                   "por_formato": filas, "errores": errores[:50]}, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
