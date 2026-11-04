"""
¿El RAG le entrega al generador el fragmento que explica el resultado del paciente?

Para un analito ALTO o BAJO, el pasaje útil es el que dice qué significa ese resultado y a qué puede deberse
("Los niveles altos de AST pueden deberse a hepatitis, cirrosis..."). Antes, la búsqueda usaba solo el nombre del
examen y a una AST alta le daba "la prueba de AST se suele incluir en un análisis de rutina"; el generador inventó
la causa (PDF Colsanitas, 2026-11-04).

Etiquetas: para cada (analito, estado), los fragmentos de data/rag/documentos.jsonl que lo explican, leídos en sus
primeros 700 caracteres (lo que recibe el generador). Cada caso se consulta con un nombre corto y uno largo como los
de los laboratorios reales. Tras la primera corrida se agregaron como correctos 4 fragmentos que el criterio elegía
y que, releídos, sí explican el resultado (glucosa-01, a1c-01, albumina-00, tsh-00). Compara la búsqueda por nombre (antes) con la que usa el estado (modelos_propios.buscar).

    python evaluacion/evaluar_rag_estado.py   ->  evaluacion/resultados/fase5_rag_estado.json
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

P = "mlp-"
ETIQUETAS = [
    # (nombres de laboratorio, estado, fragmentos correctos)
    (["Hemoglobina", "HEMOGLOBINA EN SANGRE TOTAL"], "BAJO", ["analisis-de-hemoglobina-03"]),
    (["HGB", "Hemoglobina"], "ALTO", ["analisis-de-hemoglobina-04"]),
    (["Hematocrito", "HTO HEMATOCRITO"], "BAJO", ["hematocrito-03", "hematocrito-04"]),
    (["HCT", "Hematocrito"], "ALTO", ["hematocrito-05"]),
    (["Glóbulos rojos", "RECUENTO DE ERITROCITOS"], "BAJO", ["conteo-de-globulos-rojos-03"]),
    (["RBC", "Recuento de glóbulos rojos"], "ALTO", ["conteo-de-globulos-rojos-03"]),
    (["Leucocitos", "RECUENTO DE LEUCOCITOS"], "ALTO", ["conteo-de-globulos-blancos-04"]),
    (["WBC", "Glóbulos blancos"], "BAJO", ["conteo-de-globulos-blancos-04"]),
    (["Plaquetas", "RECUENTO DE PLAQUETAS"], "BAJO", ["pruebas-de-plaquetas-02"]),
    (["PLT", "Plaquetas"], "ALTO", ["pruebas-de-plaquetas-02"]),
    (["VCM", "Volumen corpuscular medio"], "BAJO", ["indices-de-globulos-rojos-02"]),
    (["MCV", "VOLUMEN CORPUSCULAR MEDIO"], "ALTO", ["indices-de-globulos-rojos-02"]),
    (["HCM", "Hemoglobina corpuscular media"], "ALTO", ["indices-de-globulos-rojos-02", "indices-de-globulos-rojos-03"]),
    (["Neutrófilos", "NEUTROFILOS %"], "ALTO", ["formula-leucocitaria-05"]),
    (["Linfocitos", "LINFOCITOS %"], "BAJO", ["formula-leucocitaria-05"]),
    (["Colesterol total", "COLESTEROL TOTAL"], "ALTO", ["niveles-de-colesterol-05", "niveles-de-colesterol-06"]),
    (["LDL", "COLESTEROL LDL - NO CALCULADO"], "ALTO", ["niveles-de-colesterol-05", "niveles-de-colesterol-06"]),
    (["HDL", "COLESTEROL DE ALTA DENSIDAD"], "BAJO", ["niveles-de-colesterol-05", "niveles-de-colesterol-06"]),
    (["Triglicéridos", "TRIGLICERIDOS"], "ALTO", ["prueba-de-trigliceridos-01", "prueba-de-trigliceridos-06"]),
    (["Glucosa", "GLUCOSA EN SUERO U OTRO FLUIDO DIFERENTE A ORINA"], "ALTO", ["prueba-de-glucosa-en-la-sangre-05"]),
    (["Glicemia basal", "GLUCOSA EN AYUNAS"], "BAJO", ["prueba-de-glucosa-en-la-sangre-06", "prueba-de-glucosa-en-la-sangre-01"]),
    (["HbA1c", "HEMOGLOBINA GLICOSILADA"], "ALTO", ["prueba-de-hemoglobina-a1c-06", "prueba-de-hemoglobina-a1c-01"]),
    (["Creatinina", "CREATININA EN SUERO"], "ALTO", ["prueba-de-creatinina-06"]),
    (["BUN", "NITROGENO UREICO EN SUERO"], "ALTO", ["examen-de-nitrogeno-ureico-en-sangre-nus-03"]),
    (["Urea", "NITROGENO UREICO"], "BAJO", ["examen-de-nitrogeno-ureico-en-sangre-nus-03"]),
    (["Ácido úrico", "ACIDO URICO EN SUERO"], "ALTO", ["prueba-de-acido-urico-04"]),
    (["Ácido úrico", "ACIDO URICO"], "BAJO", ["prueba-de-acido-urico-05"]),
    (["Calcio", "CALCIO TOTAL EN SUERO"], "ALTO", ["prueba-de-calcio-en-la-sangre-05"]),
    (["Ca", "Calcio"], "BAJO", ["prueba-de-calcio-en-la-sangre-05"]),
    (["Albúmina", "ALBUMINA EN SUERO"], "BAJO", ["prueba-de-albumina-en-la-sangre-03"]),
    (["ALB", "Albúmina"], "ALTO", ["prueba-de-albumina-en-la-sangre-04", "prueba-de-albumina-en-la-sangre-00"]),
    (["ALT", "ALANINO AMINO TRANSFERASA (ALAT - TGP)"], "ALTO", ["prueba-de-sangre-de-alt-03", "prueba-de-sangre-de-alt-04"]),
    (["TGP", "ALANINO AMINOTRANSFERASA"], "BAJO", ["prueba-de-sangre-de-alt-03"]),
    (["AST", "ASPARTATO AMINO TRANSFERASA (ASAT - TGO)"], "ALTO", ["prueba-de-ast-03"]),
    (["Fosfatasa alcalina", "FOSFATASA ALCALINA"], "ALTO", ["fosfatasa-alcalina-03", "fosfatasa-alcalina-04"]),
    (["Bilirrubina total", "BILIRRUBINA TOTAL"], "ALTO", ["examen-de-bilirrubina-en-sangre-03"]),
    (["GGT", "GAMMA GLUTAMIL TRANSFERASA"], "ALTO", ["prueba-de-gamma-glutamil-transferasa-ggt-04"]),
    (["TSH", "HORMONA ESTIMULANTE DEL TIROIDES ULTRASENSIBLE"], "ALTO", ["prueba-de-tsh-00", "prueba-de-tsh-01", "prueba-de-tsh-05", "prueba-de-tsh-06"]),
    (["TSH", "Tirotropina"], "BAJO", ["prueba-de-tsh-00", "prueba-de-tsh-01", "prueba-de-tsh-05", "prueba-de-tsh-06"]),
    (["T4 libre", "TIROXINA LIBRE"], "BAJO", ["prueba-de-tiroxina-t4-07"]),
    (["FT4", "TIROXINA LIBRE"], "ALTO", ["prueba-de-tiroxina-t4-08"]),
    (["Sodio", "SODIO EN SUERO"], "ALTO", ["prueba-de-sodio-en-la-sangre-03"]),
    (["Na", "SODIO"], "BAJO", ["prueba-de-sodio-en-la-sangre-03", "prueba-de-sodio-en-la-sangre-04"]),
    (["Potasio", "POTASIO EN SUERO"], "ALTO", ["prueba-de-potasio-en-sangre-03"]),
]
TIPO = "Química sanguínea"


def main():
    import modelos_propios as mp
    r = mp.RecuperadorRAG()
    filas, aciertos = [], {"antes": 0, "con_estado": 0}
    for nombres, estado, correctos in ETIQUETAS:
        correctos = {P + c for c in correctos}
        for nombre in nombres:
            consulta = f"{nombre} ({TIPO})"
            antes = r.buscar(consulta, examen=nombre, tipo_examen=TIPO)
            ahora = r.buscar(consulta, examen=nombre, tipo_examen=TIPO, estado=estado)
            fila = {"examen": nombre, "estado": estado, "correctos": sorted(correctos),
                    "antes": antes and antes["id"], "con_estado": ahora and ahora["id"]}
            for clave in aciertos:
                fila[f"acierta_{clave}"] = fila[clave] in correctos
                aciertos[clave] += fila[f"acierta_{clave}"]
            filas.append(fila)
    n = len(filas)
    resumen = {"consultas": n, "casos": len(ETIQUETAS),
               "acierto_antes": aciertos["antes"] / n, "acierto_con_estado": aciertos["con_estado"] / n,
               "empeora": [f for f in filas if f["acierta_antes"] and not f["acierta_con_estado"]],
               "sigue_fallando": [f for f in filas if not f["acierta_con_estado"]], "filas": filas}
    salida = RAIZ / "evaluacion" / "resultados" / "fase5_rag_estado.json"
    salida.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{n} consultas ({len(ETIQUETAS)} analito-estado)")
    print(f"  Fragmento correcto, solo por nombre: {aciertos['antes']:3d}  ({resumen['acierto_antes']:.1%})")
    print(f"  Fragmento correcto, con el estado:   {aciertos['con_estado']:3d}  ({resumen['acierto_con_estado']:.1%})")
    for f in resumen["sigue_fallando"]:
        print(f"  falla: {f['examen']} {f['estado']}: {f['con_estado']} (esperado {f['correctos']})")
    print(f"Resultados en {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
