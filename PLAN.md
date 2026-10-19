# Plan: LetraDeMédico con modelos propios

**Objetivo:** pasar de "orquestar prompts de Gemini" a un pipeline con **componentes entrenados y evaluados por el equipo**, atendiendo el feedback del hito 1. Gemini queda como **baseline de comparación** y para la extracción del documento (OCR + NER), que queda fuera del alcance de entrenamiento salvo que sobre tiempo.

| Componente | Modelo | Entrenamiento propio |
|---|---|---|
| Explicación en lenguaje claro | LLM pequeño (ej. Qwen2.5-3B-Instruct) con QLoRA | Fine-tuning generativo (SFT) |
| Guardrails | `BSC-LT/MrBERT-biomed` (alternativa: `BSC-LT/MrBERT-es`) | Fine-tuning de clasificación |
| RAG | `SINAI/ALIA-MrBERT-es-biomedical-embeddings` + FAISS | Evaluación de recuperación y ajuste contrastivo opcional |
| Estado ALTO/BAJO/NORMAL | Reglas deterministas (`evaluar_estado_analito`) | No aplica: se mide y se corrige |
| ~~Predicción de riesgo~~ | ~~Med-BERT~~ | Descartado: usa códigos ICD, no texto, y choca con "no diagnosticar" |

---

## Estado actual

**Fase en curso:** Fase 3. Los scripts están listos y el dataset de frases se está generando con Groq. La Fase 2 terminó con ALIA y e5 ajustados y evaluados; el modelo para la app es `alia-ajustado-completo` + alias.

| Fase | Estado |
|---|---|
| 0. Preparación | ✅ Terminada |
| 1. Generador de exámenes sintéticos | ✅ Terminada |
| 2. RAG con embeddings | ✅ Terminada |
| 3. Clasificador de guardrails | 🟡 En curso |
| 4. Fine-tuning QLoRA del generador | ⬜ Pendiente |
| 5. Integración en la app | ⬜ Pendiente |
| 6. Informe | ⬜ Pendiente |

Leyenda: ✅ terminada · 🟡 en curso · ⬜ pendiente

---

## Fase 0: preparación ✅

- [x] Estructura de carpetas:
  ```
  data/sintetico/        # datasets generados (no se versionan; se regeneran con la semilla)
  training/sintetico/    # generador + catálogo de analitos
  training/rag/          # Fase 2
  training/guardrails/   # Fase 3
  training/generador/    # Fase 4
  evaluacion/            # scripts de métricas
  evaluacion/resultados/ # resultados en JSON (sí se versionan, son evidencia)
  models/                # pesos entrenados (no se versionan)
  tests/                 # pruebas con pytest
  ```
- [x] `requirements-train.txt` separado de `requirements.txt` (la app no carga dependencias de entrenamiento).
- [x] `.gitignore`: excluye `models/` y los `.jsonl` generados.
- [x] Commit de los cambios previos en `app.py` y `backend.py` (reintentos y modelo de respaldo de Gemini), hecho por el equipo (`fb14d22`).

## Fase 1: generador de exámenes sintéticos ✅

- [x] `training/sintetico/catalogo_analitos.json`: 41 analitos en 8 paneles, con alias reales (ej. TGO/AST, Hb), unidades, rangos por sexo y límites fisiológicos.
- [x] `training/sintetico/generar_examenes.py`: genera casos con la misma estructura que la extracción del backend, más etiquetas correctas (`estado`, `severidad`), paciente (sexo, edad), contexto opcional y split 70/15/15.
- [x] Mezcla de formatos: canónicos (que la regla entiende) y variantes reales (`"70 a 99"`, `"Hasta 200"`, `"Menor de 200"`, `"Mayor de 40"`, `"Trazas"`, `"1+"`), además de coma decimal.
- [x] `evaluacion/evaluar_reglas.py`: mide la regla determinista del backend contra las etiquetas correctas.
- [x] `tests/test_generador.py`: 12 pruebas.
- [ ] **Pendiente del equipo:** revisar los rangos del catálogo contra una fuente médica y citarla en el informe.

### Resultados (dataset `--n 2000 --seed 42`)

- 2.000 casos y 10.604 analitos. Split: 1.403 train, 300 val, 297 test.
- Estados: 6.312 NORMAL, 2.110 ALTO, 1.611 BAJO, 571 ANORMAL.
- **Regla del backend:**
  - **100 %** de exactitud en formatos canónicos (8.847 de 8.847).
  - **0 %** en formatos variantes: responde `DESCONOCIDO` en los 1.757 casos.
  - **Nunca asigna un estado equivocado.**
  - Exactitud global: **83,4 %**.
  - Detalle en `evaluacion/resultados/fase1_reglas.json`.
- **Hallazgo para el informe y para la Fase 5:** la regla no entiende rangos como `"7 a 56"`, `"Hasta 200"`, `"Menor de 200"`, `"Mayor de 40"` ni cruces (`"Trazas"`, `"1+"`). El paciente recibe "desconocido" en datos que sí tienen interpretación.

### Limitaciones conocidas
- Los valores de un mismo caso son independientes: no hay correlación clínica (ej. hemoglobina baja con hematocrito bajo). Mejora posible para la Fase 4 si las explicaciones lo necesitan.
- Solo adultos (18 a 85 años) y rangos generales, no pediátricos ni de embarazo.

---

## Fase 2: RAG con embeddings ✅

- [x] **Base de conocimiento** (`training/rag/construir_base_conocimiento.py` → `data/rag/documentos.jsonl`, versionado):
  - 31 páginas de pruebas de laboratorio de **MedlinePlus en español** (NLM, dominio público), mapeadas en `training/rag/fuentes.json`.
  - Solo se guardan las secciones útiles: qué es, para qué se usa, por qué se pide, qué significan los resultados y "algo más".
  - 3 fichas redactadas como borrador (`training/rag/fichas_equipo.json`) para densidad urinaria, pH urinario y leucocitos en orina. La NLM solo publica el uroanálisis general en la enciclopedia A.D.A.M., que tiene licencia.
  - Total: **208 fragmentos, cubren los 41 analitos del catálogo**.
- [x] **Recuperadores** (`training/rag/recuperador.py`):
  - Denso (sentence-transformers + FAISS), BM25 y una réplica de la búsqueda actual del backend.
  - Opción de enriquecer cada fragmento con los alias del catálogo.
  - El índice se guarda en `models/rag/` (no versionado; se regenera con `python training/rag/recuperador.py`).
- [x] **Evaluación** (`evaluacion/evaluar_rag.py`): 505 consultas.
  - 465 generadas del catálogo: alias solo, alias + estado, pregunta, alias + panel.
  - 40 difíciles escritas a mano (`training/rag/consultas_manuales.json`): coloquiales, sin tildes, con errores.
- [x] `tests/test_rag.py`: 7 pruebas que no requieren descargar modelos.
- [x] **Decisión:** se hizo el ajuste contrastivo; modelo final `alia-ajustado-completo` + alias (ver abajo).
- [ ] **Pendiente del equipo:** revisar las 3 fichas propias, citar su fuente y marcar `"revisada": true`.

### Resultados (acierto en el primer resultado, hit@1)

| Recuperador | Total (505) | Alias + panel (así consulta la app) | Manuales (40, la prueba más justa) |
|---|---|---|---|
| Subcadena (sistema actual) | 6,3 % | 6,7 % | 5,0 % |
| BM25 | 62,2 % | 55,2 % | 75,0 % |
| e5-small (genérico) | 70,3 % | 68,6 % | 85,0 % |
| ALIA-MrBERT | 70,1 % | 72,4 % | 72,5 % |
| BM25 + alias | 83,4 % | 68,6 % | 82,5 % |
| **e5-small + alias** | **85,3 %** | 85,7 % | **87,5 %** |
| ALIA-MrBERT + alias | 82,6 % | **87,6 %** | 72,5 % |

Detalle con hit@3, hit@5, MRR y cada fallo en `evaluacion/resultados/fase2_rag.json`.

**Hallazgos:**
1. **El sistema actual casi nunca recupera nada útil** (6 %). Su base tiene solo 5 entradas y la búsqueda por subcadena falla incluso en lo que cubre: "Hemoglobina" devuelve la ficha de HbA1c.
2. **El cuello de botella eran las abreviaturas de laboratorio** (MCH, NEU%, TG, GLU, CREA, TGP, K, BT): MedlinePlus no las usa. Enriquecer los fragmentos con los alias del catálogo sube todos los modelos de unos 70 % a unos 85 %. Ojo: con alias, las consultas de tipo alias son fáciles por construcción; por eso la columna de consultas manuales es la referencia justa.
3. **ALIA-MrBERT no supera a e5-small**, un modelo genérico más pequeño (118M frente a 150M de parámetros). Es mejor en consultas con panel, pero peor en las coloquiales (ej. "TGO alta" → página de hemoglobina). El set manual es pequeño (1 consulta = 2,5 %), así que la diferencia ahí es indicativa, no concluyente.
4. Algunos textos de MedlinePlus usan lenguaje diagnóstico (ej. nitritos: "eso significa que usted tiene una infección urinaria"). El clasificador de la Fase 3 debe revisar también lo que el LLM copia del RAG.

### Ajuste contrastivo del modelo de embeddings (aprobado, en curso)
- [x] **Consultas de entrenamiento** (`training/rag/generar_consultas_entrenamiento.py` → `data/rag/consultas_entrenamiento.jsonl`, versionado):
  - **1.198 consultas** generadas por Qwen (Groq), unas 6 por fragmento. Cubren los 208 fragmentos, con entre 10 y 57 consultas por analito.
  - Coloquiales, con abreviaturas y con errores de tipeo (ej. "tengo el LDL por las nubes", "hcto que significa en mi reporte").
  - Se descartaron 46 repetidas o idénticas a alguna consulta de evaluación.
- [x] **Split por analitos** (`training/rag/split_analitos.json`): 29 analitos de entrenamiento y 12 de test, que el modelo nunca ve.
  - Los analitos que comparten página van juntos (ej. HDL, LDL y colesterol total), para que ningún fragmento de test se filtre.
  - La variante `split` entrena con 814 consultas. La evaluación reporta los grupos `vistos` y `no_vistos`.
- [x] **Entrenamiento** (`training/rag/ajustar_embeddings.py`):
  - Tripletas (consulta, fragmento positivo, negativo difícil) con `MultipleNegativesRankingLoss`.
  - Los negativos difíciles salen de las confusiones medidas: sangre/orina, HbA1c/hemoglobina, ALT/AST…
  - Variantes `split` (experimento honesto) y `completo` (modelo para la app), para ALIA y para e5-small.
- [x] Corrida de prueba en CPU: funciona, pero tarda unos 50 s por paso. Los 4 entrenamientos tomarían unas 6 horas, así que se corren en Colab.
- [x] Correr `training/rag/ajuste_embeddings_colab.ipynb` en Colab con GPU T4, descargar `fase2_ajuste.zip` y descomprimirlo en la raíz del proyecto.
- [x] Entrenamiento en Colab (T4, 3 épocas, batch 32, max_seq 384): **e5-small** `split` (814 ejemplos, 2 min, pérdida 3,24 → 1,34) y `completo` (1.198 ejemplos, 3 min, pérdida 3,28 → 1,09).
- [x] **ALIA falló al principio en Colab por falta de memoria** (`CUDA out of memory` en el primer paso): ModernBERT con 32 tripletas de 384 tokens supera los 15 GB de la T4.
  - Corrección: `--mini-batch 8` usa `CachedMultipleNegativesRankingLoss` (GradCache), que calcula la misma pérdida del batch de 32 por trozos. Así ALIA y e5 se comparan con los mismos hiperparámetros.
  - La evaluación ahora omite los modelos que falten en vez de caerse.
- [x] Reentrenamiento de **ALIA** en Colab (T4, mismos hiperparámetros + `--mini-batch 8`): `split` (814 ejemplos, 10,4 min, pérdida 1,80 → 1,03) y `completo` (1.198 ejemplos, 15,9 min, pérdida 1,77 → 0,80).
- [x] Analizar resultados (abajo). **Modelo elegido para la app: `alia-ajustado-completo` + alias** (alternativa casi empatada: `e5-small-ajustado-completo`).

#### Resultados del ajuste (hit@1)

| Recuperador | Total (505) | Manuales (40) | Analitos vistos (340) | **Analitos no vistos (160)** |
|---|---|---|---|---|
| Subcadena (sistema actual) | 6,3 % | 5,0 % | 9,4 % | 0,0 % |
| BM25 + alias | 83,4 % | 82,5 % | 81,5 % | 88,8 % |
| ALIA-MrBERT + alias (base) | 82,6 % | 72,5 % | 82,4 % | 84,4 % |
| e5-small + alias (base) | 85,3 % | 87,5 % | 83,8 % | 88,8 % |
| **e5-small ajustado (split) + alias** | **93,7 %** | 85,0 % | 92,6 % | **96,2 %** |
| **ALIA ajustado (split) + alias** | 92,1 % | 82,5 % | **92,9 %** | 91,2 % |
| e5-small ajustado (completo) + alias | 96,4 % | 85,0 % | 95,9 % | *98,1 % (no es justo: los vio)* |
| **ALIA ajustado (completo) + alias** | 95,4 % | **92,5 %** | 93,8 % | *99,4 % (no es justo: los vio)* |

Mismas filas con **hit@3** (el generador recibirá los 3 primeros fragmentos):

| Recuperador | Total | Manuales | No vistos | MRR@10 total |
|---|---|---|---|---|
| e5-small ajustado (split) | 96,8 % | 90,0 % | **99,4 %** | 0,955 |
| ALIA ajustado (split) | **97,2 %** | **92,5 %** | 98,1 % | 0,948 |
| e5-small ajustado (completo) | 98,0 % | 90,0 % | — | **0,974** |
| ALIA ajustado (completo) | **98,6 %** | **95,0 %** | — | 0,970 |

**Lectura:**
1. **El ajuste generaliza, con los dos modelos.** En los 12 analitos que el modelo nunca vio, e5 pasa de 88,8 % a **96,2 %** y ALIA de 84,4 % a **91,2 %**; ambos llegan a 98-99 % en hit@3. No es memorización: aprendieron a relacionar abreviaturas y lenguaje coloquial con la página correcta.
2. **ALIA es el que más gana con el ajuste.** Sube 9,5 puntos en total (82,6 → 92,1 %), frente a 8,4 de e5. En las consultas manuales ALIA pasa de 72,5 % a 82,5 % (split) y a **92,5 %** (completo), mientras que e5 baja de 87,5 % a 85,0 %. El modelo biomédico en español no destacaba de base, pero aprovecha mejor el ajuste.
3. **Fallos manuales del modelo final (ALIA completo, 3 de 40):** "defensas bajas en el hemograma" (fuera del top 10), "tamaño de los glóbulos rojos" (5.º) y "función del riñón" (2.º). e5 completo falla 6, entre ellas "sal en la sangre baja", que ni aparece en su top 10, y "azúcar en la sangre", que pone HbA1c antes que glucosa. "Defensas bajas" sigue siendo una limitación para el informe; se mitigaría con consultas de entrenamiento más coloquiales.
4. **Elección: ALIA ajustado (completo).** En hit@1 total, e5 está 1 punto arriba, pero la ventaja sale de las consultas de alias, que son fáciles por construcción. ALIA gana en hit@3, que es lo que recibe el generador, y en las consultas manuales, que son las más parecidas a lenguaje real (37 frente a 34 de 40; el set es pequeño, así que la diferencia es indicativa). Su costo: en CPU indexa y busca unas 3 veces más lento que e5 (unos 4 min frente a 1,3 min para las 505 consultas) y pesa 600 MB frente a 470 MB. En la app la base se indexa una sola vez, así que es aceptable.
5. Los analitos no vistos resultan algo más fáciles que los vistos incluso para los modelos base. Por eso la comparación se hace siempre dentro del mismo grupo.
6. El modelo `completo` es el que usará la app. Sus números sobre analitos vistos son optimistas; las cifras honestas para el informe son las del `split`.

## Fase 3: clasificador de guardrails con MrBERT-biomed 🟡

- [ ] **Dataset** (`training/guardrails/generar_frases.py` → `data/guardrails/frases.jsonl`): Qwen escribe 8 frases por petición, para cada (analito, clase) y en 2 rondas. En total son 246 peticiones y unas 1.970 frases.
  - Cada petición fija un resultado sintético de la Fase 1 (valor, rango, estado y contexto del paciente). El estado se sortea igual para las tres clases, para que el modelo no aprenda el atajo "valor alto = frase peligrosa".
  - Se piden casos difíciles para el filtro actual: informativas que usan "tienes" o "diagnóstico", diagnósticas sin esas palabras y alarmistas sutiles, sin signos de exclamación.
  - **Generadas: 1.774 frases** (219 de 246 peticiones). El límite real de Groq es de **1.000 tokens de salida por minuto** por modelo, unas 2 peticiones por minuto. Además, el proceso se cortó por falta de RAM en el equipo.
  - Faltan 27 peticiones de la ronda 2 de los 9 analitos de orina. Todos tienen su ronda 1. Se completan con el mismo comando (es reanudable) y luego se reentrena.
- [x] **Verificación de etiquetas** (`training/guardrails/verificar_etiquetas.py`): `openai/gpt-oss-120b` vuelve a etiquetar cada frase sin ver la clase pedida.
  - Sobre las 1.774 frases: acuerdo 88,7 %, **kappa de Cohen 0,83**.
  - Los desacuerdos se concentran en las diagnósticas: el juez ve 89 como informativas y 90 como alarmistas. Las informativas y alarmistas casi no generan dudas. Muchas de las diagnósticas que el juez ve como informativas descartan una enfermedad con certeza sobre un valor normal ("no tienes ninguna infección"). Por definición son diagnósticas.
  - Se entrena solo con las 1.574 frases en que coinciden profesor y juez. Los 200 desacuerdos van a `data/guardrails/revisar_equipo.csv`.
- [ ] **Pendiente del equipo:** llenar la columna `clase_equipo` de `revisar_equipo.csv` (`informativa`, `diagnostica`, `alarmista` o `descartar`).
- [x] **Split por analito** (`training/guardrails/dataset.py`): las frases de los 12 analitos de test de la Fase 2 van a test, para medir la generalización a exámenes no vistos. Del resto, 15 % va a validación.
- [x] **Entrenamiento** (`training/guardrails/ajustar_clasificador.py`):
  - MrBERT-biomed con cabeza de clasificación de 3 clases. Se elige la mejor época por F1 macro en validación.
  - Corre en CPU en unos 32 min y necesita unos 6 GB de RAM libres, así que no hace falta Colab.
- [x] **Evaluación** (`evaluacion/evaluar_guardrails.py`): tarea binaria (¿frase riesgosa?) frente al filtro de palabras actual, más F1 por clase y matriz de confusión.
- [x] **MrBERT-biomed entrenado** (899 frases, 4 épocas, lr 3e-5, batch 16, 32 min en CPU). Validación: F1 macro 0,993.
- [x] **MrBERT-es** (general, sin adaptación biomédica) como comparación: mismos hiperparámetros, 29 min en CPU, F1 de 0,987 en validación.
- [x] **Red neuronal propia** (`training/guardrails/red_propia.py`), diseñada y entrenada desde cero en PyTorch, sin preentrenamiento:
  - Arquitectura: `Embedding(128) → BiLSTM(2×128, empaquetada para ignorar el relleno) → pooling media + máximo → Linear(512→128) → ReLU → Dropout(0,3) → Linear(128→3)`.
  - Tamaño: 456.707 parámetros y un vocabulario de 988 palabras sacado del train.
  - Entrenamiento: AdamW con lr 2e-3, parada temprana por F1 en validación (paciencia 5), 1 min en CPU. Se corrió con 3 semillas para medir la variación.
  - Un test (`test_red_propia_ignora_el_relleno`) detectó que, sin empaquetar, la dirección hacia atrás de la LSTM leía primero el relleno. Por eso la salida de una frase dependía de las demás frases del lote. Se corrigió antes de reportar resultados.
- [x] **Baseline clásico**: TF-IDF de 1 y 2 palabras + regresión logística, entrenado dentro de `evaluar_guardrails.py`.

#### Resultados (test: 506 frases de 12 analitos no vistos)

| Modelo | Tipo | Parámetros | F1 macro (3 clases) | F1 diagnóstica | F1 binaria (¿riesgosa?) | FP / FN binaria |
|---|---|---|---|---|---|---|
| Filtro de palabras (actual) | Reglas | 0 | — | — | 0,219 | 60 / 276 |
| TF-IDF + regresión logística | Clásico | 8 mil | 0,929 | 0,890 | 0,977 | 4 / 11 |
| BiLSTM propia (3 semillas) | Red desde cero | 457 mil | 0,902 ± 0,009 | 0,866* | 0,963 ± 0,004 | 18 / 8* |
| MrBERT-es ajustado | Transformer preentrenado (general) | 308 M | 0,967 | 0,944 | **0,992** | 2 / 3 |
| **MrBERT-biomed ajustado** | Transformer preentrenado (biomédico) | 308 M | **0,973** | **0,961** | 0,991 | **1** / 5 |

\* De la semilla 13. Binaria = ¿la frase es diagnóstica o alarmista? Hay 323 frases riesgosas en test.

**Lectura:**
1. **El filtro actual no sirve como guardrail.** Deja pasar el 85 % de las frases riesgosas (276 de 323). Además, marca 60 frases informativas por usar "tienes" ("tienes el valor dentro del rango"). Frases como "Esto es una anemia" o "Deberías ir a urgencias de inmediato" no tienen ninguna de sus 4 palabras.
2. **El preentrenamiento es lo que más aporta.** Con solo 899 frases, la red propia queda **por debajo de TF-IDF** (0,90 frente a 0,93). Tiene que aprender el español desde cero: 988 palabras de vocabulario, y una palabra que no vio en train es `<unk>`. Los transformers ya traen ese conocimiento y suben unos 4-7 puntos. La clase diagnóstica es la más difícil para todos, y es donde más se nota la diferencia (0,87 → 0,96).
3. **Biomédico frente a general:** MrBERT-biomed gana en 3 clases (0,973 frente a 0,967), sobre todo en diagnóstica (0,961 frente a 0,944). En la tarea binaria empatan. La diferencia es de unas pocas frases, así que es indicativa. **Para la app se elige MrBERT-biomed**, por ser el mejor en distinguir diagnóstica de alarmista y el que menos falsos positivos tiene.
4. **Los errores que quedan son los casos sutiles.** Hay frases que descartan algo con certeza ("Como tu potasio está normal, tu corazón está funcionando bien y no hay riesgo de arritmias") y alarmistas con tono técnico ("suele ser un signo de deterioro hepático avanzado").
5. **Limitación importante para el informe:** el test solo tiene frases en que los dos LLM coinciden, que son las más claras, así que el 0,99 es optimista. Los 200 desacuerdos son el set difícil. Sobre ellos, MrBERT suele dar la razón al profesor: en 57 de los 89 casos "diagnóstica vs. informativa" dice diagnóstica. Cuando el equipo los revise, entran al dataset con su etiqueta y se reevalúa. Ese número es el más honesto.
6. Las frases de entrenamiento y de test las escribió el mismo profesor (Qwen). En la Fase 4 se mide el clasificador sobre las salidas reales del generador.

## Fase 4: fine-tuning QLoRA del generador ⬜

- [ ] Dataset SFT de 1.000 a 2.000 ejemplos: entrada (caso sintético + pasajes RAG + contexto), salida (JSON `{explicaciones, preguntas_sugeridas}`). Los redacta un modelo abierto grande como profesor; el equipo revisa una muestra.
- [ ] Notebook QLoRA (Unsloth o PEFT + TRL) sobre un modelo base de unos 3B.
- [ ] Evaluar en el test set tres variantes (base, fine-tuneado, Gemini):
  - % de JSON válido.
  - Fidelidad al estado y a los valores.
  - Legibilidad (INFLESZ / Fernández-Huerta).
  - % de frases marcadas por el clasificador de la Fase 3.
  - Rúbrica humana sobre unos 50 casos.

## Fase 5: integración en la app ⬜

- [ ] Selector `MODO=gemini|propio` en `backend.py` y switch en Streamlit para comparar en la demo.
- [ ] Ampliar `evaluar_estado_analito` a los formatos variantes de la Fase 1. Cuando esté, quitar el `xfail` de `test_regla_del_backend_entiende_formatos_variantes`.
- [ ] Reemplazar `recuperar_conocimiento_rag` por el índice FAISS y el filtro de `step_4_validation` por el clasificador.

## Fase 6: informe ⬜

- [ ] Tablas y gráficas de cada experimento, README actualizado con la arquitectura nueva y una sección de limitaciones y alcance.

---

## Decisiones

1. **Fecha de entrega:** entre 2 y 3 semanas desde el 2026-10-18 (entre el **2026-11-01** y el **2026-11-08**). Se planifica para 2 semanas:
   - Fuera: NER propio y modelo de extracción.
   - Solo si hay una tercera semana: ajuste contrastivo de embeddings (Fase 2 opcional) y rúbrica humana de 50 casos en vez de 20 a 30.
2. **GPU:** **Colab gratuito con GPU T4** (16 GB; no hay Colab Pro). Para la Fase 4 conviene un modelo base de 1,5B a 3B con QLoRA (Unsloth) y guardar checkpoints en Drive, porque la sesión gratuita se corta. Respaldo: Kaggle Notebooks (T4 con cuota semanal).
3. **Modelo profesor:** **Groq** (cuenta creada, `GROQ_API_KEY` en `.env` y verificada el 2026-10-18). No usar Gemini, porque sus términos prohíben usar sus salidas para entrenar modelos que compitan.
   - Principal: `qwen/qwen3.8-27b`. En la prueba tuteó, fue cálido, usó el contexto del paciente y respetó los estados.
   - Secundario o juez: `openai/gpt-oss-120b`. Más técnico; en un valor BAJO dijo "suele considerarse normal", lo que contradice el estado.
   - Límite medido: **8.000 tokens por minuto por modelo, de los que solo 1.000 pueden ser de salida**. La salida es el verdadero cuello de botella. Con unos 1.000 a 1.600 tokens por ejemplo salen 5 a 8 ejemplos por minuto, o sea unas 2 a 3 horas para 1.000 ejemplos. Falta confirmar el límite diario en console.groq.com → Settings → Limits.

### Calendario

| Días | Trabajo |
|---|---|
| 1 a 3 | Fase 2: base de conocimiento, índice FAISS y evaluación de recuperación |
| 3 a 6 | Fase 3: dataset de frases, fine-tuning de MrBERT y métricas |
| 6 a 10 | Fase 4: dataset SFT, QLoRA y evaluación comparativa |
| 10 a 12 | Fase 5: integración en la app y corrección de la regla |
| 12 a 14 | Fase 6: informe, README y ensayo de la demo |

**Prioridad si el tiempo no alcanza:** la Fase 3 es el fine-tuning garantizado (corto y con métricas claras). La Fase 4 es la de mayor valor pero también la de mayor riesgo.

---

## Cómo verificar

Desde la raíz del proyecto:

```bash
pip install -r requirements-train.txt

# Pruebas (esperado: 26 passed, 1 xfailed)
python -m pytest tests -v

# Fase 1: regenerar el dataset sintético y medir la regla determinista del backend
python training/sintetico/generar_examenes.py --n 2000 --seed 42
python evaluacion/evaluar_reglas.py

# Fase 2: volver a descargar MedlinePlus (solo si se quiere actualizar data/rag/documentos.jsonl)
python training/rag/construir_base_conocimiento.py
# Fase 2: construir el índice FAISS y evaluar los recuperadores (la primera vez descarga unos 1,2 GB de modelos)
python training/rag/recuperador.py
python evaluacion/evaluar_rag.py

# Fase 2, ajuste contrastivo: generar consultas (Groq) y entrenar (mejor en Colab: training/rag/ajuste_embeddings_colab.ipynb)
python training/rag/generar_consultas_entrenamiento.py
python training/rag/ajustar_embeddings.py --modelo alia --variante split
```

El `xfail` es intencional: documenta la brecha de formatos de la regla, que se corrige en la Fase 5.

---

## Bitácora

- **2026-10-18:**
  - Feedback del profesor: el proyecto era "usar un LLM y ya"; hay que hacer fine-tuning y participar en la construcción del modelo.
  - Revisión de modelos: Med-BERT descartado (la rama `version_2` agrega medicamentos y procedimientos, no idiomas, y trabaja con códigos ICD). MrBERT-biomed elegido para guardrails y ALIA-MrBERT para RAG.
  - Fases 0 y 1 completadas.
  - Decisiones: entrega en 2 a 3 semanas, Colab gratuito con GPU T4, Groq verificado (profesor principal: Qwen).
  - Fase 2: base de conocimiento de MedlinePlus (208 fragmentos), recuperadores y evaluación con 505 consultas. Mejor resultado: e5-small + alias (85,3 % hit@1) frente a 6,3 % del sistema actual. ALIA-MrBERT no supera a e5-small.
  - Ajuste contrastivo: 1.198 consultas generadas con Qwen, split de 29/12 analitos, script de entrenamiento y notebook de Colab listos.
  - Entrenamiento en Colab T4: e5 sin problemas. ALIA se quedó sin memoria y se corrigió con CachedMNRL (`--mini-batch 8`). Resultado: ambos generalizan a analitos no vistos (e5 96,2 %, ALIA 91,2 % hit@1). ALIA ajustado completo queda elegido por hit@3 y consultas manuales (92,5 % frente a 85,0 %). Fase 2 terminada.
  - Fase 3: 1.774 frases generadas con Qwen y verificadas con gpt-oss-120b (kappa 0,83); 1.574 utilizables. En test (analitos no vistos), MrBERT-biomed logra F1 binario 0,991 frente a 0,219 del filtro de palabras. Se agregaron como comparación una red BiLSTM propia desde cero (0,963), TF-IDF (0,977) y MrBERT-es (0,992). El preentrenamiento es lo que más aporta. Falta la revisión del equipo de los 200 desacuerdos.
