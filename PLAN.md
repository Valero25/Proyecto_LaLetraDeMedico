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

**Fase en curso:** Fase 6 (informe). La app funciona de punta a punta con los tres modelos propios, verificada con un PDF real; el generador corre en GPU en ~28 s por examen. Modelos en la app: `alia-ajustado-completo` + alias (RAG), `mrbert-biomed-v2` (guardrail) y Qwen2.5-3B + QLoRA en GGUF (generador). Alcance: solo exámenes de sangre.

| Fase | Estado |
|---|---|
| 0. Preparación | ✅ Terminada |
| 1. Generador de exámenes sintéticos | ✅ Terminada |
| 2. RAG con embeddings | ✅ Terminada |
| 3. Clasificador de guardrails | ✅ Terminada (falta la revisión del equipo) |
| 4. Fine-tuning QLoRA del generador | ✅ Terminada (falta la línea base de Gemini y la revisión del equipo) |
| 5. Integración en la app | ✅ Terminada (falta la rúbrica humana contra el modo Gemini) |
| 6. Informe | 🟡 En curso |

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

## Fase 3: clasificador de guardrails con MrBERT-biomed ✅

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
- [x] **Pérdida ponderada** (`--pesos 1,2,2` en los dos scripts de entrenamiento): en la entropía cruzada, equivocarse con una frase diagnóstica o alarmista cuesta el doble que con una informativa. Los pesos se fijaron de antemano, sin mirar el test. Se comparó con **bajar el umbral** de decisión sobre P(riesgosa), que no necesita reentrenar (`--umbrales` en la evaluación).

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
#### Pérdida ponderada frente a umbral (test, tarea binaria)

| Modelo | Sin ponderar: FN / FP | Ponderado [1, 2, 2]: FN / FP | Sin ponderar, umbral 0,3: FN / FP | Sin ponderar, umbral 0,1: FN / FP |
|---|---|---|---|---|
| TF-IDF + regresión logística | 8 / 7 (umbral 0,5) | — | **2** / 14 | 0 / 63 |
| BiLSTM propia (media de 3 semillas) | 11,3 / 12,7 | 11,3 / 12,7 | — | — |
| MrBERT-biomed | 5 / 1 | 5 / 2 | 4 / 2 | 4 / 8 |

FN = frases riesgosas que pasan; FP = frases informativas marcadas.

**Lectura** (desarrollada en la sección 7 de `informe/fundamentos.md`):
- **Ponderar la pérdida no mejora nada** en ninguno de los dos modelos. Una explicación: el train es separable para estos modelos y la pérdida llega casi a 0, así que los pesos multiplican errores que ya no existen y la solución final es la misma. En la BiLSTM, la variación entre semillas (6 a 21 FN) pesa más que los pesos.
- **Bajar el umbral sí funciona, pero solo en el modelo calibrado.** En TF-IDF, con umbral 0,3 los FN bajan de 8 a 2. MrBERT está sobreconfiado: sus 4-5 errores tienen P(riesgosa) cercana a 0, y ni con umbral 0,1 los detecta. Sus errores no son de umbral. Vienen de la representación o de la etiqueta: hay al menos un caso de etiqueta dudosa ("tú no tienes riesgo" está etiquetada como informativa, y el modelo ponderado la marca).
- **Decisión:** la app usa MrBERT-biomed sin ponderar con umbral 0,5. Bajar el umbral quita 1 FN y suma 1 FP, una diferencia que no justifica cambiarlo. Si se cambia, debe elegirse en validación y no en test.

5. **Limitación importante para el informe:** el test solo tiene frases en que los dos LLM coinciden, que son las más claras, así que el 0,99 es optimista. Los 200 desacuerdos son el set difícil. Sobre ellos, MrBERT suele dar la razón al profesor: en 57 de los 89 casos "diagnóstica vs. informativa" dice diagnóstica. Cuando el equipo los revise, entran al dataset con su etiqueta y se reevalúa. Ese número es el más honesto.
6. Las frases de entrenamiento y de test las escribió el mismo profesor (Qwen). En la Fase 4 se mide el clasificador sobre las salidas reales del generador.

## Fase 4: fine-tuning QLoRA del generador ✅

- [x] **Formato** (`training/generador/prompt.py`):
  - El estudiante ve un prompt de sistema corto con las reglas, más los datos del caso y los pasajes. Es el mismo formato que usará la app.
  - El profesor recibe además una guía de estilo (largo según el estado, causas en condicional, 3-4 preguntas) que el estudiante **no** ve: debe aprender el estilo de los ejemplos.
- [x] **Casos** (`training/generador/preparar_casos.py` → `data/generador/casos.jsonl`):
  - 1.000 casos de train, 100 de val y 150 de test, del mismo split de la Fase 1, con hasta 8 analitos.
  - Cada analito lleva el primer pasaje que recupera el RAG de la Fase 2 (ALIA ajustado). El 95,8 % de esos pasajes es correcto: el generador aprende con los mismos errores de recuperación que tendrá en la app.
- [ ] **Respuestas de los profesores** (`training/generador/generar_sft.py`): `qwen/qwen3.8-27b` y `openai/gpt-oss-120b` en paralelo, cada uno con la mitad de los casos de train y val. *En curso.*
  - gpt-oss es unas 5 veces más rápido, pero más escueto, y en la prueba tuvo un error de contenido (atribuyó la ALT baja a "menor masa muscular"). Qwen escribe explicaciones más completas. Cada ejemplo registra su profesor, para compararlos o filtrar.
  - Límites de Groq: 1.000 peticiones por día por modelo y unos 1.000 tokens de salida por minuto. Además, parece haber un límite diario de tokens: tras unas horas de uso aparecen esperas de 5 a 15 minutos. El script se puede reanudar.
  - **Decisión del 2026-10-19: entrenar con unos 500 ejemplos.** Con la cuota de Groq salen unos 100 a 250 casos por modelo al día, y 1.000 tomarían 3 a 4 días más. Para que avance más rápido:
    - El profesor recibe pasajes de 350 caracteres en lugar de 700; su prompt queda un 20 % más corto. El estudiante sigue viendo los pasajes completos.
    - Los casos de validación se generan primero.
    - La generación sigue corriendo. Si se llega a 1.000, se entrena una versión 2 y se comparan 500 contra 1.000 ejemplos (curva de datos para el informe).
  - Avance al 2026-10-19: 225 respuestas (gpt-oss 150, Qwen 75), de las que ~190 pasan las verificaciones.
- [ ] **Tercer profesor, propio y sin cuotas** (`training/generador/profesor_local.py` + `profesor_colab.ipynb`):
  - Qwen2.5-14B-Instruct-AWQ (Apache 2.0) en la T4 de Colab, con vLLM. Si vLLM no funciona en la T4, se usa Qwen2.5-7B-Instruct con transformers en 4 bits.
  - Recorre los casos de train desde el final, mientras Groq avanza desde el principio, y guarda en Drive tras cada lote.
  - Si un caso tiene respuestas de varios profesores, `dataset.py` se queda con la del más grande (Qwen3.8-27B > gpt-oss-120b > Qwen2.5-14B > Qwen2.5-7B).
  - Probado en CPU con SmolLM2-135M y el motor transformers.
- [x] **Verificaciones automáticas** (`training/generador/verificacion.py`): JSON válido, un examen por analito, menciona el valor, estado consistente, número de preguntas e índice INFLESZ (con un contador de sílabas en español). Sirven para filtrar el dataset y para evaluar.
- [x] **Dataset SFT** (`training/generador/dataset.py` → `sft_train.jsonl` y `sft_val.jsonl`): **555 ejemplos de train y 97 de validación**, de 810 respuestas.
  - Por profesor, en train: Qwen2.5-14B 273, gpt-oss-120b 177, Qwen3.8-27B 105. Validación: Qwen3.8-27B 50, gpt-oss 47.
  - Descartes: falta un examen (113; casi todos del 14B, que se salta exámenes o los renombra), estado inconsistente (35), preguntas o exámenes extra (9), JSON inválido (1).
  - **Se armó sin el filtro del guardrail** (`--sin-guardrails`), porque el guardrail marca de más en texto real (ver el hallazgo abajo). Con el filtro quedaban solo 376 de train.
- [x] **Hallazgo: el guardrail no generaliza a texto real.** Al pasar las 6.240 oraciones de los profesores por MrBERT-biomed, marcó 301 oraciones (en 194 respuestas), con probabilidades altas: 0,96 para "Las plaquetas son pequeñas células de la sangre que ayudan a formar coágulos".
  - En una muestra de 30 revisada a mano, solo 5 o 6 eran riesgosas según nuestras definiciones: descartar con certeza ("tu tiroides está funcionando correctamente") o sugerir un tratamiento ("considerar suplementos"). La precisión en texto real ronda el **20 %**, frente a 0,997 en el test sintético.
  - Causa probable: **cambio de distribución**. MrBERT aprendió con frases que un solo profesor escribió por encargo, una clase a la vez. Los generadores reales escriben muchas definiciones ("X es una hormona que..."), casi ausentes en el entrenamiento. Además está sobreconfiado (sección 7.1 de `fundamentos.md`), así que subir el umbral no lo corrige.
  - Las oraciones marcadas quedaron en `data/generador/descartadas_guardrail.jsonl`.
  - Siguiente paso: **guardrail v2**. Se etiquetan unas 300 oraciones reales (equipo o juez), se suman al entrenamiento, se reentrena y se mide la precisión en texto real.
- [x] **Guardrail v2** (`training/guardrails/muestra_reales.py` y `ajustar_clasificador.py --con-reales` → `models/guardrails/mrbert-biomed-v2`):
  - Muestra de **574 oraciones reales**, sin repetidas: las 287 que el v1 marcó y 287 que no marcó. El juez gpt-oss-120b las etiquetó sin ver la predicción del v1 (`data/guardrails/reales.csv`, con la columna `clase_equipo` para que el equipo corrija).
  - Según el juez, de las 287 marcadas por el v1 solo **9 (3 %) son riesgosas**. De las no marcadas, 2 (1 %).
  - El v2 se entrenó con 899 frases sintéticas y 345 reales del split de train (mismos hiperparámetros, 50 min en CPU). Las reales de los 12 analitos de test (168) quedan como **test real**.

| Test | Métrica | v1 (solo sintético) | **v2 (sintético + real)** |
|---|---|---|---|
| Sintético (506) | F1 binaria | 0,991 | 0,989 |
| | F1 macro (3 clases) | 0,973 | 0,967 |
| | F1 diagnóstica | 0,961 | 0,950 |
| | FN / FP | 5 / 1 | 6 / 1 |
| **Real (168: 166 informativas, 2 diagnósticas)** | **Falsos positivos** | **69 (42 % de las informativas)** | **3 (1,8 %)** |
| | Riesgosas detectadas | 2 de 2 | 1 de 2 |

  **Lectura:**
  1. **El v2 corrige casi por completo el cambio de distribución.** Los falsos positivos en texto real bajan de 69 a 3, y en el test sintético pierde muy poco (F1 binaria de 0,991 a 0,989). TF-IDF tampoco sirve en texto real: 52 falsos positivos.
  2. La pequeña caída en la clase diagnóstica sintética (0,961 → 0,950) es coherente con el **sesgo del juez**: gpt-oss tiende a ver como informativas las frases que descartan algo con certeza, que nuestra definición considera diagnósticas, y el v2 aprendió un poco de esa tendencia.
  3. **Limitación:** el test real tiene solo 2 frases riesgosas, así que el recall en texto real no se puede medir con confianza. La riesgosa que el v2 deja pasar es dudosa: "Esto puede deberse a que tu tiroides no está produciendo suficiente T4" está en condicional. Las etiquetas reales vienen del juez y no de humanos; la revisión del equipo de `reales.csv` daría la cifra definitiva.
  4. **Decisión:** la app usará **MrBERT-biomed v2**. La evaluación del generador también se rehízo con el v2.
- [x] **Entrenamiento** (`training/generador/ajustar_qlora.py` + `qlora_colab.ipynb`):
  - Qwen2.5-3B-Instruct cuantizado a 4 bits (NF4), con LoRA r = 16 y α = 32 en las 7 capas lineales de atención y MLP. lr de 2e-4 con calentamiento y caída coseno, lote efectivo de 16, 2 épocas.
  - Los checkpoints van a Drive y el entrenamiento se reanuda si Colab se desconecta.
  - Probado en CPU con SmolLM2-135M: el conteo de LoRA cuadra con r(d + k), se guardan las curvas de pérdida y **la pérdida cubre solo la respuesta**. En el ejemplo verificado, 832 de 2.743 tokens llevan pérdida, y el primero es el `{"explicaciones"` del asistente.
- [x] **Respuestas del test** (`training/generador/generar_respuestas.py`), del modelo base y del ajustado, con decodificación voraz. **Evaluación** (`evaluacion/evaluar_generador.py`). Los dos están probados de punta a punta en CPU.
- [x] **Entrenamiento en Colab (T4), 2026-10-20:**
  - 555 ejemplos × 2 épocas = 70 pasos, en 50 min. Entrenables: 29,9 M, cerca del 1 % del modelo (Qwen2.5-3B tiene ~3.090 M; el total de 1.729 M que imprime el script cuenta los pesos de 4 bits empaquetados).
  - Pérdida en train de 0,79 a 0,42. En validación: 0,607 (paso 25), 0,547 (paso 50) y 0,538 (paso 70), siempre bajando. La exactitud por token en validación pasa de 82,4 % a 84,0 %.
  - La norma del gradiente se mantuvo entre 0,24 y 0,39: el recorte a 1 nunca actuó.
  - Primer intento fallido: los pesos en bfloat16 no los soporta la T4. Se corrigió con LoRA en float32 y el resto en float16.
- [x] **Resultados en test (150 casos no vistos, sin guardrail por ahora):**

| Métrica | Qwen2.5-3B base | **Qwen2.5-3B ajustado (QLoRA)** |
|---|---|---|
| JSON válido | 97,3 % | **99,3 %** |
| Cobertura (un examen por analito, con su nombre) | 68,7 % | **99,3 %** |
| Estado consistente (alto, bajo, normal) | 69,7 % | **99,4 %** |
| Menciona el valor del paciente | 10,3 % | **70,7 %** |
| Legibilidad INFLESZ (55-65 normal, 65-80 bastante fácil) | 65,2 | 63,6 |
| Palabras por respuesta | 123 | 198 |
| Segundos por caso en la T4 | 8 | 14 |

  **Lectura:**
  1. **El ajuste cumple su función.** El base omite o renombra exámenes en un tercio de los casos, y escribe el estado en mayúsculas o lo inventa ("muestran niveles NATURALES"). El ajustado cubre todos los exámenes, respeta el estado y casi siempre menciona el valor y el rango.
  2. **El base diagnostica:** "Esto indica que tu glucosa está controlada y que **no tienes diabetes ni prediabetes**". El ajustado, en el mismo caso: "tu resultado es 5.4%, lo cual está dentro del rango normal (<5.7%)...".
  3. **El estudiante hereda los hábitos de sus profesores.** "Menciona el valor" queda en 70,7 % porque el 49 % del train viene del Qwen2.5-14B, que solo lo hace en el 41,5 % de sus respuestas (gpt-oss: 99,4 %; Qwen3.8-27B: 75,1 %). El promedio ponderado por profesor da ~66 %, casi lo mismo que logra el estudiante. Para una v2: exigir "menciona el valor" en el filtro del dataset.
  4. La legibilidad casi no cambia (65 frente a 64, en el límite entre "normal" y "bastante fácil"). Las respuestas son más largas porque explican más: qué mide el examen, el valor, el rango y las posibles causas.
  5. **Calidad del contenido:** aparecen explicaciones poco precisas ("tu cuerpo necesita más energía de la que está proporcionando" para una glucosa alta). Las métricas automáticas no detectan esto; hace falta la rúbrica humana.
- [x] **Guardrail sobre las salidas del test** (P(riesgosa) ≥ 0,5):

| | Base, con v1 | Ajustado, con v1 | **Base, con v2** | **Ajustado, con v2** |
|---|---|---|---|---|
| Oraciones marcadas como riesgosas | 19,6 % | 4,6 % | **6,4 %** | **0,5 %** |
| Respuestas con al menos una oración marcada | 74,0 % | 29,5 % | **41,8 %** | **5,4 %** |

  - Con el **v2**, que confunde mucho menos en texto real, la diferencia se agranda: el ajustado escribe **13 veces menos** oraciones riesgosas que el base (0,5 % frente a 6,4 %), y solo 1 de cada 20 respuestas tiene alguna.
  - Lo que el v2 marca en el **base** son riesgos reales: "no tienes diabetes ni prediabetes", "tu tiroides no está produciendo suficiente hormona tiroidea", "tu tiroides está funcionando correctamente y no hay evidencia de hipotiroidismo".
  - En el **ajustado** marca afirmaciones con certeza ("indica que tus riñones no están eliminando este desecho de manera adecuada", "no hay evidencia de daño hepático") y otras ya en condicional ("puede indicar prediabetes o diabetes"). En la app (Fase 5), el guardrail v2 revisará cada respuesta y señalará esas oraciones.
- [ ] Generar las respuestas de Gemini sobre el mismo test, como referencia, solo para evaluar. Sus salidas no se usan para entrenar, porque sus términos lo prohíben.
- [ ] El equipo revisa una muestra de unos 30 ejemplos del dataset SFT.
- [ ] Evaluar en el test set tres variantes (base, fine-tuneado, Gemini):
  - % de JSON válido.
  - Fidelidad al estado y a los valores.
  - Legibilidad (INFLESZ / Fernández-Huerta).
  - % de frases marcadas por el clasificador de la Fase 3.
  - Rúbrica humana sobre unos 50 casos.
- [x] **Rúbrica humana preparada (`evaluacion/rubrica.py`, 2026-11-02): base frente a ajustado, a ciegas.**
  - **Casos:** 30 del split de test, solo de sangre, repartidos entre los 8 tipos de examen; 24 tienen algún valor fuera de rango, porque los casos todo normal casi no discriminan.
  - **Ciego:** en cada caso las dos respuestas aparecen como A y B en orden aleatorio (el ajustado es A en 15 de 30). La clave está en `evaluacion/rubrica/clave.json` y no se abre antes de evaluar.
  - **Criterios (1 a 5, con anclas):** exactitud, claridad, seguridad, completitud y utilidad de las preguntas. Además, la respuesta preferida (A, B o empate) y un comentario.
  - **Pasos para el equipo:**
    1. Cada evaluador copia `evaluacion/rubrica/plantilla_rubrica.xlsx` a `evaluacion/rubrica/respuestas/rubrica_<nombre>.xlsx` y la llena.
    2. Se corre `python evaluacion/rubrica.py resumir` → `evaluacion/resultados/fase4_rubrica.json`.
  - **Qué calcula `resumir`:** media y desviación por criterio y modelo, prueba de Wilcoxon pareada por caso (con el promedio entre evaluadores), conteo de preferencias, acuerdo y kappa de Cohen entre evaluadores, y los comentarios atribuidos al modelo correcto.
  - Se excluye Gemini: el motor principal es el propio, y la comparación que importa para el informe es el efecto del ajuste.
  - `tests/test_rubrica.py`: ciego, solo sangre y resumen con dos evaluadores simulados. **154 passed.**
- [ ] **El equipo llena la rúbrica** (al menos 2 evaluadores, para medir acuerdo) y se corre `resumir`.
- [x] **Generador v2: datos preparados (2026-11-02).** Mismo modelo y mismos hiperparámetros; solo cambian los datos, para aislar el efecto de su calidad.
  - **Verificaciones de calidad nuevas** (`verificacion.calidad`), a partir de errores vistos con PDFs reales:
    - causas repetidas entre exámenes;
    - siglas con un significado inventado (solo cuenta un paréntesis cuyas iniciales forman la sigla, así "HDL (colesterol bueno)" no es error);
    - "usted" en vez de "tú";
    - preguntas que la revisión de la app quitaría.

    Se usan como filtro del dataset y como métricas en `evaluar_generador.py`.
  - **Diagnóstico:**

    | | Profesor 14B | gpt-oss-120b | Qwen3.8-27B | Estudiante v1 (test) | Qwen base (test) |
    |---|---|---|---|---|---|
    | Causas repetidas | **11,8 %** | 0 % | 0 % | **13,4 %** | 13,0 % |
    | Siglas inventadas | 0 % | 0 % | 0,6 % | 1,3 % | 0 % |
    | Usa "usted" | 0 % | 2,1 % | 1,8 % | 0 % | 15,8 % |
    | Con alguna pregunta que se quitaría | 10,8 % | 6,6 % | 12,0 % | 6,7 % | 20,5 % |
    | Menciona el valor en todos los exámenes | **34 %** | 97,5 % | 59,9 % | 68,5 % (en alterados) | 10,7 % (en alterados) |

    - Las causas repetidas y el valor omitido vienen del profesor 14B, que aportó el 49 % del train del v1: **el estudiante hereda los defectos de sus datos** (sección 10 de `fundamentos.md`).
    - Las siglas inventadas, en cambio, casi no están en los datos: son una alucinación propia del estudiante. Por eso además se corrigen en la app (`verificacion.corregir_siglas` dentro de `revisar_coherencia`): "VCM (Volumen de Célula Móvil)" → "VCM (volumen corpuscular medio)".
  - **Dataset v2** (`dataset.py --sin-guardrails --version v2` → `sft_train_v2.jsonl` y `sft_val_v2.jsonl`; el v1 no se toca):
    - Los filtros del v1, más:
      - menciona el valor de todo examen fuera de rango;
      - sin causas repetidas, sin siglas inventadas y sin "usted";
      - las preguntas se limpian con la revisión de la app y quedan al menos 2.
    - Resultado: **421 de train y 93 de val** (v1: 555 y 97). El 14B baja del 49 % al 35 % del train.
    - Descartes nuevos: 118 por no mencionar el valor de un alterado, 12 por causas repetidas, 6 por "usted", 1 por una sigla inventada y 1 por menos de 2 preguntas coherentes.
  - **Entrenamiento:** `ajustar_qlora.py --version v2` y `qlora_v2_colab.ipynb` (entrena, responde los 150 casos de test y exporta a GGUF en una sesión de la T4, unas 1,5 h). Probado en CPU con SmolLM2-135M.
  - **Pruebas:** `test_pdf_real.py` con la corrección de siglas y las verificaciones de calidad. **156 passed.**
- [x] **Generador v2 entrenado y evaluado (2026-11-03).** Colab T4, 35,9 min, 54 pasos. La pérdida de train baja de 0,80 a 0,45 y la de validación de 0,58 a 0,55, sin sobreajuste. No se compara con la validación del v1 (0,538) porque los dos conjuntos de validación son distintos.

  | 150 casos de test | Base | Ajustado v1 | **Ajustado v2** |
  |---|---|---|---|
  | JSON válido | 97,3 % | 99,3 % | **100 %** |
  | Respeta el estado calculado | 69,7 % | 99,4 % | 99,4 % |
  | Menciona el valor en los alterados | 10,7 % | 68,5 % | **79,3 %** |
  | Causas repetidas | 13,0 % | 13,4 % | **5,3 %** |
  | Siglas inventadas | 0 % | 1,3 % | **0 %** |
  | Con alguna pregunta quitada | 12,3 % | 2,0 % | **1,3 %** |
  | Respuestas con oración riesgosa (guardrail v2) | 41,8 % | 5,4 % | **2,7 %** |
  | INFLESZ (legibilidad) | 65,2 | 63,6 | 63,3 |
  | Segundos por caso (T4) | 8 | 14 | 15 |

  - **Con menos datos pero más limpios, el v2 mejora en todo lo que ataca el filtro y no empeora en nada.** Las causas repetidas bajan a menos de la mitad, y la mención del valor sube 11 puntos aunque sigue lejos del 97,5 % de gpt-oss. Es la evidencia de que la calidad de los datos pesa más que su cantidad (sección 10 de `fundamentos.md`). Figura: `informe/figuras/fig6_generador_v2.png`.
  - **El filtro de preguntas tenía falsos positivos.** Al principio daba 10,7 % de preguntas quitadas en el v2 frente a 6,7 % en el v1. Revisando caso por caso, casi todas eran preguntas correctas:
    - "glucosa **bajo control**" se leía como "glucosa baja";
    - "bilirrubina" a secas se tomaba como la de orina, aunque el examen tenía "Bilirrubina total";
    - "colesterol" a secas se tomaba como colesterol total, aunque el examen tenía Col. LDL y HDL;
    - "mis antecedentes familiares de colesterol alto" se marcaba como examen ajeno, y era el motivo de consulta del paciente.

    Se corrigió `revisar_coherencia`, que también usa la app, con una prueba nueva en `test_pdf_real.py`. Las cifras de la tabla ya usan el filtro corregido para los tres modelos. **157 passed.**
  - **Lo que queda:** el guardrail aún marca una confusión real del v2, "Un MCV elevado puede indicar una anemia **microcítica**" (debería ser macrocítica). La app la señala al paciente.
  - **Decisión:** la app pasará a usar el **v2**, después de probar su GGUF (`letrademedico-qwen2.5-3b-v2-q4_k_m.gguf`) con los dos PDFs reales.
- [x] **GGUF v2 probado de punta a punta con los dos PDFs reales (2026-11-04, RTX 4050, lectura local, sin Gemini).**

  | PDF | Analitos | v1 | **v2** |
  |---|---|---|---|
  | Endocrinología 2021 (incluye la carga de los modelos) | 8 | 158 s | **130 s** |
  | Colsanitas 2011 (con los modelos ya cargados) | 4 | 19 s | **15 s** |

  - En los dos modelos: todos los exámenes con su valor, sin siglas inventadas, sin "usted", sin explicaciones cambiadas por plantilla y 0 oraciones marcadas por el guardrail.
  - El v2 además escribe el rango de referencia dentro del texto ("ligeramente por encima del rango normal (menor a 150 mg/dl)").
  - Lo que queda en los dos: en Colsanitas, LDL y triglicéridos comparten la misma causa ("dieta alta en grasas o menor actividad física"). Es correcta según MedlinePlus (`mlp-prueba-de-trigliceridos-01` y `mlp-niveles-de-colesterol-06`), pero el RAG no le entregó esos fragmentos: el modelo acertó por lo aprendido en el entrenamiento.
  - **Error del v2 en la AST alta:** escribe "una menor actividad física", una causa inventada y al revés (MedlinePlus, `mlp-prueba-de-ast-03`: enfermedad hepática, hepatitis, alcohol…, y es el *ejercicio* lo que puede alterar el resultado). El guardrail no lo marca porque va en condicional. **Causa de fondo:** el RAG busca solo por el nombre del examen y le entregó el fragmento `-01` ("se incluye en un análisis de rutina"), que no trae causas.
- [x] **RAG según el estado (2026-11-04).** Para un analito alto o bajo, se elige entre los fragmentos de ese mismo analito el que explica ese resultado (`modelos_propios.puntaje_estado`).
  - **Criterio:** cuenta las oraciones, en los 700 caracteres que recibe el generador, que dicen la dirección del resultado ("altos", "más bajo de lo normal", "hipercalcemia"…) junto con una causa o un significado ("pueden deberse", "signo de", "incluyen"). Suma 1 si el fragmento es de "¿Qué significan los resultados?". En empate gana el fragmento con más palabras de esa dirección que de la contraria.
  - **Exclusiones:** no cuentan las oraciones sobre orina, porque la app es solo de sangre, ni "baja/alta densidad" ni "alto contenido".
  - **Sin cambios:** los resultados normales y los analitos sin alias siguen con la búsqueda anterior.
  - **Evaluación** (`evaluacion/evaluar_rag_estado.py` → `fase5_rag_estado.json`): 44 combinaciones de analito y estado, etiquetadas leyendo los fragmentos, cada una con un nombre corto y uno largo de laboratorio (88 consultas).

    | Búsqueda | Fragmento que explica el resultado |
    |---|---|
    | Solo por el nombre (antes) | 14 / 88 (15,9 %) |
    | **Con el estado** | **84 / 88 (95,5 %)** |

  - **Proceso:**
    - El primer intento, solo con palabras sueltas, empeoraba glucosa, creatinina y hematocrito.
    - Restar las oraciones del resultado contrario castigaba a los fragmentos que explican las dos direcciones (glóbulos blancos, VCM).
    - Agregar "puede/pueden" como palabras de causa bajó el acierto a 75 %.
    - Tras la primera corrida se aceptaron como correctos 4 fragmentos que, releídos, sí explican el resultado: glucosa-01, a1c-01, albumina-00 y tsh-00.
    - **El criterio se ajustó sobre este mismo conjunto, así que el 95,5 % es optimista.**
  - **Lo que falla:**
    - Plaquetas altas: el texto útil de `-02` queda después de los 700 caracteres.
    - VCM alto: el criterio elige `-03`, que habla de la HCM alta; no separa dos índices dentro del mismo fragmento.
  - **PDF Colsanitas con el v2:**
    - La AST alta ahora recibe `mlp-prueba-de-ast-03` y la "menor actividad física" desaparece: "puede deberse a una inflamación leve del hígado o a una condición metabólica".
    - Queda una muletilla del generador: "una condición metabólica" se repite en LDL, triglicéridos y AST. No viene del pasaje.
    - **Causa de fondo:** el v1 y el v2 se entrenaron con pasajes buscados solo por nombre, así que aún no aprovechan del todo un fragmento de causas. Un v3 entrenado con los pasajes según el estado lo atacaría desde los datos.
  - **Pruebas:** dos nuevas en `test_pdf_real.py`, sin modelos. **159 passed.**
  - **Decisión:** el v2 queda por defecto (`modelos_propios.generador_por_defecto`). Si su GGUF no está, se usa el v1, y `LDM_MODELO_GENERADOR` elige otro.
- [x] Evidencia teórica que exige `informe/fundamentos.md` (sección 10). Las secciones 10.2 y 10.3 se agregaron el 2026-11-04:
  - **10.2:** calidad frente a cantidad de datos con el v2. Incluye por qué la pérdida de validación no ve la mejora.
  - **10.3:** cambio de distribución en la entrada, con los pasajes del RAG según el estado.
  - `fig4_curvas.png` muestra las curvas del v1 y del v2 por época.
  - Conteo de parámetros entrenables de LoRA frente al total, por matriz: $r(d+k)$ frente a $d \cdot k$.
  - Curvas de pérdida de train y validación, para el análisis de sobreajuste.
  - Pérdida solo sobre los tokens de la respuesta.
  - Justificación de la tasa de aprendizaje.

## Fase 5: integración en la app ✅

- [x] **Regla de estados ampliada** (`evaluar_estado_analito`):
  - Entiende "7 a 56", "Hasta 200" (incluye el límite), "Menor de 5.7", "Mayor de 50", "<=" y ">=", y cruces o trazas en orina ("1+", "++", "Trazas").
  - Exactitud sobre los 10.604 analitos sintéticos: **83,4 % → 100 %**, sin ningún estado equivocado (`evaluacion/resultados/fase5_reglas.json`; el de la Fase 1 se conserva).
  - Se quitó el `xfail`, que ahora pasa, y se agregaron 11 tests de casos límite.
- [x] **Selector de modo** (`backend.py`): `LetraDeMedicoBackend(modo="gemini" | "propio")`, o la variable `MODO`.
  - **gemini**: el sistema original intacto, como línea base: RAG por subcadena, Gemini redacta y filtro de palabras.
  - **propio**: los tres modelos del equipo (`modelos_propios.py`). En ambos modos, la lectura del documento (OCR + NER) sigue en Gemini.
  - Los modelos se cargan una sola vez y se pueden inyectar; los tests usan dobles.
  - Nuevo `analizar_entidades()`: corre el pipeline desde los analitos, sin OCR. Lo usa la demo.
- [x] **`modelos_propios.py`:**
  - `RecuperadorRAG`: ALIA ajustado + FAISS, con el índice guardado en `models/rag/indice-alia-ajustado/`. Carga en ~27 s; la primera vez lo construye en ~4 min. Agrega un **reordenamiento por alias**: entre los 5 primeros resultados, prefiere el fragmento que el catálogo asocia al nombre exacto del examen. En las 182 consultas del set del generador sube el acierto de **97,8 % a 99,5 %** y corrige "Glucosa" → página de HbA1c, que salía en la propia demo.
  - `Guardrail`: MrBERT-biomed v2. Devuelve las oraciones marcadas de cada explicación y carga en ~12 s.
  - `GeneradorPropio`: el GGUF del generador con `llama-cpp-python`, decodificación voraz y salida restringida a JSON con una gramática.
  - RAM medida del RAG más el guardrail: 2,3 GB.
- [x] **App (`app.py`):**
  - Selector de motor en la barra lateral. El modo propio solo se ofrece si sus modelos están en disco (`modelos_propios.faltantes()`).
  - Los modelos se cachean con `st.cache_resource`.
  - El botón de ejemplo, en modo propio, pasa por el RAG, el generador y el guardrail **de verdad**, sin OCR ni Gemini.
  - Las oraciones marcadas por el guardrail se muestran una por una, y hay una sección con las fuentes de MedlinePlus de cada examen.
- [x] Tests de integración (`tests/test_integracion.py`), sin modelos ni Gemini. **51 passed.**
- [x] `llama-cpp-python` 0.3.36 instalado con ruedas precompiladas para CPU (`requirements-propio.txt`).
- [x] **Exportación a GGUF** (`training/generador/exportar_gguf_colab.ipynb`, 2026-10-22): LoRA fusionado con el modelo base en float16 y cuantizado a Q4_K_M. Pasa de 5,9 GB a **1,8 GB** (4,99 bits por peso). En Colab hubo que desinstalar `torchao` 0.10, que la versión actual de peft rechaza.
- [x] **Prueba de punta a punta en CPU** (12 hilos lógicos, 8 núcleos físicos):
  - Generador solo: carga en 4 s; un caso de 4 analitos tarda **69 s con 8 hilos** (75 s con 12). JSON válido, cobertura completa y valores mencionados.
  - Pipeline completo con el examen de ejemplo (6 analitos): los modelos cargan en 28 s y el examen se explica en **98 s**. 6 de 6 estados correctos, 6 de 6 páginas de RAG correctas y 0 oraciones marcadas. Usa 5,5 GB de RAM.
  - App en Streamlit, probada sin navegador con `streamlit.testing.AppTest`: arranca en 31 s sin excepciones, ofrece los dos motores y la demo en modo propio tarda 85 s.
- [x] **Robustez del pipeline y de la app (2026-11-02):**
  - **Una explicación por analito, siempre** (`alinear_explicaciones` en `backend.py`): se empareja por nombre normalizado, luego por inclusión y luego por posición (solo si las cantidades coinciden). Antes, si el generador renombraba un examen, la app lo mostraba como **NORMAL por defecto**, y si lo omitía, desaparecía de la pantalla. Ahora conserva su estado real y, si no hay explicación, muestra un aviso.
  - **Generación por lotes de 8 analitos**, el máximo visto en el entrenamiento. Un lote que falla no tumba a los demás. Los analitos omitidos se piden una vez más en un lote distinto (con decodificación voraz, repetir el mismo prompt daría la misma salida).
  - **El prompt nunca desborda el contexto:** `GeneradorPropio` cuenta los tokens y, si no queda sitio para la respuesta, acorta los pasajes a la mitad o los quita.
  - **Fallos aislados:**
    - Si el RAG falla, se genera sin pasajes y se avisa.
    - Si el guardrail falla, la app dice "No verificado" en vez de "Guardrail activo".
    - Las salidas raras de Gemini (lista suelta, texto nulo, números en vez de cadenas) se limpian con `limpiar_entidades`.
  - **App:**
    - El HTML que viene del documento o del generador se escapa.
    - La tarjeta del guardrail muestra lo que marcó de verdad.
    - Nuevo estado "SIN EVALUAR" en gris, que también aparece en la ficha descargable.
    - El indicador gráfico usa la misma normalización de rangos que la regla, así que ya entiende "Hasta", "Menor de" y "7 a 56".
    - La nota del guardrail ya no sale duplicada.
    - El archivo se lee con `getvalue()`.
    - Los presets del modo Gemini se rotulan como muestra escrita.
  - **Chat de dudas:**
    - En modo propio responde con el fragmento de MedlinePlus que recupera el RAG, sin las oraciones que marca el guardrail y con su fuente. Antes devolvía siempre el mismo texto fijo.
    - En modo Gemini usa los reintentos y el modelo de respaldo.
  - Versiones de la app fijadas en `requirements.txt` y `requirements-propio.txt`.
  - `tests/test_robustez.py`: 21 pruebas nuevas. **77 passed.**
  - **Prueba real en CPU, hemograma y química de 14 analitos:**
    - 2 lotes (8 + 6), 14 de 14 explicaciones, estados correctos y 0 oraciones marcadas.
    - Análisis: 249 s. Carga de los modelos: 130 s (en frío).
    - Antes de este cambio, un examen así superaba lo visto en el entrenamiento y el contexto de 4.096 tokens.
  - **Hallazgo de calidad (no lo detecta ninguna métrica automática):** el generador inventa expansiones de siglas, como "VCM (Volumen de Célula Móvil)" o "HCM (Hidrógeno de la Célula Móvil)". Para una v2: incluir las expansiones correctas en los pasajes o en el dataset, y revisarlo en la rúbrica humana.
- [x] **Primer PDF real (laboratorio Compensar, 8 analitos, 2026-11-02).** El equipo lo probó en el navegador y se verificó contra el documento.
  - **Bien:**
    - Gemini extrajo los 8 valores, unidades y rangos.
    - La regla acertó los 8 estados (la IGF-1 ALTA, que el PDF marca con "*").
  - **Errores encontrados y corregidos:**
    1. **RAG:** "GLUCOSA EN SUERO U OTRO FLUIDO DIFERENTE A ORINA" recuperaba *glucosa en orina*. La SOMATOMEDINA C (IGF-1), que no está en la base, recibía *creatinina*, y el generador la explicó como creatinina y como "normal".
       - Corrección: alias del catálogo como palabras completas dentro del nombre (gana el más largo; un empate sangre/orina se resuelve por el contexto), más alias reales de laboratorio en `ALIAS_EXTRA` (no en el catálogo, para no cambiar la Fase 1).
       - Sin alias, se exige similitud ≥ 0,55: el máximo de 37 exámenes no cubiertos × 2 tipos fue 0,54. Por debajo, el analito va **sin guía** y la app avisa.
       - En las 182 consultas del set del generador: **99,5 % → 100 %**, ninguna sin guía.
    2. **Preguntas incoherentes:** "¿Qué significa un resultado alto en la hormona estimulante del tiroides?" con la TSH normal. `revisar_coherencia` quita las que dicen alto o bajo de un analito que no lo está, o que nombran un examen que el paciente no se hizo.
    3. **Explicación contra el estado:** si un texto no menciona su estado (`estado_consistente`, la métrica de la Fase 4), se reemplaza por una plantilla con valor, rango y estado, y se avisa.
    4. **HDL sin rango:** el PDF solo da niveles de riesgo. El prompt de extracción ahora toma el nivel "sin riesgo" ("Mayor de 55") → NORMAL. Gemini también tomó ese nivel en el colesterol total y el LDL, aunque había intervalo; el estado no cambia.
  - **Velocidad del generador** (mismo examen de 8 analitos; i5-13420H y RTX 4050 Laptop de 6 GB):

    | Configuración | Leer el prompt (~1.640 tokens) | Escribir (~710 tokens) | Total |
    |---|---|---|---|
    | CPU con gramática JSON (antes) | 32 s | 5,6 tok/s | 159 s |
    | CPU sin gramática | 31 s | 13,2 tok/s | **85 s** |
    | CPU + prompt lookup decoding | 34 s | 3,4 tok/s | 282 s (descartado) |
    | GPU con gramática | 2,4 s | 4,1 tok/s | 175 s |
    | **GPU sin gramática** | **2,1 s** | **27,6 tok/s** | **28 s** |

    - **La gramática JSON era el cuello de botella:** llama.cpp la evalúa en CPU sobre los ~151 mil tokens del vocabulario en cada paso.
    - El modelo ajustado ya escribe JSON válido (99,3 % en la Fase 4), y en CPU la salida es idéntica con y sin gramática. Ahora se genera sin ella y solo se reintenta con ella si el JSON sale inválido.
    - La GPU se detecta sola (`requirements-gpu.txt`; `LDM_GPU_CAPAS=0` la desactiva). Si no hay GPU, sigue en CPU.
    - De punta a punta con el PDF: lectura con Gemini en 25 s (con dos reintentos por un 503) y el resto en **27 s**.
  - **Ficha descargable en PDF** (A4, texto seleccionable, casillas y espacio para notas) **e imagen PNG** (vertical, para el celular), además del texto (`ficha_consulta.py`, fuentes DejaVu en `assets/fuentes`).
  - Calidad que queda: "ultraseñalada" en lugar de "ultrasensible", y causas posibles algo especulativas para la IGF-1 ("dieta alta en proteínas"). Van a la rúbrica humana y al generador v2.
  - `tests/test_pdf_real.py`: 18 pruebas de regresión con ese documento. **95 passed.**
- [x] **Robustez con formatos reales de sangre y alcance solo sangre (2026-11-02):**
  - **Sondeo de la regla** con 33 formatos reales que el dataset sintético no cubría: **22 → 33 correctos**.
  - 4 de los fallos daban un estado **equivocado**:
    - Rangos por sexo: siempre el de hombres. Una mujer con hemoglobina 13,0 salía BAJO.
    - HbA1c 5,9 % con "Normal: < 5.7 Prediabetes: 5.7 - 6.4" salía NORMAL.
    - Colesterol 110 con "Deseable: < 200 Limítrofe: 200 - 239" salía BAJO.
  - Correcciones:
    - `segmentos_rango` y `_estado_con_categorias` usan el segmento del sexo del paciente, si no el intervalo único del laboratorio, si no el segmento "normal/deseable/sin riesgo".
    - Con solo categorías de riesgo da DESCONOCIDO. Sin sexo, un rango por sexo solo se resuelve si los dos dan el mismo estado.
    - Nuevas palabras: "menor a", "superior a", "máximo", "< o =", "igual o mayor a".
    - Dataset sintético: sigue en 10.604 / 10.604.
  - **Rango aplicado:** la regla devuelve el segmento que usó. Ese segmento es el que ven el indicador gráfico, la ficha y el generador, para que nada la contradiga.
  - **Extracción:** Gemini copia el rango completo con sus categorías (no elige un nivel) y devuelve el sexo, la edad (no el nombre) y la muestra de cada analito.
    - En el PDF real: paciente {M, 15}, 8 de 8 en "sangre" y 8 de 8 estados correctos con los rangos completos.
    - El generador recibe el sexo y la edad, como en su entrenamiento.
  - **Solo sangre:**
    - Los analitos de orina u otras muestras conservan su estado, pero no pasan por el RAG ni por el generador, y llevan un aviso.
    - Un documento sin analitos de sangre devuelve un error claro.
    - El preset de función renal cambió "Proteínas en orina" por potasio.
  - **Preguntas:** con el PDF real, el generador repitió 4 veces "¿Qué significa tener un nivel **alto** de hormona estimulante del tiroides?" (la TSH era normal), cada vez más larga.
    - La revisión las quitó todas.
    - Ahora también descarta las casi repetidas (70 % de palabras en común) y registra en consola qué quita y por qué.
  - La ficha PDF ya no parte el aviso entre dos páginas.
  - `tests/test_sangre.py` (37 pruebas) y una regresión más en `test_pdf_real.py`. **133 passed.**
- [x] **Motor principal por defecto (2026-11-02):** el backend usaba Gemini si no se le indicaba el modo; ahora usa los modelos propios. Gemini queda como segunda opción (comparación y respaldo), con su aviso en la app. **134 passed.**
- [x] **Lectura local de PDFs digitales (`lector_pdf.py`, 2026-11-02):** con el motor principal, un PDF del laboratorio se procesa entero sin Gemini.
  - pypdf extrae la capa de texto en modo layout, que conserva las columnas. Unas reglas reconocen dos diseños:
    - por bloques: nombre y debajo "Resultado: valor unidad rango", con las categorías debajo;
    - por tabla: una fila por analito, con o sin columna de banderas H/L/*.
  - También extrae el sexo y la edad (nunca el nombre) y la muestra por la sección (UROANÁLISIS → orina).
  - Si no hay capa de texto (foto o escaneo) o la lectura no es confiable (un resultado sin nombre), pasa a Gemini.
  - La app dice cómo se leyó el documento y ofrece "Leerlo otra vez con Gemini".
  - **Medición** (`evaluacion/evaluar_lector_pdf.py` → `fase5_lector_pdf.json`): 150 casos de test × 3 diseños = 450 PDFs sintéticos de respuesta conocida, leídos al **100 %** en nombre, valor, unidad, rango, muestra y paciente, sin analitos de más.
    - En el camino se corrigieron tres errores. Las abreviaturas de una o dos letras ("K", "Ca") se descartaban como nombre. "NIT" en la lista de ruido borraba "Nitritos" y "Nitrógeno ureico". Las cruces "3+" no se reconocían como valor.
    - Primera medición: 50-55 % de los documentos leídos.
  - **PDF real:** la lectura local es idéntica a la de Gemini en los 8 analitos, incluidos los rangos con categorías, y da sexo y edad.
    - De punta a punta sin clave de Gemini en el proceso (el cliente ni se crea): **36 s** en GPU, frente a 52 s con la lectura de Gemini.
  - Limitación: los 3 diseños sintéticos son propios. Un diseño real muy distinto podría leerse mal sin que el lector lo detecte; de ahí el aviso y el botón para releer.
  - `tests/test_lector_pdf.py` (13 pruebas). **146 passed.**
- [x] **Segundo PDF real (Colsanitas 2011, química: LDL, triglicéridos, ALT y AST), 2026-11-02:**
  - **No se leía en local y la app pasó a Gemini**, como estaba previsto. Causa: el sistema del laboratorio escribe los fragmentos fuera del orden visual, y el modo layout de pypdf los pegaba ("mg/dl102.37", "U/l 0.00 31.0028.70").
  - Corrección: si el modo layout no da una lectura confiable, `texto_por_posicion` rearma las filas con la posición (x, y) de cada fragmento.
    - Probada sola con los 450 PDFs sintéticos: 100 %.
    - Formatos nuevos: bandera antes del valor ("* 35.60"), unidad y rango en la misma celda, rango como mínimo y máximo sin guion, y la edad separada de su número ("Edad · Años 72").
    - La nota "VALORES DE REFERENCIA SEGÚN…" ya no corta las categorías de debajo.
  - **Resultado:** 4 de 4 analitos y estados correctos (LDL, triglicéridos y AST altos; ALT normal).
    - Las categorías completas ("Óptimo: Menor de 100 · Casi óptimo: …"), donde Gemini solo había dado "Menor de 100".
    - **El sexo (mujer, 72 años), que Gemini no detectó.**
  - **Preguntas:** el filtro de repetidas quitaba "¿…mis niveles de AST?" y "¿…de ALT?" por parecidas, y la ficha quedó con una sola pregunta. Ahora dos preguntas solo son repetidas si nombran los mismos exámenes. Se agregaron los alias ALAT, ASAT, "Alanino amino transferasa" y "Aspartato amino transferasa".
  - La consola dice por qué no se pudo leer un PDF en local (sin texto, diseño desconocido o la línea dudosa).
  - Calidad que queda para el generador v2: repitió la misma frase de causas ("dieta alta en grasas o azúcares…") en LDL, triglicéridos y AST.
  - 6 pruebas nuevas, con los dos PDFs reales como regresión. **152 passed.**
- [ ] Comparar contra el MODO gemini en los mismos exámenes (rúbrica humana).

## Fase 6: informe 🟡

- [x] **Figuras** (`informe/figuras.py` → `informe/figuras/`, 2026-11-02): RAG (hit@1 total y consultas manuales), guardrail (sintético frente a texto real, con las riesgosas que detecta cada modelo), generador base frente a ajustado, curvas de aprendizaje (QLoRA y BiLSTM) y velocidad. Se generan desde los JSON de `evaluacion/resultados/`; las curvas se copian a `evaluacion/resultados/curvas.json` para no depender de `models/`. Paleta validada para daltonismo y valores escritos en cada barra.
- [x] **README** reescrito: qué hace, alcance (solo sangre), arquitectura con diagrama, tabla de resultados, figuras, instalación en CPU y GPU, estructura y limitaciones.
- [ ] Revisión final del equipo: leer el README y `fundamentos.md` completos y ensayar la demo.
- [x] `informe/fundamentos.md`: relaciona cada principio del curso con un resultado medido del proyecto. Se revisó contra el PDF del curso (*PTIA Principios NN*) y contra el código el 2026-10-18.

### Cobertura de los temas del curso

Fuente: *PTIA Principios NN*, 93 diapositivas. **Regla:** cada fase nueva agrega su vínculo en `informe/fundamentos.md`, con un dato medido, antes de marcarse como terminada.

| Tema del curso (diapositivas) | Dónde aparece en el proyecto | Sección de `fundamentos.md` |
|---|---|---|
| Asociacionismo, 4 leyes de Aristóteles (4-6) | Ajuste contrastivo del RAG | 1 |
| Conexionismo: Hartley, Bain (7-11) | El conocimiento está en las conexiones (pesos) | 1 |
| McCulloch y Pitts, sinapsis excitadora e inhibidora (12-13) | Filtro de palabras original | 2 |
| Hebb y su inestabilidad (14-15) | Softmax, coseno acotado y weight decay | 1 |
| Perceptrón, sesgo como peso, convergencia y separabilidad (16-41) | TF-IDF + regresión logística | 3 |
| Funciones de activación: escalón, signo, sigmoide, tanh, ReLU (18, 42-44) | LSTM, cabezas, MrBERT | 4 |
| Red neuronal y conteo de neuronas, pesos y parámetros (45-46) | Red propia BiLSTM | 5 |
| Poder de representación, Cybenko, sobreajuste (47) | Red propia frente a MrBERT | 6 |
| Función de pérdida: MSE, MAE, L1, L2, cero-uno, hinge (52-57) | Entropía cruzada para entrenar, F1 para evaluar | 7 |
| Gradiente, descenso, tasa de aprendizaje, mínimos locales (58-68) | AdamW, recorte, 3 semillas | 8 |
| Retropropagación y regla de la cadena, XOR (69-91) | GradCache, retropropagación a través del tiempo | 9 |
| Curva de aprendizaje (92) | Tabla y gráfica de épocas de la red propia y del QLoRA (`fig4_curvas.png`) | 6 y 10 |
| Historia: aprendizaje profundo = algoritmos + cómputo + datos (3) | Preentrenamiento frente a red desde cero | 6 |

**Pendiente teórico:**
- Agregar weight decay al ajuste del RAG. Hoy vale 0, así que el problema de Hebb "sin reducción de pesos" queda sin resolver ahí.
- [x] ~~La pérdida ponderada por clase en guardrails (sección 7)~~: hecha; no mejora (ver 7.1). Queda la calibración (temperature scaling).
- [x] ~~Las curvas de aprendizaje como gráfica~~: hecha (`informe/figuras/fig4_curvas.png`).

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

# Pruebas (esperado: 159 passed)
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
