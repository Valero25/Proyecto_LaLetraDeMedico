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

**Fase en curso:** Fase 1 terminada. Siguiente: Fase 2, pendiente de las decisiones abiertas.

| Fase | Estado |
|---|---|
| 0. Preparación | ✅ Terminada |
| 1. Generador de exámenes sintéticos | ✅ Terminada |
| 2. RAG con ALIA-MrBERT | ⬜ Pendiente |
| 3. Clasificador de guardrails | ⬜ Pendiente |
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
- [ ] **Pendiente del equipo:** hacer commit de los cambios previos en `app.py` y `backend.py` (reintentos y modelo de respaldo de Gemini). No se tocaron.

## Fase 1: generador de exámenes sintéticos ✅

- [x] `training/sintetico/catalogo_analitos.json`: 42 analitos en 8 paneles, con alias reales (ej. TGO/AST, Hb), unidades, rangos por sexo y límites fisiológicos.
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

## Fase 2: RAG con ALIA-MrBERT ⬜

- [ ] Base de conocimiento: una ficha en español por cada analito del catálogo (qué mide, qué significa alto o bajo, causas comunes), con fuente citada (ej. MedlinePlus en español; revisar licencia).
- [ ] Fragmentar las fichas, generar embeddings con `SINAI/ALIA-MrBERT-es-biomedical-embeddings` y crear el índice FAISS.
- [ ] Set de unas 150 consultas con los alias que aparecen en exámenes reales ("TGO", "Col. HDL", "BUN").
- [ ] Medir Recall@k y MRR contra la búsqueda por subcadena actual y contra un modelo genérico (bge-m3 o multilingual-e5).
- [ ] Opcional: ajuste contrastivo con pares propios y comparación antes y después.

## Fase 3: clasificador de guardrails con MrBERT-biomed ⬜

- [ ] Dataset de unas 2.000 frases en tres clases (*informativa*, *diagnóstica*, *alarmista*), con etiquetado y revisión del equipo. Split 70/15/15.
- [ ] Notebook de fine-tuning para Colab (`BSC-LT/MrBERT-biomed`, alternativa `MrBERT-es`).
- [ ] Métricas: F1 por clase y matriz de confusión, contra la lista de palabras actual de `step_4_validation`.

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

## Decisiones abiertas

1. **Fecha de entrega:** define si entran la Fase 2 opcional y el NER propio.
2. **GPU para entrenar:** ¿Colab gratuito (T4), Colab Pro u otra?
3. **Modelo profesor** para generar datos de las Fases 3 y 4: ¿Groq, Together, HF Inference u otro? No usar Gemini, porque sus términos prohíben usar sus salidas para entrenar modelos que compitan.

---

## Cómo verificar

Desde la raíz del proyecto:

```bash
pip install -r requirements-train.txt

# Pruebas (esperado: 11 passed, 1 xfailed)
python -m pytest tests -v

# Regenerar el dataset sintético
python training/sintetico/generar_examenes.py --n 2000 --seed 42

# Medir la regla determinista del backend
python evaluacion/evaluar_reglas.py
```

El `xfail` es intencional: documenta la brecha de formatos de la regla, que se corrige en la Fase 5.

---

## Bitácora

- **2026-10-18:**
  - Feedback del profesor: el proyecto era "usar un LLM y ya"; hay que hacer fine-tuning y participar en la construcción del modelo.
  - Revisión de modelos: Med-BERT descartado (la rama `version_2` agrega medicamentos y procedimientos, no idiomas, y trabaja con códigos ICD). MrBERT-biomed elegido para guardrails y ALIA-MrBERT para RAG.
  - Fases 0 y 1 completadas.
