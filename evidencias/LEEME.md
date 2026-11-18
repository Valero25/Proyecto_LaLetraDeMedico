# Evidencias para el informe (recogidas el 2026-11-18)

| # | Evidencia | Archivos | Estado |
|---|---|---|---|
| 1 | Capturas de la app con un examen real cargado (RF1 a RF8): carga, resumen con estados, explicaciones, preguntas sugeridas, ficha, aviso de seguridad y chat | `01a` a `01e_*.png` | Listo |
| 2 | Ficha PDF generada por la app (8 analitos, sin nombre ni documento del paciente), con el aviso «NO es un diagnóstico» (RF10) | `02_ficha_consulta.pdf`, `02_ficha_consulta_pag1.png`, `02_ficha_consulta_pag2.png` | Listo |
| 3 | Tiempo por examen en GPU y en CPU, con el equipo y el modelo usados | `03_tiempo_gpu.txt`, `03_tiempo_cpu.txt`, script `medir_tiempos.py` | Listo |
| 4 | Lectura local del PDF real de 8 analitos (2 páginas), con datos personales tapados | `04_pdf_real_anonimizado_pag1.png`, `04_pdf_real_anonimizado_pag2.png`, `04_lectura_local.txt` | Listo |
| extra | Las mismas evidencias 3 y 4 con el segundo PDF real (7 analitos, otro diseño de Compensar) | `7_analitos/` | Listo |
| 5 | Rúbrica humana aplicada | `evaluacion/rubrica/respuestas/` está vacía | **Falta**: no hay respuestas en el repo |
| 6 | Pesos de los modelos | ver tabla abajo | Listo (locales; no están publicados en Drive ni Hugging Face) |
| 7 | Comparación con el modo Gemini | — | **Falta**: confirmar si se hizo |
| 8 | Equipo de pruebas | ver abajo | Listo |

## 3. Tiempos (pipeline completo: lectura + reglas + RAG + generador + guardrail)

| | PDF de 8 analitos (el de las capturas) | PDF de 7 analitos (`7_analitos/`) |
|---|---|---|
| Por examen, GPU | **19,7 s** | 21,3 s |
| Por examen, CPU | **63,2 s** | 71,1 s |
| Carga de modelos, GPU / CPU (una vez por sesión) | 47,0 s / 37,9 s | 44,0 s / 71,6 s |

Una sola corrida por modo, el 2026-11-18, con la app abierta en paralelo (ocupaba ~3 GB de la VRAM).
El README dice 28 s / 85 s para el mismo PDF de 8 analitos (medido el 2026-11-02); hoy salió más rápido,
así que esas cifras quedan como cota superior conservadora.

## 4. Lectura local

PDF de 8 analitos: los 8 salen con nombre completo, valor, unidad y rango correctos (también la página 2).
PDF de 7 analitos: valores y unidades correctos; el nombre de la glucosa queda cortado («... DIFERENTE A», sin
«ORINA», porque en el PDF salta de línea) y la glicemia post 2 horas y la insulina 2 horas no traen rango porque
el propio PDF no lo trae.

## 6. Pesos usados por la app

| Modelo | Ruta | Tamaño |
|---|---|---|
| Generador v2 (Qwen2.5-3B + LoRA, GGUF Q4_K_M) | `models/generador/letrademedico-qwen2.5-3b-v2-q4_k_m.gguf` | 1,8 GB |
| Adaptador LoRA v2 | `models/generador/qwen2.5-3b-instruct-lora-v2/` | 126 MB |
| Guardrail MrBERT-biomed v2 | `models/guardrails/mrbert-biomed-v2/` | 1,2 GB |
| RAG ALIA-MrBERT ajustado | `models/rag/alia-ajustado-completo/` | 578 MB |
| Índice FAISS del RAG | `models/rag/indice-alia-ajustado/` | 872 KB |

## 8. Equipo de pruebas

- CPU: Intel Core i5-13420H (8 núcleos / 12 hilos)
- RAM: 15,7 GB
- GPU: NVIDIA GeForce RTX 4050 Laptop, 6 GB (driver 616.92)
- Windows 11 Home, Python 3.11.9
