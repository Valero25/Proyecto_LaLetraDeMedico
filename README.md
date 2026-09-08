# 🩺 LetraDeMédico

> **Traductor inteligente de resultados médicos para pacientes.**

![Estado](https://img.shields.io/badge/Estado-Prototipo_Activo-brightgreen)
![Versión](https://img.shields.io/badge/Versi%C3%B3n-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)

En muchos sistemas de salud, las consultas médicas tienen una duración limitada (a veces de solo 15 minutos). Esto dificulta que los pacientes comprendan completamente su diagnóstico o los resultados de sus exámenes de laboratorio.

**LetraDeMédico** es una herramienta de Inteligencia Artificial capaz de traducir resultados de laboratorio e información médica a un lenguaje claro, coloquial y comprensible. Su objetivo es empoderar al paciente, reducir su ansiedad y prepararlo con preguntas relevantes para su próxima consulta.

⚠️ *El objetivo NO es diagnosticar ni reemplazar al profesional de la salud.*

---

## ✨ Características Principales

- **Lectura de Exámenes (Simulada/OCR):** Capacidad de procesar texto extraído de PDFs o imágenes de resultados de laboratorio.
- **Traducción a Lenguaje Claro:** Explica si un valor es Alto, Bajo o Normal de forma sencilla.
- **Generación de Preguntas:** Prepara automáticamente una lista de preguntas sugeridas para que el paciente hable con su médico.
- **Interfaz Intuitiva:** Desarrollado con Streamlit para ofrecer una experiencia limpia y fácil de usar, inspirada en diseño centrado en el usuario.

## 🏗️ Arquitectura del Sistema

El flujo de trabajo (Pipeline) de la Inteligencia Artificial consta de tres pasos principales:

1. **OCR (Optical Character Recognition):** Extracción del texto en bruto del examen médico.
2. **NER (Named Entity Recognition):** Un modelo de IA extrae datos estructurados (Nombre del examen, valor, unidad, rango, estado) en formato JSON estricto.
3. **LLM Generativo:** Traduce el JSON a lenguaje natural considerando el contexto reportado por el paciente.

*Próximamente:* **RAG (Retrieval-Augmented Generation)** para anclar las respuestas a bases de datos y diccionarios médicos oficiales, evitando "alucinaciones" de la IA.

---

## 🚀 Instalación y Uso

Sigue estos pasos para correr el proyecto en tu entorno local.

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/Proyecto_LaLetraDeMedico.git
cd Proyecto_LaLetraDeMedico
```

### 2. Instalar dependencias
Asegúrate de tener Python instalado. Luego ejecuta:
```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno
Crea un archivo llamado `.env` en la raíz del proyecto y agrega tu llave de la API de Google Gemini:
```env
GEMINI_API_KEY=tu_clave_api_aqui
```

### 4. Ejecutar la aplicación web
Arranca el servidor local de Streamlit:
```bash
streamlit run app.py
```
Abre tu navegador en `http://localhost:8501`.

---

## 👨‍💻 Autor

- **Rubiano Valero** - Idea original y desarrollo inicial.

---

*“Cerrando la brecha de comunicación entre el resultado técnico y la persona que necesita entenderlo.”*