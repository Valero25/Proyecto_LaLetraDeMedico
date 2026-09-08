import json
import os
from google import genai
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# La nueva librería google-genai lee automáticamente GEMINI_API_KEY de las variables de entorno,
# por lo que no es necesario configurarlo manualmente si ya está en .env
if "GEMINI_API_KEY" not in os.environ:
    print("Advertencia: No se encontró GEMINI_API_KEY en las variables de entorno.")

class LetraDeMedicoBackend:
    def __init__(self, model_name="gemini-flash-latest"):
        # Usamos un modelo potente para el razonamiento médico y la extracción
        self.client = genai.Client()
        self.model_name = model_name

    def step_1_ocr(self, file_path):
        """
        Paso 1: OCR (Optical Character Recognition)
        En un entorno real, aquí usaríamos Tesseract, Google Cloud Vision o pdfplumber.
        Para el prototipo, si pasamos la imagen a Gemini, él mismo puede hacer el OCR.
        Por ahora, simularemos la extracción de texto.
        """
        print(f"--- Ejecutando OCR en {file_path} ---")
        # Aquí iría el código real de lectura de PDF o Imagen.
        texto_extraido = """
        LABORATORIO CLÍNICO
        Paciente: Juan Pérez
        Glucosa en suero: 110 mg/dL (Referencia: 70 - 99 mg/dL)
        Hemoglobina A1c: 5.8 % (Referencia: 4.0 - 5.6 %)
        Colesterol Total: 180 mg/dL (Referencia: < 200 mg/dL)
        """
        return texto_extraido

    def step_2_ner(self, ocr_text):
        """
        Paso 2: NER (Named Entity Recognition)
        Extrae los valores estructurados del texto crudo (examen, valor, unidad, rango, estado).
        Usamos el LLM para forzar una salida estructurada en JSON.
        """
        print("--- Ejecutando NER (Extracción de Entidades) ---")
        
        prompt = f"""
        Eres un experto en extracción de datos médicos.
        Analiza el siguiente texto de resultados de laboratorio y extrae la información en formato JSON.
        El JSON debe ser una lista de objetos, donde cada objeto represente un examen con las siguientes claves:
        - "examen": Nombre del examen.
        - "valor": El valor numérico reportado.
        - "unidad": La unidad de medida.
        - "rango_referencia": El rango normal indicado en el examen.
        - "estado": Clasifícalo estrictamente como "NORMAL", "ALTO", "BAJO" o "LIMITE" según el rango de referencia.

        Texto OCR:
        {ocr_text}
        
        Responde ÚNICAMENTE con el JSON, sin texto adicional (ni siquiera bloques de código ```json).
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        
        try:
            # Limpiamos la respuesta en caso de que el modelo devuelva los backticks de markdown
            json_str = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            entidades = json.loads(json_str)
            return entidades
        except json.JSONDecodeError:
            print("Error al decodificar el JSON del NER.")
            return []

    def step_3_generative_llm(self, entidades, patient_context=""):
        """
        Paso 3: LLM Generativo (Traducción + Preguntas)
        Toma las entidades estructuradas y el contexto del paciente para generar la explicación
        en lenguaje claro y las preguntas para el médico.
        """
        print("--- Ejecutando LLM Generativo (Traducción a lenguaje claro) ---")
        
        # Convertimos las entidades a un string legible para el prompt
        datos_laboratorio = json.dumps(entidades, indent=2, ensure_ascii=False)
        
        prompt = f"""
        Eres "LetraDeMédico", un asistente de IA empático y claro diseñado para pacientes.
        Tu objetivo es explicar resultados de laboratorio en un lenguaje sencillo, cotidiano y sin jerga médica.
        
        REGLA DE ORO: NO ESTÁS DIAGNOSTICANDO. Tu trabajo es informar y preparar al paciente para su consulta médica.
        Siempre debes recordar que esta información no reemplaza el criterio del médico.

        Datos del laboratorio extraídos (NER):
        {datos_laboratorio}

        Contexto del paciente (opcional):
        {patient_context if patient_context else "No proporcionado."}

        Por favor, genera tu respuesta en formato JSON con la siguiente estructura:
        {{
            "explicaciones": [
                {{
                    "examen": "Nombre del examen",
                    "explicacion": "Explicación sencilla de qué mide este examen y qué significa el resultado específico del paciente (normal, alto, etc.)."
                }}
            ],
            "preguntas_sugeridas": [
                "Pregunta 1 para el médico...",
                "Pregunta 2 para el médico..."
            ]
        }}
        
        Responde ÚNICAMENTE con el JSON válido.
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        
        try:
            json_str = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            resultado_final = json.loads(json_str)
            return resultado_final
        except json.JSONDecodeError:
            print("Error al decodificar el JSON del LLM Generativo.")
            return {}

    def procesar_resultado(self, file_path, patient_context=""):
        """
        Orquesta el pipeline completo: OCR -> NER -> LLM Generativo
        """
        texto_ocr = self.step_1_ocr(file_path)
        entidades = self.step_2_ner(texto_ocr)
        
        if not entidades:
            return {"error": "No se pudieron extraer entidades del texto."}
            
        resultado = self.step_3_generative_llm(entidades, patient_context)
        
        return {
            "datos_extraidos": entidades,
            "analisis_final": resultado
        }

# Ejemplo de uso (solo se ejecuta si se corre este archivo directamente)
if __name__ == "__main__":
    backend = LetraDeMedicoBackend()
    
    # Simulamos el contexto del paciente que ingresaría en la app
    contexto = "Últimamente me he sentido con mucha sed y voy mucho al baño."
    
    # Ejecutamos el pipeline completo
    resultado = backend.procesar_resultado("ruta/simulada/examen.pdf", contexto)
    
    print("\n--- RESULTADO FINAL DEL PIPELINE ---")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
