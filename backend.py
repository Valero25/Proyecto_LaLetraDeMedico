import json
import os
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# La nueva librería google-genai lee automáticamente GEMINI_API_KEY de las variables de entorno,
# por lo que no es necesario configurarlo manualmente si ya está en .env
if "GEMINI_API_KEY" not in os.environ:
    print("Advertencia: No se encontró GEMINI_API_KEY en las variables de entorno.")

class LetraDeMedicoBackend:
    def __init__(self, model_name="gemini-3.6-flash"):
        # Usamos un modelo potente para el razonamiento médico y la extracción
        self.client = genai.Client()
        self.model_name = model_name

    def step_1_and_2_multimodal_ner(self, file_bytes, mime_type):
        """
        Paso 1 y 2 Unificados: Extracción Multimodal Directa.
        Usamos las capacidades visuales del LLM para extraer la información directamente del documento 
        (PDF o Imagen) estructurándola en JSON, sin pasar por un OCR de texto plano.
        """
        print("--- Ejecutando Extracción Multimodal de Entidades ---")
        
        prompt = """
        Eres un experto en extracción de datos médicos.
        Analiza el documento de resultados de laboratorio adjunto y extrae la información en formato JSON.
        
        El JSON debe ser estrictamente un OBJETO con las siguientes dos claves:
        - "tipo_de_examen": Identifica el tipo general de examen (ej. "Análisis de Orina Completo", "Perfil Lipídico", "Hemograma", "Bioquímica Sanguínea", etc.).
        - "analitos": Una lista de objetos, donde cada objeto represente un examen individual con las claves:
            - "examen": Nombre del examen.
            - "valor": El valor numérico o cualitativo (ej. "Negativo", "1.015") reportado.
            - "unidad": La unidad de medida (si aplica, si no dejar vacío).
            - "rango_referencia": El rango normal indicado en el examen (ej. "70 - 99", "Negativo", "< 200").

        No deduzcas ni infieras nada, extrae los datos numéricos y cualitativos tal cual aparecen en el documento.
        Si hay tablas, lee cada fila cuidadosamente.
        
        Responde ÚNICAMENTE con el JSON válido, sin texto adicional (ni siquiera bloques de código ```json).
        """
        
        try:
            document_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    document_part,
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            # Limpiamos la respuesta (ya no debería tener backticks gracias al config, pero por si acaso)
            json_str = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            entidades = json.loads(json_str)
            return entidades
        except json.JSONDecodeError:
            print("Error al decodificar el JSON del Multimodal NER.")
            return []
        except Exception as e:
            print(f"Error procesando el documento: {e}")
            return []

    @staticmethod
    def evaluar_estado_analito(valor_str, rango_str):
        """
        Paso 2.5: Reglas deterministas para evaluar el estado.
        Evalúa si un valor numérico o cualitativo está dentro del rango.
        Retorna "NORMAL", "ALTO", "BAJO", "ANORMAL" o "DESCONOCIDO".
        """
        try:
            val_lower = str(valor_str).lower().strip()
            ref_lower = str(rango_str).lower().strip()
            
            # Evaluación cualitativa (texto exacto)
            if val_lower and val_lower == ref_lower:
                return "NORMAL"
                
            cualitativos_sanos = ["negativo", "no reactivo", "ausente", "no se observan", "normal"]
            cualitativos_enfermos = ["positivo", "reactivo", "presente", "anormal"]
            
            # Si el paciente tiene negativo y lo normal era negativo
            if any(p in val_lower for p in cualitativos_sanos) and any(p in ref_lower for p in cualitativos_sanos):
                return "NORMAL"
            
            # Si el paciente tiene positivo pero lo normal era negativo
            if any(p in val_lower for p in cualitativos_enfermos) and any(p in ref_lower for p in cualitativos_sanos):
                return "ANORMAL"

            # Extraer el valor numérico del paciente
            valor_match = re.search(r'[\d\.]+', str(valor_str))
            if not valor_match:
                return "DESCONOCIDO"
            valor = float(valor_match.group())

            # Limpiar el string del rango numérico
            rango_str = str(rango_str).replace(' ', '').replace(',', '.')
            
            # Caso 1: Rango con guion ej. "70-99" o "4.0-5.6"
            rango_match = re.search(r'([\d\.]+)-([\d\.]+)', rango_str)
            if rango_match:
                min_val = float(rango_match.group(1))
                max_val = float(rango_match.group(2))
                if valor < min_val: return "BAJO"
                if valor > max_val: return "ALTO"
                return "NORMAL"
                
            # Caso 2: Menor que ej. "<200" o "<=200"
            menor_match = re.search(r'<[=]?([\d\.]+)', rango_str)
            if menor_match:
                max_val = float(menor_match.group(1))
                if valor >= max_val: return "ALTO"
                return "NORMAL"
                
            # Caso 3: Mayor que ej. ">40" o ">=40"
            mayor_match = re.search(r'>[=]?([\d\.]+)', rango_str)
            if mayor_match:
                min_val = float(mayor_match.group(1))
                if valor <= min_val: return "BAJO"
                return "NORMAL"
                
            return "DESCONOCIDO"
        except Exception as e:
            print(f"Error evaluando rango matemáticamente: {e}")
            return "DESCONOCIDO"

    def recuperar_conocimiento_rag(self, entidades):
        """
        Paso 2.8: Recuperación RAG.
        Busca información médica curada para los analitos, especialmente los que están fuera de rango.
        """
        print("--- Ejecutando Recuperación RAG ---")
        pasajes_recuperados = {}
        kb_path = os.path.join(os.path.dirname(__file__), "mock_medical_kb.json")
        
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                kb = json.load(f)
                
            for entidad in entidades:
                nombre = entidad.get("examen", "").lower()
                # Buscar coincidencias parciales simples (en un RAG real sería búsqueda semántica/vectorial)
                for key_kb, texto_kb in kb.items():
                    if key_kb in nombre or nombre in key_kb:
                        pasajes_recuperados[entidad["examen"]] = texto_kb
                        break
        except FileNotFoundError:
            print("No se encontró la base de conocimientos RAG.")
            
        return pasajes_recuperados

    def step_3_generative_llm(self, entidades, patient_context="", pasajes_rag=None, tipo_examen="Examen de Laboratorio"):
        """
        Paso 3: LLM Generativo (Traducción + Preguntas)
        Toma las entidades estructuradas y el contexto del paciente para generar la explicación
        en lenguaje claro y las preguntas para el médico.
        """
        print("--- Ejecutando LLM Generativo (Traducción a lenguaje claro) ---")
        
        # Convertimos las entidades a un string legible para el prompt
        datos_laboratorio = json.dumps(entidades, indent=2, ensure_ascii=False)
        pasajes_str = json.dumps(pasajes_rag or {}, indent=2, ensure_ascii=False)
        
        prompt = f"""
        Eres "LetraDeMédico", un asistente de IA experto, empático y diseñado para pacientes.
        Tu objetivo es explicar resultados de laboratorio usando un lenguaje técnico apropiado, pero acompañándolo siempre de una explicación personal, cálida y accesible para que cualquier persona sin estudios médicos pueda entenderlo perfectamente.
        
        REGLA DE ORO: NO ESTÁS DIAGNOSTICANDO. Tu trabajo es informar y preparar al paciente para su consulta médica.
        
        TIPO DE EXAMEN IDENTIFICADO: {tipo_examen}
        Adapta tu lenguaje al tipo de examen (si es orina, no hables de sangre, etc.).
        
        1. Explica basándote PRINCIPALMENTE en los 'Pasajes Médicos (RAG)' provistos. Mantén el rigor técnico, pero explícalo como si se lo estuvieras contando a un amigo.
        2. Los estados (NORMAL, ALTO, BAJO, ANORMAL) ya fueron calculados de forma determinista, no los cambies.

        Datos del laboratorio evaluados:
        {datos_laboratorio}

        Pasajes Médicos (RAG) de guías oficiales:
        {pasajes_str}

        Contexto del paciente (opcional):
        {patient_context if patient_context else "No proporcionado."}

        Por favor, genera tu respuesta en formato JSON con la siguiente estructura:
        {{
            "explicaciones": [
                {{
                    "examen": "Nombre del examen",
                    "explicacion": "Explicación sencilla basada en el RAG provisto, y qué significa el resultado para este paciente."
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
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        try:
            json_str = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            resultado_final = json.loads(json_str)
            return resultado_final
        except json.JSONDecodeError:
            print("Error al decodificar el JSON del LLM Generativo.")
            return {}

    def step_4_validation(self, resultado_final):
        """
        Paso 4: Guardrails y Validación Médica.
        Asegura que no se realicen diagnósticos definitivos e inyecta un disclaimer obligatorio.
        """
        print("--- Ejecutando Capa de Validación y Guardrails ---")
        
        disclaimer = "⚠️ IMPORTANTE: Esta información es generada por IA para propósitos informativos y NO constituye un diagnóstico médico. Consulta siempre a tu médico."
        
        for exp in resultado_final.get("explicaciones", []):
            texto = exp.get("explicacion", "")
            # Filtro básico de palabras que suenan a diagnóstico
            palabras_riesgo = ["diagnóstico", "tienes", "padeces", "sufres de"]
            if any(palabra in texto.lower() for palabra in palabras_riesgo):
                exp["explicacion"] = texto + " (Nota del sistema: Este texto fue marcado preventivamente. Recuerda que solo un médico puede diagnosticar)."
                
        # Asegurarnos de que el JSON siempre devuelva un campo de disclaimer
        resultado_final["disclaimer"] = disclaimer
        return resultado_final

    def procesar_resultado(self, file_bytes, mime_type, patient_context=""):
        """
        Orquesta el pipeline completo: Extracción Multimodal -> Reglas -> RAG -> LLM Generativo
        """
        resultado_multimodal = self.step_1_and_2_multimodal_ner(file_bytes, mime_type)
        
        if not resultado_multimodal or "analitos" not in resultado_multimodal:
            return {"error": "No se pudieron extraer entidades del documento."}
            
        tipo_examen = resultado_multimodal.get("tipo_de_examen", "Examen Médico")
        entidades = resultado_multimodal.get("analitos", [])
            
        # FASE 1: Evaluar los estados de forma determinista por reglas
        for entidad in entidades:
            estado_calculado = self.evaluar_estado_analito(
                entidad.get("valor", ""), 
                entidad.get("rango_referencia", "")
            )
            entidad["estado"] = estado_calculado
            
        # FASE 3: Recuperación RAG
        pasajes_rag = self.recuperar_conocimiento_rag(entidades)
            
        resultado = self.step_3_generative_llm(entidades, patient_context, pasajes_rag, tipo_examen)
        
        # FASE 2: Capa de Validación y Guardrails
        resultado = self.step_4_validation(resultado)
        
        return {
            "datos_extraidos": entidades, # Devolvemos solo la lista para no romper el frontend
            "analisis_final": resultado,
            "tipo_examen": tipo_examen
        }


