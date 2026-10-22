import os
import streamlit as st
import time
from backend import LetraDeMedicoBackend
import modelos_propios

# Configuración de la página
st.set_page_config(page_title="LetraDeMédico", page_icon="🩺", layout="wide")

MOTORES = {
    "propio": "Modelos propios (entrenados por el equipo)",
    "gemini": "Gemini (sistema original)",
}


@st.cache_resource(show_spinner=False)
def obtener_backend(modo):
    """Un backend por modo; en modo propio carga los tres modelos una sola vez."""
    backend = LetraDeMedicoBackend(modo=modo)
    if modo == "propio":
        for nombre in ("rag", "guardrail", "generador"):
            backend.modelo_propio(nombre)
    return backend


# --- SELECTOR DE MOTOR ---
faltan = modelos_propios.faltantes()
opciones = [m for m in MOTORES if m != "propio" or not faltan]
modo_inicial = os.environ.get("MODO", "propio")
with st.sidebar:
    st.header("Motor de IA")
    modo = st.radio("Elige quién explica tus resultados", opciones, format_func=MOTORES.get,
                    index=opciones.index(modo_inicial) if modo_inicial in opciones else 0)
    if faltan:
        st.warning("Modelos propios no disponibles. Faltan: " + ", ".join(faltan) + ". Ver PLAN.md, Fase 5.")
    st.caption("En ambos modos, la lectura del documento (OCR) la hace Gemini. El modo propio usa el RAG, "
               "el generador y el guardrail entrenados en las Fases 2 a 4.")

with st.spinner("Cargando los modelos propios (solo la primera vez)..." if modo == "propio" else "Preparando..."):
    backend = obtener_backend(modo)

# --- DATOS DEL MODO DEMOSTRACIÓN ---
DEMO_ANALITOS = [
    {"examen": "Hemoglobina", "valor": "14.2", "unidad": "g/dL", "rango_referencia": "13.0 - 17.0"},
    {"examen": "Leucocitos", "valor": "7.20", "unidad": "10³/µL", "rango_referencia": "4.0 - 11.0"},
    {"examen": "Glucosa", "valor": "108", "unidad": "mg/dL", "rango_referencia": "70 - 99"},
    {"examen": "Colesterol total", "valor": "178", "unidad": "mg/dL", "rango_referencia": "< 200"},
    {"examen": "Colesterol HDL", "valor": "36", "unidad": "mg/dL", "rango_referencia": "> 40"},
    {"examen": "Triglicéridos", "valor": "118", "unidad": "mg/dL", "rango_referencia": "< 150"},
]
DEMO_EXPLICACIONES = [
    {"examen": "Hemoglobina", "explicacion": "La hemoglobina es la proteína que lleva el oxígeno a todo el cuerpo. Un valor de 14.2 g/dL está dentro del rango de referencia (13.0 - 17.0), lo que indica un transporte de oxígeno adecuado."},
    {"examen": "Leucocitos", "explicacion": "Los leucocitos son los glóbulos blancos, las defensas del organismo. El valor de 7.20 está dentro del rango de referencia (4.0 - 11.0)."},
    {"examen": "Glucosa", "explicacion": "La glucosa es el azúcar que circula en la sangre y sirve de energía. El valor de 108 mg/dL está un poco por encima del rango de referencia en ayunas (70 - 99). Vale la pena comentarlo con el médico para saber si conviene repetir la prueba."},
    {"examen": "Colesterol total", "explicacion": "El colesterol es una grasa que el cuerpo necesita en cierta cantidad. El valor de 178 mg/dL está por debajo de 200, que es el nivel deseable."},
    {"examen": "Colesterol HDL", "explicacion": "El HDL se conoce como el colesterol 'bueno' porque ayuda a retirar grasa de las arterias. El valor de 36 mg/dL está por debajo del nivel recomendado (más de 40). El médico puede orientar sobre hábitos que ayudan a subirlo."},
    {"examen": "Triglicéridos", "explicacion": "Los triglicéridos son otro tipo de grasa en la sangre. El valor de 118 mg/dL está dentro del nivel deseable (menos de 150)."},
]
DEMO_PREGUNTAS = [
    "Mi glucosa salió un poco alta, ¿debo repetir el examen o hacerme otra prueba?",
    "¿Qué puedo hacer para subir mi colesterol HDL?",
    "¿Estos resultados explican el cansancio que he sentido?",
    "¿Cada cuánto debería repetir estos exámenes?",
]

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .resultado-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .badge-normal { background-color: #d4edda; color: #155724; padding: 5px 10px; border-radius: 5px; font-weight: bold; float: right;}
    .badge-alto { background-color: #f8d7da; color: #721c24; padding: 5px 10px; border-radius: 5px; font-weight: bold; float: right;}
    .badge-bajo { background-color: #fff3cd; color: #856404; padding: 5px 10px; border-radius: 5px; font-weight: bold; float: right;}
    .badge-limite { background-color: #cce5ff; color: #004085; padding: 5px 10px; border-radius: 5px; font-weight: bold; float: right;}
    .disclaimer {
        border: 2px solid #ccc;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        font-weight: bold;
        color: #666;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CABECERA ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063206.png", width=80) # Logo genérico temporal
with col2:
    st.title("LetraDeMédico")
st.markdown("---")

# --- PANTALLA 1: SUBIDA DE ARCHIVO ---
if "analisis_completado" not in st.session_state:
    st.session_state.analisis_completado = False

if not st.session_state.analisis_completado:
    st.subheader("1. Sube tu resultado de Laboratorio")
    
    archivo_subido = st.file_uploader("Subir imagen o PDF", type=['pdf', 'png', 'jpg', 'jpeg'])
    
    contexto_paciente = st.text_area(
        "Cuéntanos brevemente tu situación (opcional)", 
        placeholder="Ej. 'El médico me mandó estos exámenes porque me sentía muy cansado últimamente.'"
    )
    
    if st.button("ANALIZAR RESULTADO", use_container_width=True):
        if archivo_subido is not None:
            # --- PANTALLA 2: SPINNER DE CARGA ---
            with st.spinner('Estamos leyendo tu resultado...'):
                # En un caso real, guardaríamos el archivo temporalmente
                # Para el prototipo, simularemos que pasamos la ruta
                time.sleep(2) # Simular tiempo de carga
                
                # Ejecutar el backend
                try:
                    file_bytes = archivo_subido.read()
                    mime_type = archivo_subido.type
                    resultado = backend.procesar_resultado(file_bytes, mime_type, contexto_paciente)
                    
                    if "error" in resultado:
                        st.error(f"Hubo un error analizando el documento: {resultado['error']}")
                    else:
                        st.session_state.datos_extraidos = resultado["datos_extraidos"]
                        st.session_state.analisis_final = resultado["analisis_final"]
                        st.session_state.tipo_examen = resultado.get("tipo_examen", "Examen Médico")
                        st.session_state.modo_usado = resultado.get("modo", modo)
                        st.session_state.pasajes_rag = resultado.get("pasajes_rag")
                        st.session_state.es_demo = False
                        st.session_state.analisis_completado = True
                        st.rerun()
                except Exception as e:
                    error_msg = str(e)
                    if "503" in error_msg or "high demand" in error_msg:
                        st.error("⚠️ Los servidores de IA de Google están experimentando mucha demanda en este momento. Por favor, intenta de nuevo en unos segundos.")
                    elif "PerDay" in error_msg:
                        st.error("⚠️ Se agotó la cuota diaria gratuita de la API de Gemini para este modelo. Intenta mañana o cambia de modelo con GEMINI_MODEL en el archivo .env.")
                    elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        st.error("⚠️ Se alcanzó el límite de solicitudes por minuto de la API de Gemini. Espera un minuto e intenta de nuevo.")
                    else:
                        st.error(f"⚠️ Ocurrió un error inesperado al conectar con la IA: {error_msg}")
        else:
            st.warning("Por favor, sube un archivo primero.")

    # Modo demostración: un examen de ejemplo, sin subir archivo ni usar el OCR de Gemini.
    # Con los modelos propios, el ejemplo pasa de verdad por el RAG, el generador y el guardrail.
    etiqueta_demo = ("Ver análisis de ejemplo con los modelos propios" if modo == "propio"
                     else "Ver análisis de ejemplo (demo, sin IA)")
    if st.button(etiqueta_demo, use_container_width=True):
        datos_demo = [dict(d) for d in DEMO_ANALITOS]
        if modo == "propio":
            with st.spinner("Los modelos propios están explicando el ejemplo (puede tardar alrededor de un minuto en CPU)..."):
                resultado = backend.analizar_entidades(datos_demo, "Hemograma y Perfil Lipídico", contexto_paciente)
            st.session_state.datos_extraidos = resultado["datos_extraidos"]
            st.session_state.analisis_final = resultado["analisis_final"]
            st.session_state.pasajes_rag = resultado["pasajes_rag"]
        else:
            for d in datos_demo:
                d["estado"] = backend.evaluar_estado_analito(d["valor"], d["rango_referencia"])
            st.session_state.datos_extraidos = datos_demo
            st.session_state.analisis_final = backend.step_4_validation({
                "explicaciones": [dict(e) for e in DEMO_EXPLICACIONES],
                "preguntas_sugeridas": list(DEMO_PREGUNTAS),
            })
            st.session_state.pasajes_rag = None
        st.session_state.tipo_examen = "Hemograma y Perfil Lipídico"
        st.session_state.modo_usado = modo
        st.session_state.es_demo = True
        st.session_state.analisis_completado = True
        st.rerun()

# --- PANTALLA 3: DASHBOARD DE RESULTADOS ---
else:
    if st.session_state.get("es_demo"):
        st.caption("Modo demostración: datos de ejemplo, no provienen de un examen real.")
    st.caption(f"Explicado por: {MOTORES.get(st.session_state.get('modo_usado', modo), '')}")

    col_izq, col_der = st.columns([1.2, 1])
    
    with col_izq:
        tipo_ex = st.session_state.get("tipo_examen", "Tus Resultados")
        st.subheader(f"📊 {tipo_ex}")
        
        # Iterar sobre las explicaciones generadas por la IA
        explicaciones = st.session_state.analisis_final.get("explicaciones", [])
        datos_ner = st.session_state.datos_extraidos
        
        for exp in explicaciones:
            # Buscar el dato crudo correspondiente para obtener el valor y estado
            dato_crudo = next((d for d in datos_ner if d["examen"].lower() == exp["examen"].lower()), None)
            
            estado = dato_crudo["estado"].upper() if dato_crudo else "NORMAL"
            valor = dato_crudo["valor"] if dato_crudo else ""
            unidad = dato_crudo["unidad"] if dato_crudo else ""
            
            # Asignar clase de color según el estado
            badge_class = "badge-normal"
            if estado == "ALTO" or estado == "ELEVADO": badge_class = "badge-alto"
            elif estado == "BAJO": badge_class = "badge-bajo"
            elif estado == "LIMITE": badge_class = "badge-limite"
            
            st.markdown(f"""
            <div class="resultado-box">
                <span class="{badge_class}">{estado}</span>
                <h4>Examen: {exp['examen']}</h4>
                <h5 style="color: #555;">{valor} {unidad}</h5>
                <p>{exp['explicacion']}</p>
            </div>
            """, unsafe_allow_html=True)
            # Guardrail propio: muestra exactamente qué oraciones revisar con el médico
            for oracion in exp.get("oraciones_marcadas", []):
                st.warning(f"Oración marcada por el guardrail (puede sonar a diagnóstico): «{oracion}»")

        pasajes = st.session_state.get("pasajes_rag")
        if pasajes:
            with st.expander("Fuentes consultadas (MedlinePlus)"):
                for p in pasajes:
                    fuente = p.get("fuente") or ""
                    enlace = f" — [{fuente}]({fuente})" if str(fuente).startswith("http") else ""
                    st.markdown(f"- **{p['examen']}**: {p['titulo']}{enlace}")
            
    with col_der:
        st.subheader("Preguntas para tu médico")
        
        preguntas = st.session_state.analisis_final.get("preguntas_sugeridas", [])
        for i, pregunta in enumerate(preguntas):
            st.info(f"**{i+1}.** {pregunta}")
            
        disclaimer_text = st.session_state.analisis_final.get("disclaimer", "⚠️ Esta información no reemplaza el diagnóstico de tu médico.")
            
        st.markdown(f"""
        <div class="disclaimer">
            {disclaimer_text}
        </div>
        """, unsafe_allow_html=True)
        
    if st.button("Analizar otro examen"):
        st.session_state.analisis_completado = False
        st.rerun()
