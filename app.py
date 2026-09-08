import streamlit as st
import time
from backend import LetraDeMedicoBackend

# Configuración de la página
st.set_page_config(page_title="LetraDeMédico", page_icon="🩺", layout="wide")

# Inicializar el backend en la sesión de Streamlit para no recargarlo cada vez
if "backend" not in st.session_state:
    st.session_state.backend = LetraDeMedicoBackend()

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
                    resultado = st.session_state.backend.procesar_resultado("simulacion.pdf", contexto_paciente)
                    
                    if "error" in resultado:
                        st.error(f"Hubo un error analizando el documento: {resultado['error']}")
                    else:
                        st.session_state.datos_extraidos = resultado["datos_extraidos"]
                        st.session_state.analisis_final = resultado["analisis_final"]
                        st.session_state.analisis_completado = True
                        st.rerun()
                except Exception as e:
                    error_msg = str(e)
                    if "503" in error_msg or "high demand" in error_msg:
                        st.error("⚠️ Los servidores de IA de Google están experimentando mucha demanda en este momento. Por favor, intenta de nuevo en unos segundos.")
                    else:
                        st.error(f"⚠️ Ocurrió un error inesperado al conectar con la IA: {error_msg}")
        else:
            st.warning("Por favor, sube un archivo primero.")

# --- PANTALLA 3: DASHBOARD DE RESULTADOS ---
else:
    col_izq, col_der = st.columns([1.2, 1])
    
    with col_izq:
        st.subheader("Tus resultados explicados")
        
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
            
    with col_der:
        st.subheader("Preguntas para tu médico")
        
        preguntas = st.session_state.analisis_final.get("preguntas_sugeridas", [])
        for i, pregunta in enumerate(preguntas):
            st.info(f"**{i+1}.** {pregunta}")
            
        st.markdown("""
        <div class="disclaimer">
            ⚠️ Esta información no reemplaza el diagnóstico de tu médico.
        </div>
        """, unsafe_allow_html=True)
        
    if st.button("Analizar otro examen"):
        st.session_state.analisis_completado = False
        st.rerun()
