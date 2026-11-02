import html
import os
import re
import streamlit as st
from backend import NOTA_GUARDRAIL, LetraDeMedicoBackend, alinear_explicaciones
import ficha_consulta
import modelos_propios
from ui_components import (
    ESTILOS_CSS,
    PRESETS,
    calcular_gauge_html,
    render_tarjeta_html,
    generar_ficha_medica_txt,
)

# 1. Configuración de página
st.set_page_config(
    page_title="LetraDeMédico - Traductor de Exámenes de Laboratorio",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inyección de estilos CSS modernos
st.html(ESTILOS_CSS)

FORMATOS_FICHA = {
    "PDF": ("pdf", "application/pdf", "Para imprimir y llevar a la cita, con casillas y espacio para notas."),
    "Imagen": ("png", "image/png", "Una imagen vertical para guardar o compartir desde el celular."),
    "Texto": ("txt", "text/plain", "Texto plano, para copiar y pegar."),
}

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


# --- SELECTOR DE MOTOR EN BARRA LATERAL ---
faltan = modelos_propios.faltantes()
opciones = [m for m in MOTORES if m != "propio" or not faltan]
modo_inicial = os.environ.get("MODO", "propio")

with st.sidebar:
    st.markdown("### ⚙️ Motor de IA")
    modo = st.radio(
        "Elige quién explica tus resultados:",
        opciones,
        format_func=MOTORES.get,
        index=opciones.index(modo_inicial) if modo_inicial in opciones else 0,
        help="El modo propio usa modelos locales (RAG, Qwen GGUF y Guardrail). El modo Gemini usa la API en la nube.",
    )
    if faltan:
        st.warning("Modelos propios no disponibles. Faltan: " + ", ".join(faltan) + ". Ver PLAN.md, Fase 5.")
    
    st.caption(
        "En ambos modos, la lectura del documento (OCR + NER) la hace Gemini. "
        "El modo propio usa el RAG, el generador y el guardrail entrenados localmente."
    )

    st.markdown("---")
    st.markdown("### 💡 Guía rápida para exámenes")
    st.markdown(
        """
        - **Ayuno correcto:** Las pruebas de glucosa y lípidos requieren de 8 a 12 horas de ayuno.
        - **Hidratación:** Beber agua facilita la extracción sanguínea y la muestra de orina.
        - **Medicamentos:** Informa siempre a tu médico si tomaste medicamentos antes de la toma.
        """
    )
    st.markdown("---")
    st.markdown("### 🛡️ Compromiso Ético")
    st.caption(
        "LetraDeMédico utiliza filtros y guardrails clínicos preventivos. "
        "La herramienta busca empoderarte para tu consulta médica, **sin sustituir el criterio de tu médico**."
    )

with st.spinner("Cargando los modelos propios (solo la primera vez)..." if modo == "propio" else "Preparando..."):
    backend = obtener_backend(modo)
if modo == "propio":
    dispositivo = getattr(backend.modelo_propio("generador"), "dispositivo", "CPU")
    st.sidebar.caption(f"⚡ Generador propio en **{dispositivo}**"
                       + ("" if dispositivo == "GPU" else " (con GPU NVIDIA es ~5 veces más rápido: ver requirements-gpu.txt)"))

# --- INICIALIZACIÓN DE ESTADO DE SESIÓN ---
if "analisis_completado" not in st.session_state:
    st.session_state.analisis_completado = False
if "preguntas_seleccionadas" not in st.session_state:
    st.session_state.preguntas_seleccionadas = []
if "preguntas_personalizadas" not in st.session_state:
    st.session_state.preguntas_personalizadas = []
if "chat_dudas" not in st.session_state:
    st.session_state.chat_dudas = []
if "ultimo_archivo_bytes" not in st.session_state:
    st.session_state.ultimo_archivo_bytes = None
if "ultimo_archivo_mime" not in st.session_state:
    st.session_state.ultimo_archivo_mime = None
if "ultimo_archivo_nombre" not in st.session_state:
    st.session_state.ultimo_archivo_nombre = None

# --- ENCABEZADO HERO PRINCIPAL ---
st.html("""
<div class="ldm-hero">
    <div class="ldm-hero-icon">🩺</div>
    <div class="ldm-hero-content">
        <span class="ldm-hero-tag">Inteligencia Artificial Médica Accesible</span>
        <h1>LetraDeMédico</h1>
        <p>Traducimos los resultados técnicos de tus análisis clínicos a un lenguaje humano, claro y sin tecnicismos confusos, preparándote para tu próxima consulta médica.</p>
    </div>
</div>
""")

# --- STEPPER DE PROGRESO ---
paso_actual = 3 if st.session_state.analisis_completado else 1
st.html(f"""
<div class="ldm-stepper">
    <div class="ldm-step {'active' if paso_actual >= 1 else ''}">
        <span class="ldm-step-num">1</span>
        <span>Subir o Seleccionar Examen</span>
    </div>
    <div class="ldm-step {'active' if paso_actual >= 2 else ''}">
        <span class="ldm-step-num">2</span>
        <span>Lectura & RAG Clínico</span>
    </div>
    <div class="ldm-step {'active' if paso_actual >= 3 else ''}">
        <span class="ldm-step-num">3</span>
        <span>Resultados & Guía Médica</span>
    </div>
</div>
""")


# =========================================================================
# PANTALLA 1: SUBIDA DE ARCHIVO O PRESETS INTERACTIVOS
# =========================================================================
if not st.session_state.analisis_completado:
    tab_subir, tab_presets = st.tabs([
        "📤 Subir mi resultado de laboratorio",
        "🧪 Probar ejemplos interactivos (sin archivo)"
    ])

    # --- PESTAÑA 1: SUBIR ARCHIVO PROPIO ---
    with tab_subir:
        st.subheader("1. Sube tu resultado de Laboratorio")
        st.markdown(
            "Carga una fotografía clara o archivo PDF de tu orden o informe de laboratorio. "
            "Nuestra IA extraerá automáticamente cada analito con su valor y rango."
        )

        col_up_izq, col_up_der = st.columns([1.2, 1])

        with col_up_izq:
            archivo_subido = st.file_uploader(
                "Selecciona un archivo PDF o imagen (JPG, PNG, JPEG)",
                type=['pdf', 'png', 'jpg', 'jpeg'],
                help="Asegúrate de que los nombres de los exámenes y los números sean legibles."
            )

            contexto_paciente = st.text_area(
                "Cuéntanos brevemente tu situación o motivo de consulta (opcional):",
                placeholder="Ej. 'El médico me solicitó estos exámenes porque he tenido cansancio y mareos en las mañanas...'",
                help="Este contexto ayuda a la IA a personalizar las explicaciones a tu caso particular.",
                height=110
            )

            btn_analizar = st.button("🚀 ANALIZAR MI RESULTADO", type="primary", use_container_width=True)

        with col_up_der:
            st.markdown("##### 👁️ Vista previa del archivo")
            if archivo_subido is not None:
                st.session_state.ultimo_archivo_bytes = archivo_subido.getvalue()
                st.session_state.ultimo_archivo_mime = archivo_subido.type
                st.session_state.ultimo_archivo_nombre = archivo_subido.name

                if archivo_subido.type.startswith("image/"):
                    st.image(archivo_subido, caption=f"Vista previa: {archivo_subido.name}", use_container_width=True)
                else:
                    st.info(f"📄 Archivo PDF cargado con éxito:\n\n**{archivo_subido.name}** ({archivo_subido.size / 1024:.1f} KB)")
            else:
                st.info("Sube una imagen o PDF a la izquierda para visualizarlo aquí antes de analizarlo.")

        if btn_analizar:
            if archivo_subido is not None:
                with st.spinner('Leyendo y analizando tu documento con inteligencia artificial...'):
                    try:
                        # getvalue() y no read(): la vista previa ya pudo mover el puntero del archivo al final
                        file_bytes = archivo_subido.getvalue()
                        mime_type = archivo_subido.type
                        resultado = backend.procesar_resultado(file_bytes, mime_type, contexto_paciente)

                        if "error" in resultado:
                            st.error(f"Hubo un problema analizando el documento: {resultado['error']}")
                        else:
                            st.session_state.datos_extraidos = resultado["datos_extraidos"]
                            st.session_state.analisis_final = resultado["analisis_final"]
                            st.session_state.tipo_examen = resultado.get("tipo_examen", "Examen Médico")
                            st.session_state.modo_usado = resultado.get("modo", modo)
                            st.session_state.pasajes_rag = resultado.get("pasajes_rag")
                            st.session_state.contexto_paciente = contexto_paciente
                            st.session_state.es_demo = False
                            st.session_state.preguntas_seleccionadas = list(resultado["analisis_final"].get("preguntas_sugeridas", []))
                            st.session_state.preguntas_personalizadas = []
                            st.session_state.chat_dudas = []
                            st.session_state.analisis_completado = True
                            st.rerun()
                    except Exception as e:
                        error_msg = str(e)
                        if "api key" in error_msg.lower() or "api_key" in error_msg.lower():
                            st.error("⚠️ Falta configurar GEMINI_API_KEY en el archivo .env: la lectura del documento la hace Gemini. "
                                     "Mientras tanto puedes probar los ejemplos interactivos.")
                        elif "503" in error_msg or "high demand" in error_msg:
                            st.error("⚠️ Los servidores de IA están experimentando alta demanda. Por favor, reintenta en unos instantes.")
                        elif "PerDay" in error_msg:
                            st.error("⚠️ Se agotó la cuota diaria gratuita de la API de Gemini. Puedes probar los modelos propios o los presets.")
                        elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                            st.error("⚠️ Límite de solicitudes por minuto alcanzado. Espera un momento y vuelve a presionar el botón.")
                        else:
                            st.error(f"⚠️ Ocurrió un error al procesar el examen: {error_msg}")
            else:
                st.warning("Por favor, selecciona primero un archivo de imagen o PDF.")

    # --- PESTAÑA 2: PRESETS CLÍNICOS INTERACTIVOS ---
    with tab_presets:
        st.subheader("🧪 Elige un caso de prueba interactivo")
        st.markdown(
            "Selecciona cualquiera de estos 3 perfiles clínicos reales para ver la interfaz en acción "
            "con explicaciones didácticas, medidores visuales de rango y preguntas para el médico."
        )

        col_p1, col_p2, col_p3 = st.columns(3)
        columnas_presets = [col_p1, col_p2, col_p3]

        for col, (clave, preset) in zip(columnas_presets, PRESETS.items()):
            with col:
                st.html(f"""
                <div class="ldm-preset-card">
                    <h4>{preset['nombre']}</h4>
                    <p style="font-size: 0.88rem; color: #64748b; margin-bottom: 8px;">{preset['descripcion']}</p>
                    <p style="font-size: 0.80rem; color: #0d9488; font-weight: 600; margin-bottom: 0;">
                        📋 {len(preset['analitos'])} analitos incluidos
                    </p>
                </div>
                """)

                # Mostrar chips de exámenes en expander
                with st.expander("Ver analitos incluidos"):
                    for a in preset["analitos"]:
                        st.markdown(f"- **{a['examen']}**: {a['valor']} {a['unidad']} *({a['rango_referencia']})*")

                etiqueta_btn = f"Explorar {preset['nombre'].split()[1]}"
                if st.button(f"🔍 {etiqueta_btn}", key=f"btn_preset_{clave}", use_container_width=True):
                    datos_demo = [dict(d) for d in preset["analitos"]]
                    
                    if modo == "propio":
                        try:
                            with st.spinner("Los modelos propios están analizando el ejemplo con RAG y Guardrails "
                                            "(en CPU tarda alrededor de un minuto y medio)..."):
                                resultado = backend.analizar_entidades(datos_demo, preset["tipo_examen"], preset["contexto_paciente"])
                        except Exception as e:
                            st.error(f"⚠️ Los modelos propios no pudieron analizar el ejemplo: {e}")
                            st.stop()
                        st.session_state.datos_extraidos = resultado["datos_extraidos"]
                        st.session_state.analisis_final = resultado["analisis_final"]
                        st.session_state.pasajes_rag = resultado["pasajes_rag"]
                    else:
                        # Modo Gemini con los datos curados del preset (explicaciones de muestra, sin llamar a la API)
                        for d in datos_demo:
                            d["estado"] = backend.evaluar_estado_analito(d["valor"], d["rango_referencia"])
                        st.session_state.datos_extraidos = datos_demo
                        st.session_state.analisis_final = backend.step_4_validation(alinear_explicaciones(datos_demo, {
                            "explicaciones": [dict(e) for e in preset["explicaciones"]],
                            "preguntas_sugeridas": list(preset["preguntas"]),
                        }))
                        st.session_state.analisis_final["muestra_escrita"] = True
                        st.session_state.pasajes_rag = None

                    st.session_state.tipo_examen = preset["tipo_examen"]
                    st.session_state.contexto_paciente = preset["contexto_paciente"]
                    st.session_state.modo_usado = modo
                    st.session_state.es_demo = True
                    st.session_state.preguntas_seleccionadas = list(st.session_state.analisis_final.get("preguntas_sugeridas", []))
                    st.session_state.preguntas_personalizadas = []
                    st.session_state.chat_dudas = []
                    st.session_state.analisis_completado = True
                    st.rerun()


# =========================================================================
# PANTALLA 2: DASHBOARD INTERACTIVO DE RESULTADOS
# =========================================================================
else:
    tipo_ex = st.session_state.get("tipo_examen", "Examen Médico")
    datos_ner = st.session_state.get("datos_extraidos", [])
    analisis = st.session_state.get("analisis_final", {})
    # Una explicación por analito y en el mismo orden (idempotente: el backend ya lo hace)
    explicaciones = alinear_explicaciones(datos_ner, analisis)["explicaciones"]
    modo_usado = st.session_state.get("modo_usado", modo)
    info_guardrail = analisis.get("guardrail") or {"motor": "palabras" if modo_usado == "gemini" else "mrbert-biomed-v2"}

    # --- BARRA SUPERIOR DE ACCIONES Y METADATOS ---
    col_nav1, col_nav2 = st.columns([3, 1])
    with col_nav1:
        st.markdown(f"### 📊 {tipo_ex}")
        fuente_motor = MOTORES.get(modo_usado, modo_usado)
        demo_tag = " • Modo Demostración interactiva" if st.session_state.get("es_demo") else " • Examen cargado por usuario"
        st.caption(f"🧠 Motor: **{fuente_motor}**{demo_tag}")
        if analisis.get("muestra_escrita"):
            st.caption("ℹ️ En este ejemplo, las explicaciones son una muestra escrita por el equipo: "
                       "no las generó Gemini. Elige el motor propio para ver una explicación generada en vivo.")
    with col_nav2:
        if st.button("🔄 Analizar otro examen", use_container_width=True, type="secondary"):
            st.session_state.analisis_completado = False
            st.session_state.chat_dudas = []
            st.session_state.preguntas_personalizadas = []
            st.rerun()

    # --- CÁLCULO DE MÉTRICAS KPI ---
    total_analitos = len(datos_ner)
    normales = sum(1 for d in datos_ner if d.get("estado", "").upper() == "NORMAL")
    alterados = sum(1 for d in datos_ner if d.get("estado", "").upper() in ("ALTO", "BAJO", "ELEVADO", "ANORMAL"))
    desconocidos = total_analitos - (normales + alterados)

    estado_resumen = "Valores estables" if alterados == 0 else f"{alterados} para comentar con tu médico"
    if desconocidos:
        estado_resumen += f" · {desconocidos} sin evaluar"

    # La tarjeta del guardrail dice lo que pasó de verdad, no un texto fijo
    marcadas = info_guardrail.get("marcadas", 0)
    if info_guardrail.get("error"):
        guard_valor, guard_color, guard_sub = "No verificado", "#b45309", "El guardrail falló: lee con cautela"
    elif info_guardrail.get("motor") == "palabras":
        guard_valor, guard_color = "Filtro de palabras", "#0d9488"
        guard_sub = f"{marcadas} explicaciones con nota" if marcadas else "Ninguna explicación marcada"
    else:
        guard_valor, guard_color = "MrBERT-biomed v2", "#0d9488"
        guard_sub = (f"{marcadas} oraciones marcadas como posible diagnóstico" if marcadas
                     else "Ninguna oración diagnóstica o alarmista")

    # --- RENDERIZADO DE TARJETAS KPI A SIMPLE VISTA ---
    st.html(f"""
    <div class="ldm-kpi-container">
        <div class="ldm-kpi-card">
            <div class="ldm-kpi-title">Exámenes analizados</div>
            <div class="ldm-kpi-value" style="color: #0f172a;">{total_analitos}</div>
            <div class="ldm-kpi-sub">Parámetros identificados</div>
        </div>
        <div class="ldm-kpi-card" style="border-left: 4px solid var(--ldm-success);">
            <div class="ldm-kpi-title">En rango normal</div>
            <div class="ldm-kpi-value" style="color: #059669;">{normales}</div>
            <div class="ldm-kpi-sub">Dentro de valores de referencia</div>
        </div>
        <div class="ldm-kpi-card" style="border-left: 4px solid var(--ldm-danger);">
            <div class="ldm-kpi-title">Requieren atención</div>
            <div class="ldm-kpi-value" style="color: #dc2626;">{alterados}</div>
            <div class="ldm-kpi-sub">{estado_resumen}</div>
        </div>
        <div class="ldm-kpi-card" style="border-left: 4px solid var(--ldm-primary);">
            <div class="ldm-kpi-title">Validación médica</div>
            <div class="ldm-kpi-value" style="color: {guard_color}; font-size: 1.4rem; padding-top: 5px;">{guard_valor}</div>
            <div class="ldm-kpi-sub">{guard_sub}</div>
        </div>
    </div>
    """)

    if info_guardrail.get("error"):
        st.warning("🛡️ No se pudo ejecutar el guardrail sobre estas explicaciones, así que no fueron revisadas "
                   "automáticamente. Léelas con cautela y confírmalas con tu médico.")
    for aviso in analisis.get("avisos") or []:
        st.info(f"ℹ️ {aviso}")

    # --- FILTRO INTERACTIVO Y BUSCADOR ---
    col_filtro, col_busca = st.columns([1.5, 1])
    with col_filtro:
        opciones_filtro = [
            f"Todos ({total_analitos})",
            f"⚠️ Requieren atención ({alterados})",
            f"🟢 En rango normal ({normales})"
        ]
        seleccion_filtro = st.radio(
            "Filtrar analitos por estado:",
            opciones_filtro,
            horizontal=True,
            label_visibility="collapsed"
        )
    with col_busca:
        busqueda = st.text_input(
            "Buscar examen:",
            placeholder="🔍 Filtrar por nombre (ej. Glucosa, Leucocitos...)",
            label_visibility="collapsed"
        )

    # Determinar qué estado filtrar
    filtro_clave = "TODOS"
    if "Requieren atención" in seleccion_filtro:
        filtro_clave = "ALTERADOS"
    elif "En rango normal" in seleccion_filtro:
        filtro_clave = "NORMAL"

    st.markdown("---")

    # --- DISPOSICIÓN PRINCIPAL EN DOS COLUMNAS ---
    col_izq, col_der = st.columns([1.35, 1])

    # ==========================
    # COLUMNA IZQUIERDA: TARJETAS DE RESULTADOS
    # ==========================
    with col_izq:
        st.subheader("📋 Interpretación de tus analitos")
        
        # Filtrar explicaciones según filtro y texto de búsqueda
        analitos_mostrados = 0

        # Se recorren los analitos (no las explicaciones): ninguno desaparece aunque el generador lo omita,
        # y cada tarjeta lleva el estado que calculó la regla para ese analito, nunca uno supuesto.
        for dato_crudo, exp in zip(datos_ner, explicaciones):
            nombre_examen = dato_crudo.get("examen", "")
            estado = str(dato_crudo.get("estado") or "DESCONOCIDO").upper()
            valor = dato_crudo.get("valor", "")
            unidad = dato_crudo.get("unidad", "")
            rango = dato_crudo.get("rango_referencia", "")

            # Aplicar filtro de estado
            if filtro_clave == "NORMAL" and estado != "NORMAL":
                continue
            if filtro_clave == "ALTERADOS" and estado not in ("ALTO", "BAJO", "ELEVADO", "ANORMAL"):
                continue

            # Aplicar filtro de búsqueda
            if busqueda.strip() and busqueda.lower().strip() not in nombre_examen.lower():
                continue

            analitos_mostrados += 1

            marcadas_exp = exp.get("oraciones_marcadas") or []
            texto_exp = exp.get("explicacion", "")
            # Con oraciones marcadas, el aviso va debajo de la tarjeta, una vez: se quita la nota pegada al texto
            if marcadas_exp:
                texto_exp = texto_exp.removesuffix(NOTA_GUARDRAIL)

            # Render de la tarjeta del analito con HTML nativo limpio
            tarjeta_html = render_tarjeta_html(
                nombre_examen=nombre_examen,
                valor=valor,
                unidad=unidad,
                estado=estado,
                rango=rango,
                explicacion=texto_exp
            )
            st.html(tarjeta_html)

            # Si el guardrail marcó oraciones preventivamente
            for oracion in marcadas_exp:
                st.warning(f"🛡️ **Nota preventiva del Guardrail Clínico:** La frase «{oracion}» fue marcada "
                           "preventivamente para recordarte que solo tu médico puede diagnosticar.")

        if analitos_mostrados == 0:
            st.info("No se encontraron analitos con los filtros seleccionados.")

        # Pasajes RAG consultados (si existen)
        pasajes = st.session_state.get("pasajes_rag")
        if pasajes:
            with st.expander("📚 Guías médicas y fuentes consultadas (MedlinePlus)"):
                for p in pasajes:
                    fuente = p.get("fuente") or ""
                    enlace = f" — [{fuente}]({fuente})" if str(fuente).startswith("http") else ""
                    titulo = p.get("titulo") or "*sin guía en nuestra base: la explicación es más general*"
                    st.markdown(f"- **{p.get('examen', '')}**: {titulo}{enlace}")

    # ==========================
    # COLUMNA DERECHA: KIT DE PREPARACIÓN MÉDICA & PREGUNTAS
    # ==========================
    with col_der:
        st.html("""
        <div class="ldm-doctor-kit">
            <h3>🩺 Mi Guía para la Consulta Médica</h3>
            <p style="font-size: 0.9rem; color: #475569;">
                Selecciona las preguntas que deseas llevar a tu cita médica y añade cualquier duda personal.
            </p>
        </div>
        """)

        st.markdown("##### 📌 Preguntas sugeridas por la IA:")
        preguntas_base = analisis.get("preguntas_sugeridas", [])
        
        # Checkboxes interactivos para seleccionar preguntas
        preguntas_actualmente_seleccionadas = []
        for i, preg in enumerate(preguntas_base):
            marcado_defecto = preg in st.session_state.preguntas_seleccionadas
            if st.checkbox(f"**{i+1}.** {preg}", value=marcado_defecto, key=f"chk_preg_{i}"):
                preguntas_actualmente_seleccionadas.append(preg)

        # Preguntas personalizadas agregadas por el paciente
        if st.session_state.preguntas_personalizadas:
            st.markdown("##### ✍️ Mis preguntas adicionales:")
            for j, p_custom in enumerate(st.session_state.preguntas_personalizadas):
                chk_c = st.checkbox(f"⭐ {p_custom}", value=True, key=f"chk_custom_{j}")
                if chk_c:
                    preguntas_actualmente_seleccionadas.append(p_custom)

        # Formulario para añadir pregunta personalizada
        with st.form("form_nueva_pregunta", clear_on_submit=True):
            nueva_duda = st.text_input(
                "Añade tu propia pregunta para el médico:",
                placeholder="Ej. ¿Puedo tomar café antes de la próxima prueba?"
            )
            btn_agregar = st.form_submit_button("➕ Añadir a mi lista")
            if btn_agregar and nueva_duda.strip():
                if nueva_duda.strip() not in st.session_state.preguntas_personalizadas:
                    st.session_state.preguntas_personalizadas.append(nueva_duda.strip())
                    st.rerun()

        st.session_state.preguntas_seleccionadas = preguntas_actualmente_seleccionadas

        # Ficha resumen para la consulta: PDF para imprimir, imagen para el celular o texto plano
        st.markdown("##### 📥 Ficha resumen para tu consulta")
        formato_ficha = st.radio(
            "Formato de la ficha:",
            list(FORMATOS_FICHA),
            horizontal=True,
            label_visibility="collapsed",
            key="formato_ficha",
        )
        extension, mime_ficha, ayuda_ficha = FORMATOS_FICHA[formato_ficha]
        st.caption(ayuda_ficha)
        # El tipo de examen viene del documento: se quitan "/", ":" y demás caracteres no válidos en un nombre de archivo
        nombre_base = re.sub(r'[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+', '_', tipo_ex).strip('_') or 'Examen'
        args_ficha = dict(tipo_examen=tipo_ex, analitos=datos_ner, explicaciones=explicaciones,
                          preguntas=st.session_state.preguntas_seleccionadas,
                          contexto_paciente=st.session_state.get("contexto_paciente", ""))
        try:
            if extension == "pdf":
                datos_ficha = ficha_consulta.ficha_pdf(**args_ficha)
            elif extension == "png":
                datos_ficha = ficha_consulta.ficha_png(**args_ficha)
            else:
                datos_ficha = generar_ficha_medica_txt(
                    tipo_examen=tipo_ex, analitos=datos_ner,
                    preguntas_seleccionadas=st.session_state.preguntas_seleccionadas,
                    contexto_paciente=st.session_state.get("contexto_paciente", ""))
        except Exception as e:
            # Si falla el PDF o la imagen, el texto plano siempre está disponible
            st.warning(f"No se pudo generar la ficha en {formato_ficha} ({e}); se ofrece en texto.")
            extension, mime_ficha = "txt", "text/plain"
            datos_ficha = generar_ficha_medica_txt(
                tipo_examen=tipo_ex, analitos=datos_ner,
                preguntas_seleccionadas=st.session_state.preguntas_seleccionadas,
                contexto_paciente=st.session_state.get("contexto_paciente", ""))

        st.download_button(
            label=f"📥 Descargar ficha ({extension.upper()})",
            data=datos_ficha,
            file_name=f"Guia_Consulta_{nombre_base}.{extension}",
            mime=mime_ficha,
            use_container_width=True,
            type="primary"
        )

        # Vista previa del documento subido si existe
        if st.session_state.get("ultimo_archivo_bytes") and st.session_state.get("ultimo_archivo_mime"):
            with st.expander("👁️ Ver documento original subido"):
                mime = st.session_state.ultimo_archivo_mime
                if mime.startswith("image/"):
                    st.image(st.session_state.ultimo_archivo_bytes, caption=st.session_state.ultimo_archivo_nombre, use_container_width=True)
                else:
                    st.info(f"📄 Archivo PDF: **{st.session_state.ultimo_archivo_nombre}**")

        # Disclaimer médico visible
        disclaimer_text = analisis.get(
            "disclaimer",
            "⚠️ IMPORTANTE: Esta información es generada por IA para propósitos informativos y NO constituye un diagnóstico médico. Consulta siempre a tu médico."
        )
        st.html(f"""
        <div class="ldm-disclaimer-card">
            <strong>Aviso de Seguridad Clínica:</strong><br>
            {html.escape(disclaimer_text)}
        </div>
        """)

    # =========================================================================
    # SECCIÓN INTERACTIVA: PREGÚNTALE A LETRADEMÉDICO (ACLARACIONES EN VIVO)
    # =========================================================================
    st.markdown("---")
    st.subheader("💬 ¿Tienes alguna duda adicional sobre estos resultados?")
    st.markdown(
        "Hazle una pregunta a nuestro asistente inteligente sobre los valores analizados, "
        "términos médicos confusos o qué hábitos saludables suelen vincularse a estos parámetros."
    )

    # Historial de conversación sobre dudas
    if st.session_state.chat_dudas:
        for item in st.session_state.chat_dudas:
            with st.chat_message("user"):
                st.write(item["pregunta"])
            with st.chat_message("assistant", avatar="🩺"):
                st.write(item["respuesta"])

    col_chat_input, col_chat_btn = st.columns([4, 1])
    with col_chat_input:
        pregunta_chat = st.text_input(
            "Escribe tu duda aquí:",
            placeholder="Ej. ¿Qué alimentos pueden ayudar a subir mi colesterol HDL sin medicamentos?",
            label_visibility="collapsed",
            key="input_duda_paciente"
        )
    with col_chat_btn:
        btn_enviar_duda = st.button("Enviar Pregunta", use_container_width=True)

    if btn_enviar_duda and pregunta_chat.strip():
        with st.spinner("LetraDeMédico está redactando una explicación clara y segura..."):
            respuesta_ia = backend.responder_pregunta_seguimiento(
                analitos=datos_ner,
                tipo_examen=tipo_ex,
                pregunta=pregunta_chat.strip(),
                contexto_paciente=st.session_state.get("contexto_paciente", "")
            )
            st.session_state.chat_dudas.append({
                "pregunta": pregunta_chat.strip(),
                "respuesta": respuesta_ia
            })
            st.rerun()
