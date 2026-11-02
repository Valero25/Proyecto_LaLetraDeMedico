"""
Componentes de interfaz de usuario y utilidades visuales para LetraDeMédico.
Contiene estilos CSS modernos, presets clínicos interactivos, generador de indicadores
gráficos de rango y exportador de guías médicas para el paciente.
"""
import html
import re
from datetime import datetime

from backend import normalizar_rango

# --- ESTILOS CSS MODERNOS PARA STREAMLIT ---
ESTILOS_CSS = """
<style>
/* Tipografía y variables base */
:root {
    --ldm-primary: #0d9488;
    --ldm-primary-light: #ccfbf1;
    --ldm-primary-dark: #0f766e;
    --ldm-success: #10b981;
    --ldm-success-bg: #ecfdf5;
    --ldm-success-border: #a7f3d0;
    --ldm-warning: #f59e0b;
    --ldm-warning-bg: #fffbeb;
    --ldm-warning-border: #fde68a;
    --ldm-danger: #ef4444;
    --ldm-danger-bg: #fef2f2;
    --ldm-danger-border: #fecaca;
    --ldm-purple: #8b5cf6;
    --ldm-purple-bg: #f5f3ff;
    --ldm-purple-border: #ddd6fe;
    --ldm-card-bg: rgba(255, 255, 255, 0.95);
    --ldm-card-border: rgba(148, 163, 184, 0.25);
}

/* Encabezado principal elegante */
.ldm-hero {
    background: linear-gradient(135deg, #0f766e 0%, #0d9488 45%, #0284c7 100%);
    border-radius: 18px;
    padding: 26px 30px;
    color: #ffffff;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(13, 148, 136, 0.35);
    display: flex;
    align-items: center;
    gap: 24px;
}
.ldm-hero-icon {
    font-size: 3.5rem;
    background: rgba(255, 255, 255, 0.15);
    width: 80px;
    height: 80px;
    min-width: 80px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 2px 4px rgba(255, 255, 255, 0.2);
}
.ldm-hero-content h1 {
    color: #ffffff !important;
    margin: 0 0 6px 0;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.ldm-hero-content p {
    color: #e0f2fe;
    font-size: 1.05rem;
    margin: 0;
    line-height: 1.45;
}
.ldm-hero-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.22);
    color: #ffffff;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Stepper de progreso interactivo */
.ldm-stepper {
    display: flex;
    justify-content: space-between;
    margin-bottom: 24px;
    background: rgba(148, 163, 184, 0.08);
    border-radius: 12px;
    padding: 10px 18px;
    border: 1px solid rgba(148, 163, 184, 0.2);
}
.ldm-step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #64748b;
}
.ldm-step.active {
    color: #0d9488;
}
.ldm-step-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #cbd5e1;
    color: #475569;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
}
.ldm-step.active .ldm-step-num {
    background: #0d9488;
    color: #ffffff;
}

/* Tarjetas de métricas KPI a simple vista */
.ldm-kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 22px;
}
.ldm-kpi-card {
    background: rgba(248, 250, 252, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 14px;
    padding: 16px 18px;
    text-align: left;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.ldm-kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.05);
}
.ldm-kpi-title {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #64748b;
    letter-spacing: 0.4px;
    margin-bottom: 4px;
}
.ldm-kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 2px;
}
.ldm-kpi-sub {
    font-size: 0.78rem;
    color: #94a3b8;
}

/* Tarjetas de resultados de analitos */
.ldm-card {
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}
.ldm-card:hover {
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.07);
    border-color: rgba(13, 148, 136, 0.4);
}
.ldm-card-normal {
    border-left: 6px solid var(--ldm-success);
}
.ldm-card-alto {
    border-left: 6px solid var(--ldm-danger);
}
.ldm-card-bajo {
    border-left: 6px solid var(--ldm-warning);
}
.ldm-card-anormal {
    border-left: 6px solid var(--ldm-purple);
}
.ldm-card-desconocido {
    border-left: 6px solid #94a3b8;
}

.ldm-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
    gap: 12px;
}
.ldm-card-title {
    font-size: 1.18rem;
    font-weight: 700;
    margin: 0;
    color: #1e293b;
}
.ldm-card-value {
    font-size: 1.05rem;
    font-weight: 600;
    color: #475569;
    margin: 3px 0 10px 0;
}

/* Badges de estado modernos */
.ldm-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}
.ldm-badge-normal {
    background-color: var(--ldm-success-bg);
    color: #065f46;
    border: 1px solid var(--ldm-success-border);
}
.ldm-badge-alto {
    background-color: var(--ldm-danger-bg);
    color: #991b1b;
    border: 1px solid var(--ldm-danger-border);
}
.ldm-badge-bajo {
    background-color: var(--ldm-warning-bg);
    color: #92400e;
    border: 1px solid var(--ldm-warning-border);
}
.ldm-badge-anormal {
    background-color: var(--ldm-purple-bg);
    color: #5b21b6;
    border: 1px solid var(--ldm-purple-border);
}
.ldm-badge-desconocido {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
}

/* Indicador visual de Rango (Gauge horizontal) */
.ldm-gauge-box {
    margin: 12px 0 14px 0;
    padding: 10px 14px 8px 14px;
    background: rgba(241, 245, 249, 0.65);
    border-radius: 10px;
    border: 1px solid rgba(226, 232, 240, 0.8);
}
.ldm-gauge-track {
    position: relative;
    height: 10px;
    border-radius: 6px;
    display: flex;
    overflow: visible;
    background: #e2e8f0;
    margin: 12px 0 6px 0;
}
.ldm-gauge-zone-low {
    height: 100%;
    background: #fde68a;
    border-top-left-radius: 6px;
    border-bottom-left-radius: 6px;
}
.ldm-gauge-zone-normal {
    height: 100%;
    background: #86efac;
}
.ldm-gauge-zone-high {
    height: 100%;
    background: #fca5a5;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}
.ldm-gauge-marker {
    position: absolute;
    top: -6px;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    z-index: 3;
}
.ldm-marker-pin {
    width: 14px;
    height: 22px;
    background: #0f172a;
    border: 2px solid #ffffff;
    border-radius: 7px;
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.35);
}
.ldm-gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    margin-top: 4px;
}

/* Explicación en lenguaje claro */
.ldm-explicacion {
    font-size: 0.95rem;
    line-height: 1.55;
    color: #334155;
    margin-top: 10px;
    padding: 10px 12px;
    background: rgba(248, 250, 252, 0.75);
    border-radius: 8px;
    border-left: 3px solid #cbd5e1;
}

/* Panel de Preparación para la Consulta Médica */
.ldm-doctor-kit {
    background: linear-gradient(180deg, #f0fdfa 0%, #ffffff 100%);
    border: 1px solid #99f6e4;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(13, 148, 136, 0.08);
}
.ldm-doctor-kit h3 {
    color: #0f766e !important;
    margin-top: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Tarjeta de Presets interactivos */
.ldm-preset-card {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px;
    text-align: left;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.ldm-preset-card:hover {
    border-color: #0d9488;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.12);
}

/* Disclaimer médico amigable */
.ldm-disclaimer-card {
    background: #fffbeb;
    border: 1px solid #fef3c7;
    border-left: 5px solid #f59e0b;
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 20px;
    color: #92400e;
    font-size: 0.88rem;
    line-height: 1.45;
}

/* Soporte para modo oscuro en Streamlit */
@media (prefers-color-scheme: dark) {
    .ldm-card {
        background: rgba(30, 41, 59, 0.75);
        border-color: rgba(148, 163, 184, 0.2);
    }
    .ldm-card-title {
        color: #f1f5f9;
    }
    .ldm-card-value {
        color: #cbd5e1;
    }
    .ldm-explicacion {
        background: rgba(15, 23, 42, 0.5);
        color: #e2e8f0;
        border-left-color: #475569;
    }
    .ldm-kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border-color: rgba(148, 163, 184, 0.15);
    }
    .ldm-gauge-box {
        background: rgba(15, 23, 42, 0.5);
        border-color: rgba(51, 65, 85, 0.8);
    }
    .ldm-doctor-kit {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        border-color: #134e4a;
    }
    .ldm-marker-pin {
        background: #38bdf8;
        border-color: #0f172a;
    }
}
</style>
"""

# --- PRESETS CLÍNICOS INTERACTIVOS DE EJEMPLO ---
PRESETS = {
    "hemograma_lipidos": {
        "nombre": "🩸 Hemograma & Perfil Lipídico",
        "descripcion": "Chequeo preventivo de rutina: defensas, azúcar y grasas en sangre.",
        "tipo_examen": "Hemograma y Perfil Lipídico",
        "contexto_paciente": "Me sentí un poco fatigado en las tardes y el médico solicitó exámenes preventivos anuales.",
        "analitos": [
            {"examen": "Hemoglobina", "valor": "14.2", "unidad": "g/dL", "rango_referencia": "13.0 - 17.0"},
            {"examen": "Leucocitos", "valor": "7.20", "unidad": "10³/µL", "rango_referencia": "4.0 - 11.0"},
            {"examen": "Glucosa", "valor": "108", "unidad": "mg/dL", "rango_referencia": "70 - 99"},
            {"examen": "Colesterol total", "valor": "178", "unidad": "mg/dL", "rango_referencia": "< 200"},
            {"examen": "Colesterol HDL", "valor": "36", "unidad": "mg/dL", "rango_referencia": "> 40"},
            {"examen": "Triglicéridos", "valor": "118", "unidad": "mg/dL", "rango_referencia": "< 150"},
        ],
        "explicaciones": [
            {"examen": "Hemoglobina", "explicacion": "La hemoglobina es la proteína de los glóbulos rojos que transporta el oxígeno a todas las células del cuerpo. Tu valor de 14.2 g/dL está dentro del rango normal (13.0 - 17.0), lo que indica un transporte de oxígeno adecuado sin signos de anemia."},
            {"examen": "Leucocitos", "explicacion": "Los leucocitos son las células de defensa (glóbulos blancos) encargadas de combatir virus y bacterias. El valor de 7.20 10³/µL está en un rango equilibrado (4.0 - 11.0), sin señales de infección activa."},
            {"examen": "Glucosa", "explicacion": "La glucosa es el azúcar en sangre que provee energía rápida. Tu valor de 108 mg/dL está ligeramente por encima de lo deseable en ayunas (70 - 99). Esto no es un diagnóstico, pero vale la pena revisarlo con tu médico para ver si conviene repetir la prueba o ajustar hábitos."},
            {"examen": "Colesterol total", "explicacion": "El colesterol es una sustancia grasa esencial para la estructura celular y hormonas. Tu nivel de 178 mg/dL es saludable y se encuentra bien por debajo del tope sugerido (< 200)."},
            {"examen": "Colesterol HDL", "explicacion": "El HDL se conoce popularmente como el 'colesterol bueno' porque ayuda a limpiar el exceso de grasa en las arterias. Tu valor de 36 mg/dL está un poco por debajo del mínimo recomendado (> 40). El ejercicio aeróbico regular y grasas saludables ayudan a elevarlo."},
            {"examen": "Triglicéridos", "explicacion": "Los triglicéridos son otra forma de grasa que almacena calorías no utilizadas. Un nivel de 118 mg/dL está en un rango favorable (menos de 150)."}
        ],
        "preguntas": [
            "Mi glucosa en ayunas salió en 108 mg/dL, ¿recomienda repetir la prueba o solicitar una Hemoglobina Glicosilada (HbA1c)?",
            "¿Qué cambios en mi alimentación o actividad física me ayudarían a subir el colesterol HDL?",
            "¿Estos resultados se relacionan con la fatiga que he sentido?",
            "¿En cuánto tiempo sería conveniente programar un próximo control de laboratorio?"
        ]
    },
    "metabolismo_tiroides": {
        "nombre": "⚡ Metabolismo & Tiroides",
        "descripcion": "Evaluación metabólica enfocada en energía, tiroides y control de glucosa.",
        "tipo_examen": "Perfil Metabólico y Tiroideo",
        "contexto_paciente": "Dificultad para bajar de peso, somnolencia después de comer y antecedentes familiares de diabetes.",
        "analitos": [
            {"examen": "TSH (Hormona Tiroestimulante)", "valor": "2.45", "unidad": "µUI/mL", "rango_referencia": "0.40 - 4.50"},
            {"examen": "T4 Libre", "valor": "1.20", "unidad": "ng/dL", "rango_referencia": "0.80 - 1.80"},
            {"examen": "Glucosa en ayunas", "valor": "115", "unidad": "mg/dL", "rango_referencia": "70 - 99"},
            {"examen": "Hemoglobina Glicosilada (HbA1c)", "valor": "6.1", "unidad": "%", "rango_referencia": "< 5.7"},
            {"examen": "Insulina basal", "valor": "18.2", "unidad": "µUI/mL", "rango_referencia": "2.6 - 24.9"},
        ],
        "explicaciones": [
            {"examen": "TSH (Hormona Tiroestimulante)", "explicacion": "La TSH regula el ritmo metabólico del organismo estimulando a la tiroides. Tu valor de 2.45 µUI/mL está óptimo y dentro del rango normal (0.40 - 4.50), lo que sugiere que tu tiroides responde con normalidad."},
            {"examen": "T4 Libre", "explicacion": "La T4 Libre es la hormona tiroidea activa en circulación. Un resultado de 1.20 ng/dL confirma que el funcionamiento de la glándula tiroidea es adecuado."},
            {"examen": "Glucosa en ayunas", "explicacion": "Tu nivel de 115 mg/dL está elevado respecto al límite normal (70 - 99). Al combinarse con antecedentes familiares, es un dato clave para evaluar con tu médico de forma preventiva."},
            {"examen": "Hemoglobina Glicosilada (HbA1c)", "explicacion": "La HbA1c muestra el promedio de azúcar en tu sangre durante los últimos 3 meses. Un valor de 6.1% se sitúa por encima del umbral esperado (< 5.7%), lo que suele abrir la puerta a medidas preventivas nutricionales."},
            {"examen": "Insulina basal", "explicacion": "La insulina ayuda a que la glucosa ingrese a las células. Tu cifra de 18.2 µUI/mL está dentro del rango formal (2.6 - 24.9), aunque tu médico valorará la relación directa entre tu insulina y tu glucosa."}
        ],
        "preguntas": [
            "Con una HbA1c de 6.1% y glucosa en 115, ¿cuáles son los pasos preventivos recomendados para evitar que progrese?",
            "¿Es necesario consultar con un nutricionista o especialista en endocrinología?",
            "¿Convendría realizar una prueba de tolerancia oral a la glucosa?",
            "¿Mis niveles de tiroides descartan que el cansancio provenga de la glándula tiroidea?"
        ]
    },
    "renal_urolisis": {
        "nombre": "🫘 Función Renal & Uroanálisis",
        "descripcion": "Evaluación del filtrado renal, hidratación y estado del tracto urinario.",
        "tipo_examen": "Función Renal y Uroanálisis",
        "contexto_paciente": "Chequeo médico tras haber iniciado ejercicio de fuerza y aumentado el consumo de agua.",
        "analitos": [
            {"examen": "Creatinina en sangre", "valor": "0.95", "unidad": "mg/dL", "rango_referencia": "0.70 - 1.30"},
            {"examen": "Nitrógeno Ureico (BUN)", "valor": "15.0", "unidad": "mg/dL", "rango_referencia": "7.0 - 20.0"},
            {"examen": "Ácido Úrico", "valor": "7.8", "unidad": "mg/dL", "rango_referencia": "3.5 - 7.2"},
            {"examen": "Tasa de Filtración Glomerular (eGFR)", "valor": "96", "unidad": "mL/min/1.73m²", "rango_referencia": "> 90"},
            {"examen": "Proteínas en orina", "valor": "Negativo", "unidad": "", "rango_referencia": "Negativo"},
        ],
        "explicaciones": [
            {"examen": "Creatinina en sangre", "explicacion": "La creatinina es un residuo de la actividad muscular filtrado exclusivamente por los riñones. Tu cifra de 0.95 mg/dL indica que tus riñones están depurando la sangre con total normalidad."},
            {"examen": "Nitrógeno Ureico (BUN)", "explicacion": "El BUN mide la cantidad de nitrógeno proveniente de la degradación proteica. Con 15.0 mg/dL dentro del rango (7.0 - 20.0), tu hidratación y procesamiento proteico se encuentran estables."},
            {"examen": "Ácido Úrico", "explicacion": "El ácido úrico proviene de la digestión de purinas presentes en ciertos alimentos y células. Tu nivel de 7.8 mg/dL está ligeramente por encima de la referencia (3.5 - 7.2). Mantener una hidratación generosa y moderar carnes rojas y bebidas azucaradas suele ser muy útil."},
            {"examen": "Tasa de Filtración Glomerular (eGFR)", "explicacion": "El eGFR estima el porcentaje y velocidad de filtrado de tus riñones. Un valor de 96 (> 90) refleja una función renal excelente y robusta."},
            {"examen": "Proteínas en orina", "explicacion": "Los riñones sanos retienen las proteínas en la sangre. El resultado 'Negativo' confirma que no hay fuga de proteínas en la orina."}
        ],
        "preguntas": [
            "Mi ácido úrico está en 7.8 mg/dL; ¿qué ajustes en mi dieta o hidratación me sugiere para normalizarlo?",
            "Con estos resultados renales, ¿puedo continuar con mi plan de ejercicio físico sin restricciones?",
            "¿Requiere el ácido úrico algún tratamiento o solo seguimiento periódico?"
        ]
    }
}


def calcular_gauge_html(valor_str, rango_str, estado):
    """
    Calcula las posiciones porcentuales y genera el HTML del indicador gráfico de rango.
    Si el examen es cualitativo ('Negativo', 'Normal'), retorna None para renderizar badge simple.
    """
    val_clean = str(valor_str).replace(',', '.').strip()
    # La misma normalización que la regla de estados: "7 a 56", "Hasta 200", "Menor de 5,7"...
    rng_clean = normalizar_rango(rango_str)
    val_attr = html.escape(val_clean, quote=True)

    val_match = re.search(r'[\d\.]+', val_clean)
    if not val_match:
        return None

    try:
        val = float(val_match.group())
    except ValueError:
        return None

    # Caso 1: Rango con guion '70 - 99' o '13.0 - 17.0'
    m_range = re.search(r'(\d[\d\.]*)-(\d[\d\.]*)', rng_clean)
    if m_range:
        try:
            min_ref = float(m_range.group(1))
            max_ref = float(m_range.group(2))
            if max_ref > min_ref:
                span = max_ref - min_ref
                margin = span * 0.4
                disp_min = min_ref - margin
                disp_max = max_ref + margin

                low_pct = ((min_ref - disp_min) / (disp_max - disp_min)) * 100
                high_pct = ((max_ref - disp_min) / (disp_max - disp_min)) * 100
                normal_width = high_pct - low_pct
                high_width = 100 - high_pct

                pct = ((val - disp_min) / (disp_max - disp_min)) * 100
                pct = max(3, min(97, pct))

                return (
                    f'<div class="ldm-gauge-box">'
                    f'<div class="ldm-gauge-track">'
                    f'<div class="ldm-gauge-zone-low" style="width: {low_pct:.1f}%;"></div>'
                    f'<div class="ldm-gauge-zone-normal" style="width: {normal_width:.1f}%;"></div>'
                    f'<div class="ldm-gauge-zone-high" style="width: {high_width:.1f}%;"></div>'
                    f'<div class="ldm-gauge-marker" style="left: {pct:.1f}%;" title="Tu valor: {val_attr}">'
                    f'<div class="ldm-marker-pin"></div>'
                    f'</div>'
                    f'</div>'
                    f'<div class="ldm-gauge-labels">'
                    f'<span>Bajo (&lt; {min_ref:g})</span>'
                    f'<span style="color: #059669;">Rango normal: {min_ref:g} - {max_ref:g}</span>'
                    f'<span>Alto (&gt; {max_ref:g})</span>'
                    f'</div>'
                    f'</div>'
                )
        except Exception:
            pass

    # Caso 2: Menor que '< 200'
    m_less = re.search(r'<\s*(=?)\s*([\d\.]+)', rng_clean)
    if m_less:
        try:
            max_ref = float(m_less.group(2))
            if max_ref > 0:
                low_pct = 0
                normal_width = 75
                high_width = 25
                if val <= max_ref:
                    pct = (val / max_ref) * 75
                else:
                    pct = 75 + min(22, ((val - max_ref) / max_ref) * 50)
                pct = max(3, min(97, pct))

                return (
                    f'<div class="ldm-gauge-box">'
                    f'<div class="ldm-gauge-track">'
                    f'<div class="ldm-gauge-zone-normal" style="width: {normal_width}%;"></div>'
                    f'<div class="ldm-gauge-zone-high" style="width: {high_width}%;"></div>'
                    f'<div class="ldm-gauge-marker" style="left: {pct:.1f}%;" title="Tu valor: {val_attr}">'
                    f'<div class="ldm-marker-pin"></div>'
                    f'</div>'
                    f'</div>'
                    f'<div class="ldm-gauge-labels">'
                    f'<span style="color: #059669;">Deseable (&lt; {max_ref:g})</span>'
                    f'<span>Elevado (&gt; {max_ref:g})</span>'
                    f'</div>'
                    f'</div>'
                )
        except Exception:
            pass

    # Caso 3: Mayor que '> 40'
    m_more = re.search(r'>\s*(=?)\s*([\d\.]+)', rng_clean)
    if m_more:
        try:
            min_ref = float(m_more.group(2))
            if min_ref > 0:
                low_width = 30
                normal_width = 70
                if val < min_ref:
                    pct = (val / min_ref) * 30
                else:
                    pct = 30 + min(65, ((val - min_ref) / min_ref) * 40)
                pct = max(3, min(97, pct))

                return (
                    f'<div class="ldm-gauge-box">'
                    f'<div class="ldm-gauge-track">'
                    f'<div class="ldm-gauge-zone-low" style="width: {low_width}%;"></div>'
                    f'<div class="ldm-gauge-zone-normal" style="width: {normal_width}%;"></div>'
                    f'<div class="ldm-gauge-marker" style="left: {pct:.1f}%;" title="Tu valor: {val_attr}">'
                    f'<div class="ldm-marker-pin"></div>'
                    f'</div>'
                    f'</div>'
                    f'<div class="ldm-gauge-labels">'
                    f'<span>Bajo (&lt; {min_ref:g})</span>'
                    f'<span style="color: #059669;">Deseable (&gt; {min_ref:g})</span>'
                    f'</div>'
                    f'</div>'
                )
        except Exception:
            pass

    return None


def render_tarjeta_html(nombre_examen, valor, unidad, estado, rango, explicacion):
    """
    Construye el HTML limpio y compacto de la tarjeta del analito.
    Al no contener saltos con sangría de 4 espacios, Streamlit nunca lo confunde con un bloque de código.
    """
    # Por defecto no se afirma nada: solo un estado calculado como NORMAL se pinta de verde
    card_class = "ldm-card-desconocido"
    badge_class = "ldm-badge-desconocido"
    badge_icono = "⚪"
    estado_texto = "SIN EVALUAR"
    estado = str(estado or "").upper()

    if estado == "NORMAL":
        card_class = "ldm-card-normal"
        badge_class = "ldm-badge-normal"
        badge_icono = "🟢"
        estado_texto = "NORMAL"
    elif estado in ("ALTO", "ELEVADO"):
        card_class = "ldm-card-alto"
        badge_class = "ldm-badge-alto"
        badge_icono = "🔴"
        estado_texto = "ELEVADO"
    elif estado == "BAJO":
        card_class = "ldm-card-bajo"
        badge_class = "ldm-badge-bajo"
        badge_icono = "🟡"
        estado_texto = "BAJO"
    elif estado == "ANORMAL":
        card_class = "ldm-card-anormal"
        badge_class = "ldm-badge-anormal"
        badge_icono = "🟣"
        estado_texto = "FUERA DE RANGO"

    # Sin un estado calculado, el indicador gráfico podría contradecir la etiqueta: solo se muestra el rango
    gauge_html = calcular_gauge_html(valor, rango, estado) if rango and estado != "DESCONOCIDO" else ""
    if not gauge_html and rango:
        gauge_html = (
            f'<div style="font-size: 0.8rem; color: #64748b; margin: 4px 0 8px 0;">'
            f'Valor de referencia esperado: <strong>{html.escape(str(rango))}</strong>'
            f'</div>'
        )

    val_unidad = f"{valor} {unidad}".strip() if valor else "No reportado"
    # Todo lo que viene del documento o del generador se escapa: un "<200" o un texto raro no rompe la tarjeta
    explicacion_html = html.escape(str(explicacion or "")).replace("\n", "<br>")

    return (
        f'<div class="ldm-card {card_class}">'
        f'<div class="ldm-card-header">'
        f'<div>'
        f'<div class="ldm-card-title">{html.escape(str(nombre_examen))}</div>'
        f'<div class="ldm-card-value">Resultado: <strong>{html.escape(val_unidad)}</strong></div>'
        f'</div>'
        f'<span class="ldm-badge {badge_class}">{badge_icono} {estado_texto}</span>'
        f'</div>'
        f'{gauge_html}'
        f'<div class="ldm-explicacion">{explicacion_html}</div>'
        f'</div>'
    )



def generar_ficha_medica_txt(tipo_examen, analitos, preguntas_seleccionadas, contexto_paciente=""):
    """
    Genera un texto ordenado y profesional que el paciente puede descargar o imprimir
    para llevar a su próxima consulta con el médico.
    """
    fecha = datetime.now().strftime("%d/%m/%Y")
    lineas = [
        "=" * 60,
        "GUÍA PARA LA CONSULTA MÉDICA - LETRADEMÉDICO",
        "=" * 60,
        f"Fecha: {fecha}",
        f"Tipo de Examen: {tipo_examen}",
    ]
    if contexto_paciente:
        lineas.append(f"Motivo / Contexto: {contexto_paciente}")

    lineas.append("\n" + "-" * 60)
    lineas.append("1. RESUMEN DE VALORES QUE REQUIEREN COMENTARIO:")
    lineas.append("-" * 60)

    alterados = [a for a in analitos if a.get("estado", "").upper() in ("ALTO", "BAJO", "ELEVADO", "ANORMAL")]
    if alterados:
        for a in alterados:
            estado = a.get("estado", "ALTERADO").upper()
            lineas.append(
                f" • [{estado}] {a.get('examen')}: {a.get('valor')} {a.get('unidad')} "
                f"(Rango ref: {a.get('rango_referencia')})"
            )
    else:
        lineas.append(" • Todos los parámetros evaluados se encuentran dentro de los rangos de referencia.")

    lineas.append("\n" + "-" * 60)
    lineas.append("2. OTROS VALORES EVALUADOS EN RANGO:")
    lineas.append("-" * 60)
    normales = [a for a in analitos if a.get("estado", "").upper() == "NORMAL"]
    for a in normales:
        lineas.append(
            f" • [NORMAL] {a.get('examen')}: {a.get('valor')} {a.get('unidad')} "
            f"(Rango: {a.get('rango_referencia')})"
        )

    # Los que la regla no pudo evaluar también van a la consulta: no se pierden en silencio
    sin_evaluar = [a for a in analitos
                   if str(a.get("estado", "")).upper() not in ("NORMAL", "ALTO", "BAJO", "ELEVADO", "ANORMAL")]
    if sin_evaluar:
        lineas.append("\n" + "-" * 60)
        lineas.append("   VALORES SIN EVALUAR AUTOMÁTICAMENTE (pregúntale a tu médico):")
        lineas.append("-" * 60)
        for a in sin_evaluar:
            lineas.append(
                f" • [SIN EVALUAR] {a.get('examen')}: {a.get('valor')} {a.get('unidad')} "
                f"(Rango: {a.get('rango_referencia')})"
            )

    lineas.append("\n" + "-" * 60)
    lineas.append("3. MIS PREGUNTAS SELECCIONADAS PARA EL MÉDICO:")
    lineas.append("-" * 60)
    if preguntas_seleccionadas:
        for idx, p in enumerate(preguntas_seleccionadas, 1):
            lineas.append(f" {idx}. {p}")
    else:
        lineas.append(" (No se seleccionaron preguntas personalizadas)")

    lineas.append("\n" + "=" * 60)
    lineas.append("AVISO IMPORTANTE:")
    lineas.append("Este resumen fue preparado con la asistencia de LetraDeMédico para facilitar")
    lineas.append("la comunicación entre el paciente y su médico. No sustituye la valoración clínica.")
    lineas.append("=" * 60)

    return "\n".join(lineas)
