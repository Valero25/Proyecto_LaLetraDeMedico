"""
Evidencia 3: tiempo del pipeline completo (lectura local -> reglas -> RAG -> generador -> guardrail)
con el PDF real de 8 analitos. Uso:  python evidencias/medir_tiempos.py "ruta/al.pdf" gpu|cpu
"""
import os
import sys
import time
import platform
import subprocess
from datetime import datetime
from pathlib import Path

ruta_pdf, modo = sys.argv[1], sys.argv[2]
if modo == "cpu":
    os.environ["LDM_GPU_CAPAS"] = "0"
os.environ["LDM_LECTURA"] = "local"

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))


def equipo():
    import psutil
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                         capture_output=True, text=True).stdout.strip()
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        gpu = "sin GPU NVIDIA"
    return (f"CPU: {cpu} ({psutil.cpu_count(logical=False)} núcleos / {psutil.cpu_count()} hilos)\n"
            f"RAM: {psutil.virtual_memory().total / 2**30:.1f} GB\n"
            f"GPU: {gpu}\n"
            f"SO: {platform.platform()} | Python {platform.python_version()}")


t0 = time.perf_counter()
from backend import LetraDeMedicoBackend  # noqa: E402
import modelos_propios as mp  # noqa: E402

backend = LetraDeMedicoBackend(modo="propio")
datos = Path(ruta_pdf).read_bytes()
# Calentamiento: carga los modelos (no se cuenta, igual que en la app, donde se cargan una sola vez)
for nombre in ("rag", "generador", "guardrail"):
    backend.modelo_propio(nombre)
t_carga = time.perf_counter() - t0

t1 = time.perf_counter()
salida = backend.procesar_resultado(datos, "application/pdf")
t_examen = time.perf_counter() - t1

gen = backend.modelo_propio("generador")
capas = getattr(gen, "capas_gpu", "?")
analitos = salida.get("datos_extraidos") or []

lineas = [
    f"LetraDeMédico - medición de tiempo ({modo.upper()})",
    f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}",
    equipo(),
    f"Generador: {mp.RUTAS['generador'].name if hasattr(mp, 'RUTAS') else '?'} | n_gpu_layers = {capas}",
    f"Lectura del documento: {salida.get('lectura')}",
    f"Analitos procesados: {len(analitos) if analitos else '?'}",
    f"Carga de modelos (una vez por sesión): {t_carga:.1f} s",
    f"TIEMPO POR EXAMEN (lectura + reglas + RAG + generador + guardrail): {t_examen:.1f} s",
]
if "error" in salida:
    lineas.append(f"ERROR: {salida['error']}")
texto = "\n".join(lineas)
print(texto)
(Path(__file__).parent / f"03_tiempo_{modo}.txt").write_text(texto + "\n", encoding="utf-8")
