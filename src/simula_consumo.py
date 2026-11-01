"""
simula_consumo.py - Simula un cliente externo que consume la API de WODPredict-AI (Fase II).

Representa a otro sistema del box (por ejemplo, la app del coach) que envía los datos de los
atletas a POST /predict y recibe el riesgo de sobrecarga en JSON.

Uso (con la API o el contenedor en ejecución):

    python src/simula_consumo.py                                   # contenedor local
    python src/simula_consumo.py https://<mi-app>.azurecontainerapps.io   # API en Azure
"""
import json
import sys
from pathlib import Path

import requests

# Dirección de la API: la que se pase al ejecutar o, si no, el contenedor local.
API_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
CARPETA_EJEMPLOS = Path(__file__).resolve().parent.parent / "tests" / "ejemplos"

# 1. Confirmar que el servicio responde (health check).
estado = requests.get(f"{API_URL}/", timeout=10)
print(f"GET / -> HTTP {estado.status_code}: {estado.json()}\n")

# 2. Enviar los tres perfiles de ejemplo, uno por uno, como lo haría la app del coach.
for perfil in ["bajo", "moderado", "alto"]:
    datos = json.loads((CARPETA_EJEMPLOS / f"atleta_riesgo_{perfil}.json").read_text(encoding="utf-8"))
    respuesta = requests.post(f"{API_URL}/predict", json=datos, timeout=10)
    resultado = respuesta.json()
    print(f"Atleta de ejemplo '{perfil}' -> HTTP {respuesta.status_code}")
    print(f"  Riesgo {resultado['riesgo']} ({resultado['probabilidad']:.1%}) | alerta: {resultado['prediccion']}")
    print(f"  Recomendación: {resultado['recomendacion']}\n")

# 3. Enviar un dato inválido para ver cómo responde la API (RPE fuera de la escala 1 a 10).
datos["prev_rpe"] = 11
respuesta = requests.post(f"{API_URL}/predict", json=datos, timeout=10)
error = respuesta.json()["detail"][0]
print(f"Dato inválido (prev_rpe = 11) -> HTTP {respuesta.status_code}: {error['loc'][-1]} - {error['msg']}")
