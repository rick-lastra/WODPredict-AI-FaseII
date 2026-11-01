"""
test_api.py - Pruebas de la API de WODPredict-AI contra el contenedor en ejecución (Fase II).

Antes de ejecutar las pruebas, el contenedor debe estar corriendo:

    docker run -d --name wodpredict -p 8000:8000 wodpredict-api:1.0
    pytest tests/ -v

Para probar otra dirección (por ejemplo, la URL de Azure):

    API_URL=https://<mi-app>.azurecontainerapps.io pytest tests/ -v        (Linux / WSL)
    $env:API_URL="https://<mi-app>.azurecontainerapps.io"; pytest tests/ -v (PowerShell)

Cada prueba tiene un identificador (F = funcional, E = caso extremo, V = validación de entrada,
C = consistencia, L = latencia) que coincide con tests/casos_extremos.md y con el documento de validación.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
CARPETA_EJEMPLOS = Path(__file__).parent / "ejemplos"


def cargar_ejemplo(nombre):
    """Lee uno de los perfiles de tests/ejemplos (bajo, moderado o alto)."""
    return json.loads((CARPETA_EJEMPLOS / f"atleta_riesgo_{nombre}.json").read_text(encoding="utf-8"))


def predecir(datos):
    """Envía los datos a POST /predict y devuelve la respuesta completa."""
    return requests.post(f"{API_URL}/predict", json=datos, timeout=10)


# ---------------------------------------------------------------------------
# F. Pruebas funcionales: el flujo principal funciona
# ---------------------------------------------------------------------------
def test_F01_health_check():
    respuesta = requests.get(f"{API_URL}/", timeout=10)
    assert respuesta.status_code == 200
    assert respuesta.json()["mensaje"] == "API funcionando"


def test_F02_ficha_del_modelo():
    respuesta = requests.get(f"{API_URL}/modelo", timeout=10)
    assert respuesta.status_code == 200
    assert respuesta.json()["umbral"] == 0.17
    assert respuesta.json()["metricas_prueba"]["recall"] >= 0.70


@pytest.mark.parametrize("perfil, alerta, riesgo", [
    ("bajo", 0, "Bajo"),
    ("moderado", 1, "Moderado"),
    ("alto", 1, "Alto"),
])
def test_F03_prediccion_por_perfil(perfil, alerta, riesgo):
    respuesta = predecir(cargar_ejemplo(perfil))
    assert respuesta.status_code == 200
    resultado = respuesta.json()
    assert 0 <= resultado["probabilidad"] <= 1
    assert resultado["prediccion"] == alerta
    assert resultado["riesgo"] == riesgo


def test_F04_frontend_disponible():
    respuesta = requests.get(f"{API_URL}/app/", timeout=10)
    assert respuesta.status_code == 200
    assert "Calcular riesgo" in respuesta.text          # el formulario existe
    assert 'fetch("/predict"' in respuesta.text         # y consume el endpoint real


def test_F05_documentacion_swagger():
    assert requests.get(f"{API_URL}/docs", timeout=10).status_code == 200
    contrato = requests.get(f"{API_URL}/openapi.json", timeout=10).json()
    assert "/predict" in contrato["paths"]


# ---------------------------------------------------------------------------
# E. Casos extremos: comportamiento en situaciones difíciles
# ---------------------------------------------------------------------------
def test_E01_datos_opcionales_vacios():
    """Atleta sin edad, sin RPE y sin comentarios: el modelo imputa con la mediana."""
    datos = cargar_ejemplo("moderado")
    for campo in ["age", "mean_session_load_28d", "acwr3_freq", "acwr3_load", "acwr_freq", "acwr_load",
                  "prev_rpe", "prob_fatiga_prev_session", "prob_fatiga_last_note", "days_since_last_note"]:
        datos[campo] = None
    respuesta = predecir(datos)
    assert respuesta.status_code == 200
    assert 0 <= respuesta.json()["probabilidad"] <= 1


def test_E02_limites_minimos():
    """Atleta que regresa después de 1 año sin entrenar: todo en cero."""
    datos = cargar_ejemplo("bajo")
    datos.update({"days_since_prev_session": 365, "sessions_last_3d": 0, "sessions_last_7d": 0,
                  "sessions_last_28d": 0, "load_last_3d": 0, "load_last_7d": 0, "acwr3_freq": 0,
                  "acwr3_load": 0, "acwr_freq": 0, "acwr_load": 0, "prev_rpe": 1, "prev_session_load": 0,
                  "prev_has_note": 0, "notes_last_28d": 0})
    respuesta = predecir(datos)
    assert respuesta.status_code == 200
    assert respuesta.json()["riesgo"] == "Bajo"


def test_E03_limites_maximos():
    """Carga, ACWR, RPE y fatiga en el máximo permitido: debe dar riesgo alto."""
    datos = cargar_ejemplo("alto")
    datos.update({"sessions_last_3d": 3, "sessions_last_7d": 7, "load_last_3d": 10000, "load_last_7d": 20000,
                  "acwr3_freq": 4, "acwr3_load": 4, "acwr_freq": 4, "acwr_load": 4, "prev_rpe": 10,
                  "prev_session_load": 5000, "prob_fatiga_prev_session": 1, "prob_fatiga_last_note": 1})
    respuesta = predecir(datos)
    assert respuesta.status_code == 200
    assert respuesta.json()["riesgo"] == "Alto"


def test_E04_mas_carga_mas_riesgo():
    """Si solo sube la carga de los últimos 7 días, la probabilidad no debe bajar."""
    probabilidades = []
    for carga in [300, 1000, 2000, 3000]:
        datos = cargar_ejemplo("moderado")
        datos["load_last_7d"] = carga
        probabilidades.append(predecir(datos).json()["probabilidad"])
    assert probabilidades == sorted(probabilidades)


def test_E05_campos_extra_se_ignoran():
    """Un cliente envía un campo que el modelo no usa: la API lo ignora y responde."""
    datos = cargar_ejemplo("moderado")
    datos["nombre_atleta"] = "Ana"
    respuesta = predecir(datos)
    assert respuesta.status_code == 200


# ---------------------------------------------------------------------------
# V. Validación de entrada: la API rechaza datos inválidos con 422
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("campo, valor", [
    ("prev_rpe", 11),               # V01: RPE fuera de la escala 1 a 10
    ("level", "pro"),               # V02: categoría que no existe
    ("load_last_7d", "mucha"),      # V03: texto en lugar de número
    ("prob_fatiga_last_note", 1.5), # V04: probabilidad mayor a 1
    ("sessions_last_7d", -1),       # V05: número negativo
    ("age", 120),                   # V06: edad fuera de rango
])
def test_V01_valores_invalidos(campo, valor):
    datos = cargar_ejemplo("moderado")
    datos[campo] = valor
    respuesta = predecir(datos)
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"][-1] == campo     # el error señala el campo correcto


def test_V07_campo_obligatorio_faltante():
    datos = cargar_ejemplo("moderado")
    del datos["load_last_7d"]
    respuesta = predecir(datos)
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["type"] == "missing"


def test_V08_json_mal_formado():
    respuesta = requests.post(f"{API_URL}/predict", data="{esto no es json",
                              headers={"Content-Type": "application/json"}, timeout=10)
    assert respuesta.status_code == 422


def test_V09_metodo_incorrecto():
    """GET a /predict no está permitido (solo POST)."""
    assert requests.get(f"{API_URL}/predict", timeout=10).status_code == 405


# ---------------------------------------------------------------------------
# C. Consistencia: misma entrada, misma respuesta
# ---------------------------------------------------------------------------
def test_C01_misma_entrada_misma_salida():
    datos = cargar_ejemplo("moderado")
    probabilidades = {predecir(datos).json()["probabilidad"] for _ in range(20)}
    assert len(probabilidades) == 1


# ---------------------------------------------------------------------------
# L. Latencia básica: tiempo por petición
# ---------------------------------------------------------------------------
def test_L01_latencia():
    """100 peticiones seguidas: la mediana debe ser menor a 100 ms y el percentil 95 menor a 300 ms."""
    datos = cargar_ejemplo("moderado")
    tiempos = []
    for _ in range(100):
        inicio = time.perf_counter()
        predecir(datos)
        tiempos.append((time.perf_counter() - inicio) * 1000)   # milisegundos
    assert np.median(tiempos) < 100
    assert np.percentile(tiempos, 95) < 300
