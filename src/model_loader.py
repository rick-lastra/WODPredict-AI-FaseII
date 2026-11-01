"""
model_loader.py - Carga el modelo de WODPredict-AI y su ficha técnica (Fase II).

Aquí se define DÓNDE está el modelo y CÓMO se carga. La API (src/app.py) y el notebook lo usan.

    model/model.pkl        -> pipeline entrenado (imputación + escalado + One-Hot + regresión logística)
    model/metadata.json    -> versión, umbral de decisión (0.17), variables, métricas y versiones

El modelo se entrena con src/entrenar_modelo.py. Este archivo solo lo CARGA, nunca vuelve a entrenar.
"""
import json
from pathlib import Path

import joblib

# Rutas del artefacto (carpeta model/ en la raíz del repositorio).
CARPETA_MODELO = Path(__file__).resolve().parent.parent / "model"
RUTA_MODELO = CARPETA_MODELO / "model.pkl"
RUTA_METADATA = CARPETA_MODELO / "metadata.json"


def cargar_modelo():
    """Devuelve el modelo entrenado y su ficha técnica (metadata)."""
    # Si falta el archivo, se explica qué hacer en lugar de mostrar un error difícil de entender.
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(f"No se encontró {RUTA_MODELO}. Ejecuta primero: python src/entrenar_modelo.py")

    modelo = joblib.load(RUTA_MODELO)
    metadata = json.loads(RUTA_METADATA.read_text(encoding="utf-8"))
    return modelo, metadata


def columnas_del_modelo(metadata):
    """Lista de variables en el mismo orden que se usó para entrenar."""
    return metadata["variables_numericas"] + metadata["variables_categoricas"]
