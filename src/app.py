"""
app.py - API de WODPredict-AI (Fase II).

Expone el modelo de sobrecarga como un servicio REST con FastAPI:

    GET  /          -> health check: ¿el servicio responde?
    GET  /modelo    -> ficha del modelo (versión, umbral, métricas)
    POST /predict   -> recibe los datos de un atleta y devuelve el riesgo de sobrecarga
    GET  /docs      -> documentación interactiva (Swagger) que genera FastAPI
    GET  /app       -> formulario web (frontend) que consume /predict

El modelo se carga con src/model_loader.py (carpeta model/).

Para ejecutarla sin Docker (en la carpeta del repositorio):

    uvicorn src.app:app --reload --port 8000
"""
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.model_loader import cargar_modelo, columnas_del_modelo

# ---------------------------------------------------------------------------
# 1. Cargar el modelo y su ficha técnica UNA sola vez, al iniciar la API (ver src/model_loader.py)
# ---------------------------------------------------------------------------
CARPETA_RAIZ = Path(__file__).resolve().parent.parent
modelo, metadata = cargar_modelo()
UMBRAL = metadata["umbral"]
COLUMNAS = columnas_del_modelo(metadata)


# ---------------------------------------------------------------------------
# 2. Contrato de entrada: qué datos recibe /predict y en qué rango son válidos
# ---------------------------------------------------------------------------
# Field(ge=..., le=...) define el mínimo y el máximo. Si un dato está fuera de rango o falta,
# FastAPI responde 422 con el detalle del error y el modelo no se ejecuta.
# Optional[...] = None significa que el dato puede venir vacío; el modelo lo imputa con la mediana.
class DatosAtleta(BaseModel):
    # Perfil del atleta
    level: Literal["novato", "intermedio", "élite"] = Field(description="Nivel del atleta")
    sex: Literal["F", "M"] = Field(description="Sexo")
    language: Literal["es", "en"] = Field(description="Idioma de sus comentarios")
    age: Optional[float] = Field(None, ge=16, le=80, description="Edad en años")
    days_since_first_session: int = Field(ge=0, le=3650, description="Días desde su primera sesión")

    # Frecuencia de entrenamiento
    days_since_prev_session: int = Field(ge=1, le=365, description="Días desde la sesión anterior")
    sessions_last_3d: int = Field(ge=0, le=3, description="Sesiones en los últimos 3 días")
    sessions_last_7d: int = Field(ge=0, le=7, description="Sesiones en los últimos 7 días")
    sessions_last_28d: int = Field(ge=0, le=28, description="Sesiones en los últimos 28 días")

    # Carga de entrenamiento (carga Foster = RPE x minutos)
    load_last_3d: float = Field(ge=0, le=10000, description="Carga de los últimos 3 días")
    load_last_7d: float = Field(ge=0, le=20000, description="Carga de los últimos 7 días")
    mean_session_load_28d: Optional[float] = Field(None, ge=0, le=5000, description="Carga promedio por sesión en 28 días")
    acwr3_freq: Optional[float] = Field(None, ge=0, le=4, description="ACWR de frecuencia (3 días)")
    acwr3_load: Optional[float] = Field(None, ge=0, le=4, description="ACWR de carga (3 días)")
    acwr_freq: Optional[float] = Field(None, ge=0, le=4, description="ACWR de frecuencia (7 días)")
    acwr_load: Optional[float] = Field(None, ge=0, le=4, description="ACWR de carga (7 días)")

    # Sesión anterior
    prev_rpe: Optional[float] = Field(None, ge=1, le=10, description="RPE de la sesión anterior (1 a 10)")
    prev_session_load: float = Field(ge=0, le=5000, description="Carga de la sesión anterior")
    prev_has_note: int = Field(ge=0, le=1, description="1 si dejó comentario en la sesión anterior")

    # Comentarios del atleta (probabilidad de fatiga calculada a partir de sus notas)
    prob_fatiga_prev_session: Optional[float] = Field(None, ge=0, le=1, description="Prob. de fatiga del comentario anterior")
    prob_fatiga_last_note: Optional[float] = Field(None, ge=0, le=1, description="Prob. de fatiga del último comentario")
    days_since_last_note: Optional[float] = Field(None, ge=0, le=3650, description="Días desde el último comentario")
    notes_last_28d: int = Field(ge=0, le=60, description="Comentarios en los últimos 28 días")

    # Ejemplo que aparece en /docs para probar con un clic (perfil de riesgo moderado).
    model_config = {"json_schema_extra": {"example": {
        "level": "élite", "sex": "F", "language": "es", "age": 31, "days_since_first_session": 365,
        "days_since_prev_session": 1, "sessions_last_3d": 2, "sessions_last_7d": 5, "sessions_last_28d": 18,
        "load_last_3d": 760, "load_last_7d": 1650, "mean_session_load_28d": 310,
        "acwr3_freq": 1.2, "acwr3_load": 1.3, "acwr_freq": 1.1, "acwr_load": 1.25,
        "prev_rpe": 8, "prev_session_load": 420, "prev_has_note": 1,
        "prob_fatiga_prev_session": 0.70, "prob_fatiga_last_note": 0.70,
        "days_since_last_note": 1, "notes_last_28d": 8}}}


# ---------------------------------------------------------------------------
# 3. Contrato de salida: qué devuelve /predict
# ---------------------------------------------------------------------------
class Prediccion(BaseModel):
    probabilidad: float      # probabilidad de sobrecarga al día siguiente (0 a 1)
    umbral: float            # si probabilidad >= umbral, hay alerta
    prediccion: int          # 1 = alerta de sobrecarga, 0 = sin alerta
    riesgo: str              # texto para el coach
    recomendacion: str
    version_modelo: str


def clasificar_riesgo(probabilidad):
    """Convierte la probabilidad en un mensaje para el coach."""
    if probabilidad >= 0.50:
        return "Alto", "Bajar la carga o programar descanso activo; revisar con el atleta."
    if probabilidad >= UMBRAL:
        return "Moderado", "Revisar con el atleta antes de la sesión y considerar escalar el WOD."
    return "Bajo", "Entrenar según lo planeado."


# ---------------------------------------------------------------------------
# 4. La aplicación y sus endpoints
# ---------------------------------------------------------------------------
app = FastAPI(
    title="WODPredict-AI API",
    description="Predice si un atleta de CrossFit tendrá sobrecarga al día siguiente (UnderCover Fitness).",
    version=metadata["version"],
)


@app.get("/")
def health_check():
    """Health check: confirma que la API y el modelo están cargados."""
    return {"mensaje": "API funcionando", "modelo": metadata["nombre"], "version": metadata["version"]}


@app.get("/modelo")
def info_modelo():
    """Ficha técnica del modelo: umbral, métricas en prueba, dataset y versiones."""
    return metadata


@app.post("/predict", response_model=Prediccion)
def predecir(datos: DatosAtleta):
    """Recibe los datos de UN atleta y devuelve su riesgo de sobrecarga al día siguiente."""
    # 1) Pasar el JSON a una tabla de una fila con las columnas en el mismo orden del entrenamiento.
    fila = pd.DataFrame([datos.model_dump()])[COLUMNAS]
    # Los vacíos (None) se convierten en NaN para que el pipeline los impute con la mediana.
    fila[metadata["variables_numericas"]] = fila[metadata["variables_numericas"]].astype(float)

    # 2) Probabilidad de la clase 1 (sobrecarga) y decisión con el umbral de la Actividad 6.
    probabilidad = float(modelo.predict_proba(fila)[0, 1])
    riesgo, recomendacion = clasificar_riesgo(probabilidad)

    return Prediccion(probabilidad=round(probabilidad, 4), umbral=UMBRAL,
                      prediccion=int(probabilidad >= UMBRAL), riesgo=riesgo,
                      recomendacion=recomendacion, version_modelo=metadata["version"])


# ---------------------------------------------------------------------------
# 5. Frontend: el formulario HTML se sirve desde la misma API en /app
# ---------------------------------------------------------------------------
app.mount("/app", StaticFiles(directory=CARPETA_RAIZ / "frontend", html=True), name="frontend")


@app.get("/formulario", include_in_schema=False)
def ir_al_formulario():
    """Atajo: /formulario lleva al frontend."""
    return RedirectResponse(url="/app/")
