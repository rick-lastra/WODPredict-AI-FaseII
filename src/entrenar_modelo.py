"""
entrenar_modelo.py - Entrena el modelo final y lo guarda como artefacto para la API (Fase II).

Repite el modelo seleccionado en las Actividades 5 y 6 (regresión logística, C = 0.1) con los
mismos datos (Actividad_5@v1.0) y la misma separación por atleta. Guarda dos archivos en model/:

    model/model.pkl        -> pipeline completo (preprocesamiento + modelo), guardado con joblib
    model/metadata.json    -> umbral, variables, métricas y versiones

Uso desde la terminal (en la carpeta del repositorio):

    python src/entrenar_modelo.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------
CARPETA_RAIZ = Path(__file__).resolve().parent.parent
CARPETA_MODELO = CARPETA_RAIZ / "model"
ARCHIVO_MODELO = CARPETA_MODELO / "model.pkl"
ARCHIVO_METADATA = CARPETA_MODELO / "metadata.json"

# Datos limpios de la Etapa 2, fijados en el tag v1.0 (siempre es el mismo archivo).
URL_DATOS = "https://raw.githubusercontent.com/rick-lastra/WODPredict-AI_etapa-2_preparacion-datos/v1.0/datos/datos_limp/sesiones_limpias.csv"
DATASET = "rick-lastra/wodpredict-synth@v1.0 (limpio: Actividad_5@v1.0)"

SEMILLA = 42
VERSION_MODELO = "1.0.0"

# Umbral elegido en la Etapa 3: el más alto con recall >= 0.70 (fuera del fold, solo entrenamiento).
UMBRAL = 0.17

# Variables de entrada: todas se conocen ANTES de la sesión (las mismas de las Actividades 5 y 6).
NUMERICAS = [
    "age", "days_since_prev_session", "sessions_last_3d", "sessions_last_7d", "sessions_last_28d",
    "load_last_3d", "load_last_7d", "mean_session_load_28d",
    "acwr3_freq", "acwr3_load", "acwr_freq", "acwr_load",
    "days_since_first_session", "prev_rpe", "prev_session_load", "prev_has_note",
    "prob_fatiga_prev_session", "prob_fatiga_last_note", "days_since_last_note", "notes_last_28d",
]
CATEGORICAS = ["level", "sex", "language"]
ETIQUETA = "overreaching_label"


def crear_modelo():
    """Pipeline: imputar vacíos con la mediana, escalar, One-Hot de categorías y regresión logística."""
    numericas = Pipeline([("imputar", SimpleImputer(strategy="median")),
                          ("escalar", StandardScaler())])
    preprocesador = ColumnTransformer([("num", numericas, NUMERICAS),
                                       ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])
    return Pipeline([("prep", preprocesador),
                     ("modelo", LogisticRegression(C=0.1, max_iter=3000, random_state=SEMILLA))])


def metricas(y_real, probabilidad, umbral):
    """Exactitud, precisión, recall, F1 y ROC-AUC con el umbral indicado."""
    prediccion = (probabilidad >= umbral).astype(int)
    return {"exactitud": round(accuracy_score(y_real, prediccion), 3),
            "precision": round(precision_score(y_real, prediccion), 3),
            "recall": round(recall_score(y_real, prediccion), 3),
            "f1": round(f1_score(y_real, prediccion), 3),
            "roc_auc": round(float(roc_auc_score(y_real, probabilidad)), 3)}


def main():
    # 1. Cargar los datos limpios.
    datos = pd.read_csv(URL_DATOS)
    X = datos[NUMERICAS + CATEGORICAS]
    y = datos[ETIQUETA]
    print(f"Datos: {len(datos):,} sesiones de {datos['athlete_id'].nunique()} atletas")

    # 2. Separación 80/20 por atleta (igual que en las Actividades 5 y 6).
    separador = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMILLA)
    idx_train, idx_test = next(separador.split(X, y, groups=datos["athlete_id"]))
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    # 3. Entrenar y medir en prueba (atletas que el modelo nunca vio).
    modelo = crear_modelo()
    modelo.fit(X_train, y_train)
    probabilidad = modelo.predict_proba(X_test)[:, 1]
    resultado = metricas(y_test, probabilidad, UMBRAL)
    print(f"Métricas en prueba con umbral {UMBRAL}: {resultado}")

    # 4. Guardar el modelo (artefacto model.pkl, con joblib) y su ficha técnica (metadata).
    CARPETA_MODELO.mkdir(exist_ok=True)
    joblib.dump(modelo, ARCHIVO_MODELO)
    metadata = {
        "nombre": "WODPredict-AI - sobrecarga al día siguiente",
        "version": VERSION_MODELO,
        "fecha_entrenamiento": date.today().isoformat(),
        "algoritmo": "Regresión logística (C = 0.1) en un Pipeline de scikit-learn",
        "umbral": UMBRAL,
        "dataset": DATASET,
        "sesiones_entrenamiento": int(len(X_train)),
        "sesiones_prueba": int(len(X_test)),
        "variables_numericas": NUMERICAS,
        "variables_categoricas": CATEGORICAS,
        "metricas_prueba": resultado,
        "versiones": {"python": sys.version.split()[0], "numpy": np.__version__,
                      "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                      "joblib": joblib.__version__},
    }
    ARCHIVO_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo guardado en {ARCHIVO_MODELO.relative_to(CARPETA_RAIZ)}")
    print(f"Metadata guardada en {ARCHIVO_METADATA.relative_to(CARPETA_RAIZ)}")


if __name__ == "__main__":
    main()
