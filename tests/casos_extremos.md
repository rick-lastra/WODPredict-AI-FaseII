# Casos extremos y validación de entrada: WODPredict-AI (Fase II)

Esta tabla documenta cómo se comporta la API con **situaciones difíciles** (datos vacíos, valores en el mínimo o en el máximo permitido, datos inválidos) y no solo con el caso "bonito". Cada fila corresponde a una prueba automatizada de [`test_api.py`](test_api.py) con el mismo identificador, y al documento de validación (`docs/validacion_pruebas.pdf`).

**Cómo reproducir:** con el contenedor corriendo (`docker run -d --name wodpredict -p 8000:8000 wodpredict-api:1.0`):

```powershell
pytest tests/ -v
```

**Perfiles base:** los casos parten de los perfiles de [`ejemplos/`](ejemplos/) (`atleta_riesgo_bajo.json`, `atleta_riesgo_moderado.json` y `atleta_riesgo_alto.json`) y cambian solo los campos indicados. **Umbral de alerta:** 0.17. **Niveles:** Bajo (< 0.17), Moderado (0.17 a 0.49) y Alto (≥ 0.50).

## E. Casos extremos (la API debe responder 200 con una predicción lógica)

| ID | Entrada | Esperado | Real | Conclusión |
|---|---|---|---|---|
| E01 | Atleta nuevo sin edad, RPE, ACWR ni comentarios (10 campos opcionales en `null`) | 200 y probabilidad válida | 200, p = 0.229, Moderado | Pasa. El pipeline imputa los vacíos con la mediana del entrenamiento; un atleta sin historial completo también recibe predicción. |
| E02 | Mínimos: regresa después de 365 días, sesiones y carga en 0, ACWR 0, RPE 1 | 200 y riesgo Bajo | 200, p < 0.001, Bajo | Pasa. Sin carga reciente no hay riesgo de sobrecarga, sin errores numéricos en el límite inferior. |
| E03 | Máximos: 7 sesiones en 7 días, carga 20,000, ACWR 4, RPE 10 y fatiga 1 | 200 y riesgo Alto | 200, p = 1.000, Alto | Pasa. En el límite superior el modelo satura en riesgo Alto sin desbordarse. |
| E04 | Solo sube la carga de 7 días: 300, 1,000, 2,000 y 3,000 | La probabilidad no baja | 0.054 → 0.140 → 0.423 → 0.768 | Pasa. A más carga, más riesgo; la alerta (0.17) se activa entre 1,000 y 2,000 para este perfil. |
| E05 | Campo extra que el modelo no usa (`"nombre_atleta": "Ana"`) | 200 y el campo se ignora | 200, p = 0.303, Moderado | Pasa. Un cliente con campos de más no rompe la API; el resultado es igual al perfil sin el campo. |

## V. Validación de entrada (la API debe rechazar con 422 y señalar el campo, sin ejecutar el modelo)

| ID | Entrada | Esperado | Real | Conclusión |
|---|---|---|---|---|
| V01 | `prev_rpe = 11` (escala 1 a 10) | 422 en `prev_rpe` | 422, "less than or equal to 10" | Pasa |
| V02 | `level = "pro"` (no existe) | 422 en `level` | 422, lista de valores permitidos | Pasa |
| V03 | `load_last_7d = "mucha"` (texto) | 422 en `load_last_7d` | 422, "valid number" | Pasa |
| V04 | `prob_fatiga_last_note = 1.5` (probabilidad > 1) | 422 en `prob_fatiga_last_note` | 422, "less than or equal to 1" | Pasa |
| V05 | `sessions_last_7d = -1` (negativo) | 422 en `sessions_last_7d` | 422, "greater than or equal to 0" | Pasa |
| V06 | `age = 120` (fuera de 16 a 80) | 422 en `age` | 422, "less than or equal to 80" | Pasa |
| V07 | Falta `load_last_7d` (obligatorio) | 422 tipo `missing` | 422, "Field required" | Pasa |
| V08 | JSON mal formado | 422 | 422, `json_invalid` | Pasa |
| V09 | `GET /predict` (solo acepta POST) | 405 | 405, "Method Not Allowed" | Pasa |

## C y L. Consistencia y latencia

| ID | Entrada | Esperado | Real | Conclusión |
|---|---|---|---|---|
| C01 | El perfil moderado enviado 20 veces | Siempre la misma probabilidad | 0.3025 en las 20 peticiones | Pasa. La inferencia es reproducible. |
| L01 | 200 peticiones seguidas a `POST /predict` (notebook, punto 8) | Mediana < 100 ms y p95 < 300 ms | Mediana 17.3 ms, p95 22.4 ms | Pasa. Tiempo adecuado para consultar antes de la sesión. |

## Análisis

- **Datos incompletos:** el caso más probable en el box es un atleta nuevo o que no dejó comentarios (E01). La imputación vive dentro del pipeline guardado, así que la API se comporta igual que en el entrenamiento.
- **Límites del contrato:** los rangos de Pydantic (`src/app.py`) protegen al modelo. Todo lo que está fuera de rango se rechaza **antes** de predecir (V01 a V09), y lo que está dentro, incluso en el extremo, produce una predicción lógica (E02 y E03).
- **Comportamiento del modelo:** la probabilidad crece con la carga reciente (E04), que es lo que espera un coach; el modelo recomienda y el coach decide.
- **Limitación:** el modelo se entrenó con datos sintéticos; estos casos verifican el comportamiento técnico, no la exactitud con atletas reales (pendiente del piloto con datos de Wodify).

> Los valores reales provienen de la ejecución contra el contenedor Docker local. Las latencias dependen del equipo; la ejecución en el equipo del autor se registra en el notebook (punto 8).
