# Dockerfile - Imagen de la API de WODPredict-AI (Fase II)
# Construir:  docker build -t wodpredict-api:1.0 .
# Ejecutar:   docker run -d --name wodpredict -p 8000:8000 wodpredict-api:1.0
# Probar:     http://localhost:8000/docs  y  http://localhost:8000/app

# 1. Imagen base: Python 3.12 en versión ligera (slim).
FROM python:3.12-slim

# 2. Python sin archivos .pyc y con salida inmediata en los logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Carpeta de trabajo dentro del contenedor.
WORKDIR /app

# 4. Instalar dependencias primero (Docker reutiliza esta capa si requirements.txt no cambia).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar solo lo necesario para la inferencia: código de la API, modelo y frontend.
COPY src/app.py src/model_loader.py src/
COPY model/ model/
COPY frontend/ frontend/

# 6. Usuario sin privilegios de administrador (buena práctica de seguridad).
RUN useradd --create-home appuser
USER appuser

# 7. Puerto que usa la API. La variable PORT permite cambiarlo en la nube (Azure usa 8000 o la que se indique).
ENV PORT=8000
EXPOSE 8000

# 8. Revisión de salud: Docker marca el contenedor como "healthy" si GET / responde.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/')" || exit 1

# 9. Comando al iniciar el contenedor: servidor uvicorn escuchando en todas las interfaces.
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
