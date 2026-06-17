# Imagen base liviana con Python 3.12 (la misma versión que usás local)
FROM python:3.12-slim

# Logs en vivo, sin .pyc, sin caché de pip (imagen más chica)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) Dependencias primero: si no cambian, Docker reusa esta capa (build más rápido)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Copiar el código y el corpus (lo que NO entra está en .dockerignore:
#    .env, .venv, chroma_db, etc. La API key NUNCA va en la imagen)
COPY . .

# 3) Construir la base vectorial DENTRO de la imagen.
#    Esto también descarga el modelo de embeddings (~90MB) y lo deja cacheado,
#    así la primera consulta en producción ya es rápida.
RUN python cargar_chroma.py

# 4) App Runner enruta el tráfico al puerto 8080 por defecto
ENV PORT=8080
EXPOSE 8080

# 5) Levantar el server. host 0.0.0.0 = aceptar tráfico de afuera del contenedor
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
