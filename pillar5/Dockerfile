# The service as a container: same environment on your laptop, a server or the cloud.
#   docker compose up --build        (see docker-compose.yml)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# dependencies first: this layer is cached and only rebuilt when requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# never run as root inside the container
RUN useradd --create-home app && mkdir -p /srv/logs && chown app /srv/logs
USER app

# download the embedding model at BUILD time, not on the first user request
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warm up'])"

COPY --chown=app app ./app
COPY --chown=app data/docs ./data/docs

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
