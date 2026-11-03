# Collector image for Cloud Run Jobs.
# Slim base: the job is network- and memory-bound, not CPU-bound.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata

WORKDIR /app

# Dependencies first so code changes do not invalidate the layer.
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
      pandas numpy pyarrow pyyaml requests websockets pyotp \
      google-cloud-storage google-cloud-secret-manager

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

ENV PYTHONPATH=/app/src

# Cloud Run's filesystem is an in-memory tmpfs, so parquet is staged here and
# uploaded to GCS continuously rather than accumulating.
ENV MCUBE_OUT=/tmp/live

ENTRYPOINT ["python", "scripts/run_collector.py"]
