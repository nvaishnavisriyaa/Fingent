# Fingent — single-image deploy: FastAPI backend that also serves the SPA.
FROM python:3.12-slim AS base

# System deps for KYC document intelligence (image OCR + scanned-PDF rendering).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FINGENT_DB=/data/fingent.db

WORKDIR /app

# Install Python deps first for layer caching.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# App code.
COPY backend ./backend
COPY frontend ./frontend

# Durable data lives on a volume; run as a non-root user.
RUN useradd --create-home --uid 10001 fingent \
    && mkdir -p /data && chown -R fingent:fingent /data /app
USER fingent
VOLUME ["/data"]

EXPOSE 8000

# Container-native healthcheck hits the app's /healthz.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

WORKDIR /app/backend
CMD ["uvicorn", "fingent.app:app", "--host", "0.0.0.0", "--port", "8000"]
