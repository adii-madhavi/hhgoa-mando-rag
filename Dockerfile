# CPU-only image. torch's CPU wheel is ~200 MB vs ~2.5 GB for the CUDA build,
# and every measurement in this project was taken on CPU anyway.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    OMP_NUM_THREADS=8

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && rm -rf /var/lib/apt/lists/*

# Install torch from the CPU index first so the later resolve cannot pull CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image. Downloading it at container start
# would put a multi-hundred-MB fetch inside the first request's latency.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small')"

COPY app/ app/
COPY ingestion/ ingestion/
COPY evaluation/ evaluation/
COPY frontend/ frontend/
COPY data/ data/
# data/ (index + processed corpus) is mounted, not copied: it is build output,
# not source, and baking a multi-GB index into the image is wasteful.

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
