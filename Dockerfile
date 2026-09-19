FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src
ENV INDEX_DIR=/app/data/index

EXPOSE 7860
# Shell form (not exec-array) so $PORT expands: HF Spaces always use 7860,
# Render/Railway/etc inject their own PORT at runtime.
CMD uvicorn aptino_claims.api.main:app --host 0.0.0.0 --port ${PORT:-7860}
