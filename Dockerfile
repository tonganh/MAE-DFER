ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cpu

FROM python:3.12-slim-bookworm

ARG PYTORCH_INDEX

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /tmp/requirements-docker.txt

RUN pip install torch torchvision --index-url "${PYTORCH_INDEX}" \
    && pip install -r /tmp/requirements-docker.txt

COPY pyproject.toml /app/pyproject.toml
COPY mae_dfer /app/mae_dfer
COPY api_server.py infer_video.py /app/

EXPOSE 8000

ENV API_HOST=0.0.0.0 \
    API_PORT=8000

CMD ["sh", "-c", "exec python -m uvicorn api_server:app --host \"${API_HOST}\" --port \"${API_PORT}\" --workers 1 --ws-max-size ${WS_MAX_SIZE:-52428800}"]
