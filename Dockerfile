FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEVICE=cpu \
    OMP_NUM_THREADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 inference \
    && useradd --uid 10001 --gid inference --no-create-home inference

COPY requirements-api.txt requirements-serving.txt ./
RUN python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements-serving.txt

COPY --chown=inference:inference src/ ./src/
COPY --chown=inference:inference reports/validation_thresholds.json ./reports/validation_thresholds.json
COPY --chown=inference:inference reports/mobilenet_v3_small_thresholds.json ./reports/mobilenet_v3_small_thresholds.json

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).close()"]

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
