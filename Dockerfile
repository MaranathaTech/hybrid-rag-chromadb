# syntax=docker/dockerfile:1

# ---- Build stage: install deps, run the test suite ----------------------------
FROM python:3.12-slim AS build

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# CPU-only torch. The default wheel pulls ~2GB of CUDA libraries that are dead
# weight in this image and slow every pull.
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY pytest.ini ./

# The build IS the CI gate: a failing unit test fails the image, so broken
# retrieval logic cannot ship. Integration tests are excluded here because
# they need a live Chroma service, which is not reachable at build time —
# `./run.sh --test` runs those against the deployed cluster instead.
RUN python -m pytest tests/unit -q

# Bake the embedding model into the image. Without this, the first pod to
# start downloads ~90MB from HuggingFace before it can answer, which makes
# readiness flaky on a slow or offline network.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ---- Runtime stage ------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HUB_OFFLINE=1 \
    PORT=8080

# Non-root: nothing here needs to write outside the model cache.
RUN useradd --create-home --uid 10001 appuser

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /root/.cache/huggingface /home/appuser/.cache/huggingface
COPY src/ ./src/

RUN chown -R appuser:appuser /home/appuser/.cache
USER appuser

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "hybrid_rag.api:app", \
     "--host", "0.0.0.0", "--port", "8080"]
