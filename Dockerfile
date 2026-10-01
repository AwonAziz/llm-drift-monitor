FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LDM_DATA_DIR=/app/data \
    LDM_ARTIFACT_DIR=/app/artifacts \
    LDM_EMBEDDING_BACKEND=hashing \
    LDM_JUDGE_PROVIDER=mock \
    OLLAMA_HOST=http://host.docker.internal:11434

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Core dependencies first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY dashboard/ dashboard/
COPY tests/ tests/

RUN mkdir -p /app/data /app/artifacts /app/mlruns

# Run as a non-root user; the data and artifact dirs are the only writable paths.
RUN useradd --create-home --uid 10001 monitor \
 && chown -R monitor:monitor /app
USER monitor

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Default: run the offline demo so `docker run` produces something inspectable.
# Override for the API:   docker run ... llm-drift-monitor uvicorn src.api.server:app --host 0.0.0.0 --port 8000
# Override for the UI:    docker run ... llm-drift-monitor streamlit run dashboard/app.py --server.address 0.0.0.0
CMD ["python", "scripts/demo_shift.py", "--fast", "--reset"]
