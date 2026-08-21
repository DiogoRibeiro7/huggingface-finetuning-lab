# Build stage: resolve dependencies with the compiler toolchain, then discard it.
FROM python:3.12-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src

# --no-root first so the dependency layer caches independently of source edits.
RUN pip install --no-cache-dir poetry==2.2.1 && \
    poetry install --only main --no-interaction --no-ansi --no-root && \
    poetry install --only main --no-interaction --no-ansi

# Runtime stage: interpreter, installed packages and source only — no compiler.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_LAB_MODEL_DIR=/models \
    HF_HOME=/tmp/huggingface

WORKDIR /app

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app/src ./src

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 serve && chown -R serve:serve /app
USER serve

EXPOSE 8000

# The image deliberately ships no model. Mount one at HF_LAB_MODEL_DIR
# (see docker-compose.yml); the server fails fast when it is absent.
VOLUME ["/models"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready', timeout=2)" || exit 1

CMD ["hf-lab", "serve", "--host", "0.0.0.0", "--port", "8000"]
