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

# This is a CPU serving image. Installing straight from the lock would pull the
# default Linux `torch` wheel, which bundles CUDA: 17 nvidia/triton packages and
# roughly 8 GB that can never be used without a GPU.
#
# So: export the locked versions, drop the GPU packages, install the rest from
# PyPI, and take torch from the CPU wheel index. Versions still come from the
# lock, so the image matches what CI resolved. The lock itself stays
# GPU-capable, because `poetry install` is also how someone sets up training.
RUN pip install --no-cache-dir poetry==2.2.1 poetry-plugin-export==1.9.0 && \
    poetry export --only main --without-hashes --format requirements.txt --output /tmp/requirements.txt && \
    grep -viE '^(torch[=<>~ ]|triton[=<>~ ]|nvidia-)' /tmp/requirements.txt > /tmp/requirements-cpu.txt && \
    # CPU torch first: accelerate, peft and transformers all require torch, so
    # installing them first would pull the CUDA wheel to satisfy it.
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0" && \
    pip install --no-cache-dir -r /tmp/requirements-cpu.txt && \
    pip install --no-cache-dir --no-deps . && \
    # Fail the build rather than ship a GPU stack that can never be used here.
    python -c "import torch, hf_finetuning_lab; assert torch.__version__.endswith('+cpu'), torch.__version__" && \
    ! pip list --format=freeze | grep -qiE '^(nvidia-|triton)'

# Runtime stage: interpreter and installed packages only — no compiler.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_LAB_MODEL_DIR=/models \
    HF_HOME=/tmp/huggingface

WORKDIR /app

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 serve && chown -R serve:serve /app
USER serve

EXPOSE 8000

# The image deliberately ships no model. Mount one at HF_LAB_MODEL_DIR
# (see docker-compose.yml); readiness stays 503 until one is present.
VOLUME ["/models"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready', timeout=2)" || exit 1

CMD ["hf-lab", "serve", "--host", "0.0.0.0", "--port", "8000"]
