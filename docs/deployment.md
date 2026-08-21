# Deployment

## Local

```bash
poetry run hf-lab serve --model-dir artifacts/models/support-triage --host 0.0.0.0 --port 8000
```

## Docker

The image ships **no model**. It declares `/models` as a volume and expects a trained
artifact to be mounted there — a model directory is the deployment input, not part of
the build.

```bash
docker build -t hf-finetuning-lab .
docker run -p 8000:8000 \
  -v "$(pwd)/artifacts/models/support-triage:/models:ro" \
  hf-finetuning-lab
```

`docker-compose.yml` does the same with a bind mount and the environment below.

The image runs as an unprivileged user (`serve`, uid 10001) and is built in two stages,
so the compiler toolchain used to install dependencies is not present at runtime.

## Configuration

Every setting is resolved by `ServingConfig.from_env`. A CLI flag always beats the
environment, so the same image works from flags locally and from Compose in deployment.

| Variable | Default | Meaning |
| --- | --- | --- |
| `HF_LAB_MODEL_DIR` | `/models` in the image | Model directory to serve. Required. |
| `HF_LAB_MODEL_VERSION` | unset | Version string echoed in health responses and request logs. |
| `HF_LAB_ENABLE_METRICS` | `false` | Mount `/metrics` and instrument requests. Needs the `metrics` extra. |
| `HF_LAB_HOST` | `127.0.0.1` | Bind address. |
| `HF_LAB_PORT` | `8000` | Bind port. |
| `HF_LAB_MAX_TEXTS_PER_REQUEST` | `256` | Reject batches larger than this. |
| `HF_LAB_MAX_CHARS_PER_TEXT` | `20000` | Reject any single text longer than this. |

Booleans accept `1/0`, `true/false`, `yes/no`, `on/off`. An unparseable value fails at
startup rather than being silently treated as false.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health`, `GET /health/live` | Liveness. Answers as soon as the process is up. |
| `GET /health/ready` | Readiness. 503 until the predictor has loaded. |
| `POST /predict` | Classify a batch of texts. |
| `GET /metrics` | Prometheus metrics. Only mounted when metrics are enabled. |

```json
{
  "texts": ["My account login is blocked"]
}
```

## Startup behaviour

The predictor is built during the lifespan startup, then primed with a warm-up request so
the first real caller does not pay the cold-start cost.

Warm-up is a latency optimisation, not a readiness gate: if the predictor loaded but
warm-up failed, the service is still ready. Readiness is keyed on the predictor being
available.

## Failure modes

| Symptom | Cause |
| --- | --- |
| `/health/ready` returns 503 with `not_ready` | The predictor failed to load — usually no model mounted at `HF_LAB_MODEL_DIR`. |
| `/predict` returns 413 | A single text exceeded `HF_LAB_MAX_CHARS_PER_TEXT`. |
| `/predict` returns 422 | The batch exceeded `HF_LAB_MAX_TEXTS_PER_REQUEST`, or the payload was malformed. |
| `/metrics` returns 404 | Metrics are disabled. Set `HF_LAB_ENABLE_METRICS=true`. |
| Startup fails on an import of `prometheus_client` | Install the extra: `poetry install --extras metrics`. |

Readiness deliberately reports an opaque message rather than the underlying exception.
The detail is written to the server log; the response is reachable by any caller and
`repr(exc)` can carry local paths and configuration values.
