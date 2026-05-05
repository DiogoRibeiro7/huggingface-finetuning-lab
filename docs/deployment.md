# Deployment

The simplest local deployment is:

```bash
poetry run hf-lab serve --model-dir artifacts/models/support-triage --host 0.0.0.0 --port 8000
```

Docker deployment:

```bash
docker build -t hf-finetuning-lab .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts hf-finetuning-lab
```

The API has two endpoints:

- `GET /health`
- `POST /predict`

Example request:

```json
{
  "texts": ["My account login is blocked"]
}
```
