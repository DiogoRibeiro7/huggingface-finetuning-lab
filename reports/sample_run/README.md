# Sample Run

This directory is reserved for generated reports. Training is not executed in this build environment because it requires downloading a Hugging Face model.

Run locally:

```bash
poetry run hf-lab train --input data/raw/support_tickets.csv --output-dir artifacts/models/support-triage --epochs 1 --batch-size 8
```
