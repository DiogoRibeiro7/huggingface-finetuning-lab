from __future__ import annotations

import os

from hf_finetuning_lab.serving.api import create_app

MODEL_DIR = os.getenv("HF_LAB_MODEL_DIR", "artifacts/models/support-triage")
app = create_app(MODEL_DIR)
