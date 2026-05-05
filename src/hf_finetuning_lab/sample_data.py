from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABEL_TEMPLATES: dict[str, list[str]] = {
    "account": [
        "I cannot log into my account after resetting my password",
        "My profile is locked and I need access again",
        "The login page keeps rejecting my credentials",
        "I need help changing the email address on my account",
    ],
    "billing": [
        "I was charged twice for the same subscription",
        "The invoice amount is wrong this month",
        "My payment failed but the money left my account",
        "I need a refund for an incorrect billing charge",
    ],
    "technical": [
        "The app crashes every time I open the dashboard",
        "The export button does not work on my browser",
        "I receive an error when uploading a file",
        "The system is slow and sometimes freezes",
    ],
    "delivery": [
        "My order has not arrived and tracking has not updated",
        "The package was marked delivered but I did not receive it",
        "I need to change the delivery address for my order",
        "The shipment is delayed and I need an update",
    ],
    "cancellation": [
        "I want to cancel my subscription before renewal",
        "Please stop my plan at the end of the month",
        "I cancelled but I still received a renewal notice",
        "I need confirmation that my membership was cancelled",
    ],
    "security": [
        "I saw suspicious activity on my account",
        "Someone changed my password without permission",
        "I received a login alert from an unknown location",
        "Please help me secure my account immediately",
    ],
}

NOISE_PHRASES = [
    "please respond quickly",
    "this has happened before",
    "I am using the mobile app",
    "I already contacted support",
    "the issue started yesterday",
    "I can provide screenshots",
    "thank you",
]


def generate_support_ticket_data(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic text-classification dataset for support triage.

    Parameters
    ----------
    rows:
        Number of examples to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Dataset with `text` and `label` columns.
    """
    if rows <= 0:
        raise ValueError("rows must be positive.")

    rng = np.random.default_rng(seed)
    labels = list(LABEL_TEMPLATES)
    probabilities = np.array([0.18, 0.18, 0.22, 0.15, 0.12, 0.15])
    probabilities = probabilities / probabilities.sum()

    records: list[dict[str, str | int]] = []
    for row_id in range(rows):
        label = str(rng.choice(labels, p=probabilities))
        template = str(rng.choice(LABEL_TEMPLATES[label]))
        n_noise = int(rng.integers(0, 3))
        noise = " ".join(rng.choice(NOISE_PHRASES, size=n_noise, replace=False))
        urgency = " urgent" if rng.random() < 0.12 else ""
        text = f"{template}{urgency}. {noise}".strip()
        records.append({"id": row_id, "text": text, "label": label})

    return pd.DataFrame.from_records(records)


def write_sample_data(output: str | Path, rows: int = 2000, seed: int = 42) -> Path:
    """Write synthetic support-ticket data to CSV."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_support_ticket_data(rows=rows, seed=seed)
    df.to_csv(output_path, index=False)
    return output_path
