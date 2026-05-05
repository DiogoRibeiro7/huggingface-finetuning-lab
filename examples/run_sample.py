from __future__ import annotations

from pathlib import Path

from hf_finetuning_lab.sample_data import write_sample_data


def main() -> None:
    output = Path("data/raw/support_tickets.csv")
    write_sample_data(output=output, rows=500)
    print(f"Sample data written to {output}")
    print("Next: poetry run hf-lab train --input data/raw/support_tickets.csv --output-dir artifacts/models/support-triage --epochs 1 --batch-size 8")


if __name__ == "__main__":
    main()
