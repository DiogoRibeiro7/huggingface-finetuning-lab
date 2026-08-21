"""Resolve every Hub preset against the real Hugging Face Hub.

Excluded from the default run and from pull-request CI: it needs network, so
it is too slow and too flaky to gate a merge. It runs on a schedule instead.

It exists because preset breakage is invisible to the rest of the suite. The
other Hub tests build datasets in memory, so nothing downloads anything, and a
dependency bump that makes every preset unresolvable still passes CI — which is
exactly what huggingface-hub 1.28 did when it started rejecting bare repository
ids that had previously resolved through Hub redirects.
"""

from __future__ import annotations

import pytest

from hf_finetuning_lab.data.hub import HUB_PRESETS

pytestmark = pytest.mark.network

datasets = pytest.importorskip("datasets")


@pytest.mark.parametrize("preset", sorted(HUB_PRESETS))
def test_preset_resolves_on_the_hub(preset: str) -> None:
    cfg = HUB_PRESETS[preset]

    builder = (
        datasets.load_dataset_builder(cfg.name, cfg.config)
        if cfg.config
        else datasets.load_dataset_builder(cfg.name)
    )

    splits = set((builder.info.splits or {}).keys())
    assert splits, f"{cfg.name} reported no splits"
    missing = {s for s in cfg.splits().values() if s not in splits}
    assert not missing, f"{cfg.name} is missing configured splits {sorted(missing)}"
