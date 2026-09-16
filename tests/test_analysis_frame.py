"""The report's data contract.

``scripts/09_analysis.py`` now separates blocks by *how they were trained*
(``max_steps``, ``signed``, ``standardize``, ``alpha``, ``w_scale``, ``steps``) and
reports the step budget each run actually spent.  Those fields come from the run's
``config.json`` and ``summary.json`` through :func:`load_summaries` and
:func:`summaries_to_frame`; if any of them stops being carried through, the report
fails with a KeyError a long way from the cause, or -- worse -- silently pools the
equal-update control with the epoch-budget curve.  This test pins the contract.
"""

from __future__ import annotations

import json

import pytest

from flynum.analysis.report import load_summaries, summaries_to_frame
from flynum.config import ExperimentConfig

REQUIRED = (
    "run_id", "task", "model", "circuit", "graph", "n_train", "model_seed",
    "test_acc_a", "max_steps", "steps", "alpha", "w_scale", "signed",
    "standardize", "batch_size", "epochs_run", "optimizer_steps",
)


@pytest.fixture()
def fake_runs(tmp_path, monkeypatch):
    """Two runs in a scratch ``runs/``: one epoch-budget, one fixed-update."""
    from flynum import paths

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(paths, "RUNS", runs)

    specs = (
        ("a_epoch_budget", 0, False, 0.70),
        ("b_fixed_updates", 18780, True, 0.55),
    )
    for name, max_steps, signed, acc in specs:
        d = runs / name
        d.mkdir()
        cfg = ExperimentConfig()
        cfg.stage = "t"
        cfg.data.circuit = "core"
        cfg.data.graph = "real"
        cfg.train.max_steps = max_steps
        cfg.train.eval_every = 2 if max_steps else 0
        cfg.model_cfg.signed_synapses = signed
        (d / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
        (d / "full_summary.json").write_text(
            json.dumps({
                "run_id": name, "task": "count", "model": "M1", "graph": "real",
                "circuit": "core", "n_train": 5000, "model_seed": 0,
                "test_acc_a": acc, "test_acc_b": 0.4, "test_acc_c": 0.4,
                "test_acc_d": 0.4,
                "train_info": {"epochs_run": 12, "optimizer_steps": max_steps or 4740},
            }),
            encoding="utf-8",
        )
    return runs


def test_frame_carries_the_block_defining_columns(fake_runs):
    df = summaries_to_frame(load_summaries())
    assert len(df) == 2
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"summary frame lost columns the report depends on: {missing}"


def test_blocks_are_distinguishable(fake_runs):
    df = summaries_to_frame(load_summaries()).set_index("run_id")
    epoch = df.loc["a_epoch_budget"]
    fxu = df.loc["b_fixed_updates"]
    # the equal-update control must not be poolable with the epoch-budget curve
    assert epoch["max_steps"] == 0 and fxu["max_steps"] == 18780
    assert bool(epoch["signed"]) is False and bool(fxu["signed"]) is True
    # ... and the step budget each one actually spent has to be readable
    assert fxu["optimizer_steps"] == 18780
    assert epoch["optimizer_steps"] == 4740
    # a config value that never existed before this round must survive too
    assert int(fxu["batch_size"]) == 64


def test_fields_added_later_still_default_to_the_legacy_behaviour():
    """Mixed-generation pools must stay one condition.

    The 10-seed replication mixes runs written before the signed-synapse and
    fixed-step fields existed with runs written after.  Those runs are only
    comparable because every field added later defaults to exactly what the old
    code did: no signs, no step cap, no staged evaluation, global normalisation.
    If a default ever moves, the older half of the pool silently becomes a
    different experiment, so pin it here.
    """
    from flynum.config import ModelConfig, TrainConfig

    mc, tc = ModelConfig(), TrainConfig()
    assert mc.signed_synapses is False
    assert mc.unknown_nt_sign == 1
    assert mc.weight_normalization == "global"
    assert mc.readout_standardize is True  # the CLI still turns it off explicitly
    assert tc.max_steps == 0
    assert tc.eval_every == 0
