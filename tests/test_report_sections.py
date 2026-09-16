"""The report's populated branches, exercised before the data exists.

Each of section 6.2-6.5 currently renders its "not run yet" placeholder, because the
fixed-update, signed, curriculum and lesion blocks are still queued.  The *populated*
branches -- grouping by sign and weight scale, splitting curriculum runs by held-out
pair, reading a lesion file -- have therefore never executed, and a KeyError in one of
them would surface only while assembling the final deliverable.

This plants synthetic run directories that look exactly like the real ones will, runs
the real analysis script against them, and checks the numbers it produces.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from flynum.config import ExperimentConfig

ANALYSIS = Path(__file__).resolve().parent.parent / "scripts" / "09_analysis.py"


def _write_run(runs: Path, name: str, *, stage="7", task="count", model="M1",
               circuit="core", graph="real", n_train=20000, seed=0, shuffle_seed=0,
               max_steps=0, eval_every=0, signed=False, w_scale=1.0, alpha=0.2,
               acc=(0.77, 0.60, 0.90, 0.60), optimizer_steps=18780, **extra):
    d = runs / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()
    cfg.stage = stage
    cfg.task = task
    cfg.model = model
    cfg.data.circuit = circuit
    cfg.data.graph = graph
    cfg.train.n_train = n_train
    cfg.train.max_steps = max_steps
    cfg.train.eval_every = eval_every
    cfg.model_cfg.signed_synapses = signed
    cfg.model_cfg.w_scale = w_scale
    cfg.model_cfg.alpha = alpha
    cfg.model_cfg.readout_standardize = False
    cfg.seeds["model_seed"] = seed
    cfg.seeds["shuffle_seed"] = shuffle_seed
    (d / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    summary = {
        "run_id": name, "stage": stage, "task": task, "model": model,
        "circuit": circuit, "graph": graph, "n_train": n_train, "model_seed": seed,
        "shuffle_seed": shuffle_seed, "status": "ok",
        "test_acc_a": acc[0], "test_acc_b": acc[1], "test_acc_c": acc[2],
        "test_acc_d": acc[3],
        "per_mode": {m: {"accuracy": acc[i], "n": 5000}
                     for i, m in enumerate("ABCD")},
        "train_info": {"epochs_run": 60, "optimizer_steps": optimizer_steps,
                       "best_val_acc": acc[0]},
    }
    summary.update(extra)
    (d / "full_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(
        {k: v for k, v in summary.items() if k != "per_mode"}), encoding="utf-8")


@pytest.fixture()
def synthetic_study(tmp_path, monkeypatch):
    """A miniature study with every block represented."""
    from flynum import paths

    runs = tmp_path / "runs"
    proc = tmp_path / "processed"
    reports = tmp_path / "reports"
    figs = tmp_path / "figures"
    for p in (runs, proc, reports, figs):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths, "RUNS", runs)
    monkeypatch.setattr(paths, "DATA_PROCESSED", proc)
    monkeypatch.setattr(paths, "REPORTS", reports)
    monkeypatch.setattr(paths, "FIGURES", figs)

    # epoch-budget headline pair
    _write_run(runs, "a_real", graph="real", seed=0, acc=(0.77, 0.60, 0.90, 0.60))
    _write_run(runs, "a_shuf", graph="shuffled", seed=0, shuffle_seed=0,
               acc=(0.66, 0.40, 0.42, 0.37))
    # equal-update control: N=5000 with the full budget, real and shuffled
    _write_run(runs, "b_fxu5k_real", graph="real", n_train=5000, max_steps=18780,
               eval_every=4, acc=(0.52, 0.21, 0.20, 0.18), optimizer_steps=18780)
    _write_run(runs, "b_fxu5k_shuf", graph="shuffled", n_train=5000, max_steps=18780,
               eval_every=4, shuffle_seed=0, acc=(0.51, 0.20, 0.22, 0.17),
               optimizer_steps=18780)
    # Fly-v2: signed at 0.5 plus the scale-matched unsigned control
    _write_run(runs, "c_signed", stage="7c", graph="real", signed=True, w_scale=0.5,
               acc=(0.66, 0.45, 0.55, 0.44))
    _write_run(runs, "c_scalectrl", stage="7c", graph="real", signed=False,
               w_scale=0.5, acc=(0.64, 0.43, 0.52, 0.42))
    # curriculum: 2x2 at the primary holdout, plus one other holdout
    for graph in ("real", "shuffled"):
        for scratch in (False, True):
            _write_run(
                runs, f"d_curr_{graph}_{'scratch' if scratch else 'pre'}", stage="8",
                task="add_curriculum", graph=graph,
                acc=(0.40, 0.35, 0.0, 0.0),
                holdout_pairs=[[2, 3], [3, 2]], scratch=scratch,
                acc_a_test=0.40, acc_b_test=0.35,
                sum_train=0.42 if not scratch else 0.33,
                sum_test=0.28 if not scratch else 0.10,
                transferred={"transferred": ["delta", "bias"],
                             "skipped": ["readout.weight", "readout.bias"],
                             "source_training": {"alpha": 0.2, "steps": 8,
                                                 "w_scale": 1.0, "signed": False}},
            )
    _write_run(runs, "d_curr_lpo13", stage="8", task="add_curriculum",
               acc=(0.38, 0.33, 0.0, 0.0), holdout_pairs=[[1, 3], [3, 1]],
               scratch=False, acc_a_test=0.38, acc_b_test=0.33,
               sum_train=0.40, sum_test=0.22)
    # lesion panel
    deltas = {"LC11": {c: 0.05 for c in "ABCD"},
              "LC10a": {c: 0.002 for c in "ABCD"}}
    for r in range(5):
        deltas[f"random_143_r{r}"] = {c: 0.001 * (r + 1) for c in "ABCD"}
    (proc / "lesion_src.json").write_text(json.dumps({
        "source_run": "a_real", "n_neuron_types": {"LC11": 143, "LC10a": 275,
                                                   "T4a": 0, "T5a": 0},
        "lesions": [{"name": k, "n_lesioned": 143} for k in deltas],
        "deltas": deltas,
    }), encoding="utf-8")
    return tmp_path


def _run_analysis(monkeypatch) -> str:
    """Execute scripts/09_analysis.py's main() with no figures, return the report."""
    from flynum import paths

    spec = importlib.util.spec_from_file_location("analysis09", ANALYSIS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["09_analysis.py", "--no-figures"])
    assert mod.main() == 0
    return (paths.REPORTS / "report.md").read_text(encoding="utf-8")


@pytest.mark.filterwarnings("ignore")
def test_report_renders_every_populated_branch(synthetic_study, monkeypatch):
    text = _run_analysis(monkeypatch)

    # 6.1 -- the seed-level table came from 20_seed_stats.json, which does not exist
    # here, so it must fall back rather than crash
    assert "### 6.1" in text
    # 6.2 -- both arms of the equal-update table must be populated
    assert "### 6.2" in text
    assert "18780" in text, "the fixed-update step budget is missing from the table"
    # 6.3 -- the signed table must separate the sign from the weight scale
    assert "### 6.3" in text
    assert "w_scale" in text
    assert "判读方式" in text
    # 6.4 -- curriculum, split by held-out pair
    assert "### 6.4" in text
    assert "h23-32" in text or "23-32" in text, "primary holdout not identified"
    assert "留出对稳健性" in text, "other holdouts were not reported separately"
    # 6.5 -- the lesion panel
    assert "### 6.5" in text
    assert "random_143_r4" in text, "random knockouts missing from the lesion table"


def test_unrun_blocks_render_a_placeholder_not_an_empty_table(synthetic_study,
                                                             monkeypatch):
    """A block with no data must say so and name the command that produces it."""
    from flynum import paths

    # remove the curriculum and lesion evidence
    for d in list(paths.RUNS.iterdir()):
        if "curr" in d.name:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
    for f in paths.DATA_PROCESSED.glob("lesion_*.json"):
        f.unlink()

    text = _run_analysis(monkeypatch)
    for section in ("### 6.4", "### 6.5"):
        i = text.find(section)
        assert i >= 0, section
        body = text[i:i + 400]
        assert "尚未运行" in body, f"{section} is empty instead of explicit"
        assert "scripts/" in body, f"{section} does not name the command"
