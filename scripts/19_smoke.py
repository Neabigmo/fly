"""CPU smoke test for the blocks sitting in the GPU queue.

The queue spends ~19 h of GPU time on code paths that had never been executed
end to end (signed synapses, the three-head curriculum, the lesion panel, the
fixed-step budget).  A typo found four hours into that queue costs the whole
night, so this script runs every one of those paths once, tiny, on the CPU, with
**fully isolated caches**: the stimulus directory is redirected into
``.cache/smoke`` so it cannot touch the real stimulus cache.

That isolation matters.  The count-stimulus cache key does not include
``n_test``, so a naive tiny run would regenerate and overwrite the 5000-sample
test set that every existing run -- and the running replication -- evaluates on.

Writes nothing permanent: the run directories it creates are deleted at the end.

Usage::

    python scripts/19_smoke.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

# Must precede the torch import: this test is deliberately single-threaded CPU so
# it can run alongside a GPU job without stealing the device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402

SCRATCH = paths.CACHE / "smoke"
CREATED_RUNS: list[str] = []
#: hands run ids between checks without a second module-level mutable
STATE: dict[str, str] = {}


def _isolate() -> None:
    """Point every stimulus cache at a scratch directory."""
    import flynum.train.data as data

    stim = SCRATCH / "stimuli"
    stim.mkdir(parents=True, exist_ok=True)
    data.STIM_DIR = stim
    # Keep the training pool tiny too, so no 20k-image pool is generated.
    data.POOL_SIZE = 64


def _tiny_config(**over):
    from flynum.config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.stage = "9"
    cfg.task = "count"
    cfg.model = "M1"
    cfg.data.circuit = "core"
    cfg.data.graph = "real"
    cfg.time.steps = 4
    cfg.model_cfg.alpha = 0.2
    cfg.model_cfg.w_scale = 1.0
    # every count and addition run in the study trains with this off
    cfg.model_cfg.readout_standardize = False
    cfg.train.n_train = 60
    cfg.train.n_val = 60
    cfg.train.n_test = 60
    # The real runs are fine at the default gain rate; a 60-image smoke run with
    # four batches blows up, which would abort before the step budget is spent.
    cfg.train.lr_gain = 3e-4
    cfg.train.lr_readout = 1e-3
    cfg.train.batch_size = 16
    cfg.train.max_epochs = 2
    cfg.train.patience = 99
    cfg.stimulus.add_reps_per_pair = 2
    cfg.time.steps_a = 2
    cfg.time.steps_gap = 2
    cfg.time.steps_b = 2
    for k, v in over.items():
        if hasattr(cfg.train, k):
            setattr(cfg.train, k, v)
        else:
            setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
def check_signed_dynamics(log) -> None:
    """h(t) must stay bounded under signed synapses (Fly-v2's whole point).

    Also takes a few gradient steps with signed weights at the grid's w_scale.  The
    forward pass alone would not have caught a divergent *training* run, and the
    signed grid is hours of GPU time.
    """
    import torch

    from flynum.pipeline import prepare
    from flynum.train.trainer import encode, make_optimizer

    models = {}
    for signed in (False, True):
        cfg = _tiny_config()
        cfg.model_cfg.signed_synapses = signed
        prep = prepare(cfg, logger=None)
        if signed:
            rep = prep.sign_report or {}
            log.info("  sign report: %s", rep)
            assert rep.get("fraction_signed", 0) > 0.5, (
                f"signed synapses barely resolved any neuron: {rep}"
            )
            assert rep.get("n_inhibitory", 0) > 0, "no inhibitory neurons resolved"
        model = prep.build_model(device="cpu", seed=0)
        w = model.base_weight
        if signed:
            log.info("  weight sign split: %d negative / %d total",
                     int((w < 0).sum()), w.numel())
            assert (w < 0).any(), "edge signs never reached the weights"
            assert (w > 0).any(), "every weight became negative"
        with torch.no_grad():
            cols = torch.rand(8, model.n_columns)
            for steps in (8, 32):
                _, hist = model.forward_count(cols, steps)
                peak = max(float(h.abs().max()) for h in hist)
                log.info("  signed=%-5s steps=%2d  peak |h| = %.4g", signed, steps, peak)
                assert np.isfinite(peak), "h went non-finite"
        models[signed] = (model, prep)

    # a few training steps at the signed grid's weight scale
    from flynum.retina.torch_encoder import TorchRetina
    from flynum.train.data import build_count_data

    cfg = _tiny_config()
    cfg.model_cfg.signed_synapses = True
    cfg.model_cfg.w_scale = 0.5
    cfg.train.lr_gain = 3e-4
    cfg.train.lr_readout = 1e-3
    prep = prepare(cfg, logger=None)
    retina = TorchRetina(prep.encoder, device="cpu")
    data = build_count_data(cfg, logger=None)
    img, y = data.subset(60, seed=cfg.seeds["split_seed"])
    torch.manual_seed(0)
    model = prep.build_model(device="cpu", seed=0)
    opt, ce = make_optimizer(model, cfg), torch.nn.CrossEntropyLoss()
    losses = []
    for i in range(0, 60, 16):
        cols = encode(img[i : i + 16], retina, "cpu")
        t = torch.as_tensor(y[i : i + 16], dtype=torch.long)
        logits, _ = model.forward_count(cols, cfg.time.steps)
        loss = ce(logits, t) + model.gain_penalty(cfg.model_cfg.lambda_gain)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip
        )
        opt.step()
        losses.append(float(loss))
    log.info("  signed w_scale=0.5 training steps: losses %s",
             [round(v, 3) for v in losses])
    assert all(np.isfinite(v) and v < 25.0 for v in losses), (
        f"signed training diverges at w_scale=0.5: {losses}"
    )


def check_fixed_steps(log) -> None:
    """The step budget must be spent exactly, with the eval cadence honoured."""
    from flynum.experiments.count import run_count_experiment

    # 60 images / batch 16 = 4 updates per epoch, so a budget of 10 spans three
    # epochs and lands mid-epoch: exactly the case where the eval cadence must
    # skip an epoch and still catch the final one.
    cfg = _tiny_config(max_steps=10, eval_every=2, max_epochs=3)
    s = run_count_experiment(cfg, logger=None)
    CREATED_RUNS.append(s["run_id"])
    STATE["count_run"] = s["run_id"]
    tr = s["train_info"]
    log.info("  steps=%s epochs_iterated=%s epochs_logged=%s val=%.3f",
             tr.get("optimizer_steps"), tr.get("epochs_iterated"),
             tr.get("epochs_run"), s.get("best_val_acc", float("nan")))
    assert tr.get("optimizer_steps") == 10, (
        f"budget not spent exactly: {tr.get('optimizer_steps')} != 10"
    )
    assert tr.get("epochs_iterated") == 3, tr.get("epochs_iterated")
    assert tr.get("epochs_run") == 2, (
        f"eval_every had no effect: {tr.get('epochs_run')} validations in "
        f"{tr.get('epochs_iterated')} epochs"
    )


def check_curriculum(log) -> str:
    """Three-head curriculum: scratch run, then a warm start from its checkpoint."""
    from flynum.experiments.curriculum import CurriculumConfig, run_curriculum

    cc = CurriculumConfig(operand_epochs=1, pretrain_from="", scratch=True)
    cfg = _tiny_config()
    cfg.task = "add"
    cfg.stimulus.holdout_pairs = ((2, 3), (3, 2))
    s = run_curriculum(cfg, cc, logger=None)
    CREATED_RUNS.append(s["run_id"])
    log.info("  scratch: a %.3f b %.3f sum %.3f (train a %.3f b %.3f sum %.3f)",
             s["acc_a_test"], s["acc_b_test"], s["sum_test"],
             s["acc_a_train"], s["acc_b_train"], s["sum_train"])
    for k in ("acc_a_test", "acc_b_test", "sum_test"):
        assert 0.0 <= s[k] <= 1.0, f"{k} out of range: {s[k]}"

    # head geometry: operands 4 classes, sum 7
    from flynum.pipeline import prepare

    prep = prepare(cfg, logger=None)
    m = prep.build_model(device="cpu", n_heads=3,
                         head_classes=(4, 4, prep.n_classes))
    assert m.head_classes == (4, 4, 7), m.head_classes
    assert m.readout.out_features == 15, m.readout.out_features
    import torch

    with torch.no_grad():
        z = torch.randn(5, len(m.readout_idx))
        outs = m.classify(z)
    assert [tuple(o.shape) for o in outs] == [(5, 4), (5, 4), (5, 7)], \
        [tuple(o.shape) for o in outs]
    log.info("  head geometry OK: 4 + 4 + 7 outputs over %d shared features",
             len(m.readout_idx))

    cc2 = CurriculumConfig(operand_epochs=1, pretrain_from=s["run_id"], scratch=False)
    s2 = run_curriculum(cfg, cc2, logger=None)
    CREATED_RUNS.append(s2["run_id"])
    tr = s2.get("transferred") or {}
    log.info("  pretrained: transferred %d recurrent tensors, skipped %d readout",
             len(tr.get("transferred", [])), len(tr.get("skipped", [])))
    assert len(tr.get("transferred", [])) > 0, "warm start transferred nothing"
    assert len(tr.get("skipped", [])) > 0, "warm start did not leave the readout fresh"
    return s2["run_id"]


def check_lesion(log, source: str) -> None:
    """Lesion panel, including the readout-only mode that used to break shapes."""
    from flynum.experiments.lesion import run_lesion_panel

    cfg = _tiny_config()
    summary = run_lesion_panel(cfg, source, random_repeats=2, seed=0, logger=None)
    CREATED_RUNS.append(summary["run_id"])
    names = [r["name"] for r in summary["lesions"]]
    log.info("  panel: %s", names)
    log.info("  neuron counts: %s", summary["n_neuron_types"])
    assert "intact" in names, names
    for t in ("LC11", "LC10a"):
        assert t in names, f"{t} not found in the core subgraph -- panel would be empty"
    assert any(n.startswith("LC11_readout") for n in names), names
    assert any(n.startswith("random_") for n in names), names
    for name, d in summary["deltas"].items():
        for cond in ("A", "B", "C", "D"):
            assert np.isfinite(d[cond]), f"{name}/{cond} is not finite"
    # A zero delta is the expected result here: this model has had one epoch of
    # training on 60 images, so silencing 143 of 33,480 neurons cannot move the
    # accuracy of a network that is already at chance.  The check is that the
    # panel runs, finds the right neuron counts, and reports finite numbers.
    log.info("  deltas: %s", {k: v for k, v in list(summary["deltas"].items())[:3]})


def check_addition_pilot(log) -> None:
    """The 2x2 addition pilot: both brain initialisations and both teaching modes.

    Worth a smoke test because it runs on a remote host where interactive debugging is
    awkward, and because the warm start crosses a head-geometry change: the counting
    model has one head with 5 classes, the pilot model has three heads of 4/4/7.
    """
    from flynum.config import ExperimentConfig
    from flynum.experiments.addition_pilot import PilotConfig, run_addition_pilot
    from flynum.experiments.count import run_count_experiment

    def tiny(**over):
        cfg = _tiny_config()
        cfg.task = "add"
        cfg.stimulus.holdout_pairs = ((2, 3),)
        cfg.stimulus.add_reps_per_pair = 2
        cfg.data.graph = "real"
        cfg.model_cfg.alpha = 0.2
        cfg.model_cfg.w_scale = 1.0
        cfg.time.steps_a, cfg.time.steps_gap, cfg.time.steps_b = 2, 2, 2
        cfg.train.batch_size = 16
        cfg.train.lr_gain = 3e-4
        cfg.train.lr_readout = 1e-3
        for k, v in over.items():
            if hasattr(cfg.train, k):
                setattr(cfg.train, k, v)
            elif hasattr(cfg.stimulus, k):
                setattr(cfg.stimulus, k, v)
            else:
                setattr(cfg, k, v)
        return cfg

    # a scratch counting run to serve as the warm-start source
    src = run_count_experiment(_tiny_config(max_steps=4, eval_every=2, max_epochs=2),
                              logger=None)
    CREATED_RUNS.append(src["run_id"])

    for cell, pretrain, full in (("A1", "", True), ("B2", src["run_id"], False)):
        pc = PilotConfig(label=cell, pretrain_from=pretrain, teach_holdout=full,
                         budget_steps=6, probe_steps=(0, 3, 6), eval_items=20)
        cfg = tiny()
        s = run_addition_pilot(cfg, pc, logger=None)
        CREATED_RUNS.append(s["run_id"])
        ups = [h["updates"] for h in s["history"]]
        log.info("  %s brain=%s teaching=%s | probes at updates %s | %d taught items",
                 cell, s["brain"], s["teaching"], ups, s["n_taught_items"])
        assert ups == [0, 3, 6], f"probes not recorded on the exact steps: {ups}"
        assert s["optimizer_steps"] == 6, (
            f"budget not spent exactly: {s['optimizer_steps']} != 6")
        if pretrain:
            tr = s.get("transferred") or {}
            assert len(tr.get("transferred", [])) > 0, "warm start transferred nothing"
            assert len(tr.get("skipped", [])) > 0, "warm start did not leave readout fresh"
        # the two teaching modes must differ in how much they are taught
        if full:
            assert s["n_taught_items"] > 0 and s["n_probe_items"] > 0
        for h in s["history"]:
            for k in ("acc_seen", "acc_2p3", "p_y5_given_2p3", "acc_a_2p3", "acc_b_2p3"):
                assert np.isfinite(h[k]), f"{k} not finite at {h['updates']} updates"
    # the two teaching modes must have different training-set sizes, same probe set
    sizes = {}
    for cell, pretrain, full in (("A1", "", True), ("B1", "", False)):
        s = run_addition_pilot(tiny(),
                               PilotConfig(label=cell, teach_holdout=full,
                                           budget_steps=3, probe_steps=(0, 3),
                                           eval_items=20),
                               logger=None)
        CREATED_RUNS.append(s["run_id"])
        sizes[cell] = (s["n_taught_items"], s["n_probe_items"])
    log.info("  taught/probe item counts: %s", sizes)
    assert sizes["A1"][0] > sizes["B1"][0], (
        "the full cell must train on more items than the holdout cell")
    assert sizes["A1"][1] == sizes["B1"][1], (
        "the two teaching modes must share a byte-identical 2+3 probe set")


# --------------------------------------------------------------------------- #
def check_config_roundtrip(log) -> None:
    """The lesion CLI rebuilds its config from a run's config.json.

    Any config the study has written must survive that round trip, otherwise a
    queued block would silently train or evaluate under default settings.
    """
    import json

    from flynum.config import ExperimentConfig

    n = 0
    for d in sorted(paths.RUNS.iterdir()):
        p = d / "config.json"
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        cfg = ExperimentConfig.from_dict(
            {k: v for k, v in raw.items() if not k.startswith("_")}, strict=False
        )
        assert cfg.data.circuit == raw["data"]["circuit"], d.name
        assert cfg.data.graph == raw["data"]["graph"], d.name
        assert cfg.train.n_train == raw["train"]["n_train"], d.name
        assert cfg.model_cfg.w_scale == raw["model_cfg"]["w_scale"], d.name
        assert cfg.task == raw["task"], d.name
        n += 1
    log.info("  %d run configs round-trip through ExperimentConfig.from_dict", n)
    assert n > 0, "no run configs found to validate"


def check_standardize_guard(log) -> None:
    """A near-silent readout pool must refuse standardisation, not diverge.

    Enabling this on the core circuit used to explode the first epoch (measured
    cross-entropy 600) because the feature rescaling also rescales the gradient
    that reaches the recurrent gains by ~3e4.  The guard has to fire on exactly
    that configuration, so this checks the real path rather than a stub.
    """
    from flynum.experiments.count import run_count_experiment

    cfg = _tiny_config(readout_standardize=True)
    cfg.model_cfg.readout_standardize = True
    cfg.train.max_epochs = 1
    s = run_count_experiment(cfg, logger=None)
    CREATED_RUNS.append(s["run_id"])
    raw = json.loads((paths.run_dir(s["run_id"]) / "metrics.jsonl")
                     .read_text(encoding="utf-8").splitlines()[0])
    log.info("  epoch 0 train_loss=%.4g diverged=%s", raw["train_loss"],
             raw.get("diverged", False))
    assert not raw.get("diverged", False), (
        f"standardisation guard did not fire: loss {raw['train_loss']:.4g}"
    )
    assert raw["train_loss"] < 25.0, raw["train_loss"]


# --------------------------------------------------------------------------- #
def main() -> int:
    log = get_logger("smoke")
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    _isolate()
    log.info("smoke test on CPU | stimulus cache redirected to %s", SCRATCH)

    ok = True
    for name, fn in (
        ("config round trip", lambda: check_config_roundtrip(log)),
        ("signed dynamics", lambda: check_signed_dynamics(log)),
        ("fixed step budget", lambda: check_fixed_steps(log)),
        ("standardisation guard", lambda: check_standardize_guard(log)),
        ("curriculum + warm start", lambda: check_curriculum(log)),
        ("2x2 addition pilot", lambda: check_addition_pilot(log)),
    ):
        log.info("=" * 78)
        log.info("CHECK %s", name)
        try:
            fn()
            log.info("  PASS %s", name)
        except Exception:
            ok = False
            log.error("  FAIL %s\n%s", name, traceback.format_exc())
    # The lesion panel needs a counting checkpoint with a single head; that is
    # what the fixed-step check produced.
    try:
        src = STATE.get("count_run")
        if src is None:
            raise RuntimeError("no counting checkpoint to lesion")
        log.info("=" * 78)
        log.info("CHECK lesion panel (source %s)", src)
        check_lesion(log, src)
        log.info("  PASS lesion panel")
    except Exception:
        ok = False
        log.error("  FAIL lesion panel\n%s", traceback.format_exc())

    # Clean up: nothing this script created may survive into the analysed runs.
    from flynum.logging_utils import rebuild_index

    for rid in CREATED_RUNS:
        d = paths.run_dir(rid)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    for leaked in paths.DATA_PROCESSED.glob("lesion_s9_count_*.json"):
        leaked.unlink()
    info = rebuild_index()
    log.info("removed %d smoke run dirs and the scratch cache", len(CREATED_RUNS))
    log.info("runs/index.csv rebuilt: %d rows, %d stale dropped",
             info["rows"], info["dropped"])
    log.info("=" * 78)
    log.info("SMOKE TEST %s", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
