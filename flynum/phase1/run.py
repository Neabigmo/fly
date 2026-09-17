"""The Phase I trainer: one cell of the plan, measured in optimiser updates.

Everything here is arranged around three commitments.

**Comparisons are made in optimiser updates, not epochs.**  The project's own
sample-efficiency result turned out to be a compute artefact -- with a fixed number of
epochs a larger dataset silently received proportionally more gradient steps -- so every
cell of a line is run for the same number of updates, and every metric is recorded at
pre-registered update counts.  A cell's learning curve is therefore sampled at identical
compute in every condition.

**Holdout purity is structural.**  The training set is built by selecting rows whose item
is taught (:mod:`flynum.phase1.data` asserts this before training starts).  There is no
early stopping and no best-checkpoint selection, because both are ways of choosing a
moment using the held-out set, and the question here is precisely *when* held-out
performance changes.

**A transition is a joint fact, not a threshold crossing.**  Accuracy is discrete: an
argmax crossing an arbitrary cut produces a step even when nothing reorganised.  So every
probe records held-out cross-entropy, P(correct), the correct-minus-runner-up margin and
predictive entropy alongside accuracy, plus internal quantities (edge-gain magnitude,
state scale, effective dimensionality, and linear decodability of the intermediate
quantities).  Deciding what counts as emergence is left to
:mod:`flynum.phase1.emergence`, which applies the criteria frozen in
:mod:`flynum.phase1.spec` to these curves.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import ExperimentConfig
from ..devices import pick_device
from ..logging_utils import RunContext, get_logger
from ..pipeline import prepare
from ..retina.torch_encoder import TorchRetina
from ..train.trainer import encode, make_optimizer
from .. import paths
from . import data as ph_data
from . import spec
from .tasks import TASKS, SequencePlan, TaskDef

DEVICE = pick_device()


# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    """One row of the plan, resolved into everything the runner needs."""

    run: str
    task: str
    brain: str
    budget: int
    probes: tuple[int, ...]
    holdout: tuple[tuple[int, ...], ...] = ()
    teach: tuple[tuple[int, ...], ...] = ()
    line: str = ""
    question: str = ""
    #: render budget per item; 600 over 49 items is ~29k rows, matching the scale of
    #: the earlier stages while staying small enough to be memorised
    reps_per_item: int = 600
    #: floor on the training set size, so a sparse fact table still gets a real dataset
    min_train_rows: int = 20_000
    eval_items: int = 1_500
    ckpt_every: int = 2_000
    permanent_every: int = 50_000
    #: gradient steps used to fit one internal linear probe
    probe_steps: int = 300
    #: processes used to render the stimulus pools (one-off; the pools are cached)
    render_workers: int = 1
    seed: int = 0
    device: str = ""
    tag: str = "10"

    @property
    def warm_start(self) -> str:
        return spec.WARM_START.get(self.brain, "")

    def resolved(self, task: TaskDef) -> tuple[set, tuple]:
        """(taught items, withheld items) for this cell."""
        if self.teach:
            taught = set(self.teach)
        else:
            taught = set(task.items) - set(self.holdout)
        return taught, tuple(self.holdout)


def cell_from_spec(t: spec.TaskSpec, **kw) -> Cell:
    return Cell(run=t.run, task=t.task, brain=t.brain, budget=t.budget, probes=t.probes,
                holdout=t.holdout, teach=t.teach, line=t.line, question=t.question, **kw)


def make_config(cell: Cell) -> ExperimentConfig:
    """The unified Fly-v2 configuration, from the frozen spec."""
    cfg = ExperimentConfig()
    cfg.task = cell.task
    cfg.data.circuit = spec.DYNAMICS["circuit"]
    cfg.data.graph = "real"
    cfg.model_cfg.w_scale = spec.DYNAMICS["w_scale"]
    cfg.model_cfg.alpha = spec.DYNAMICS["alpha"]
    cfg.model_cfg.signed_synapses = spec.DYNAMICS["signed_synapses"]
    cfg.model_cfg.readout_standardize = spec.DYNAMICS["readout_standardize"]
    cfg.model_cfg.readout = spec.DYNAMICS["readout"]
    cfg.model_cfg.learn_gains = "delta" in spec.DYNAMICS["learn"]
    cfg.model_cfg.lambda_gain = spec.OPTIM["lambda_gain"]
    for k, v in spec.STIMULUS.items():
        setattr(cfg.stimulus, k, v)
    cfg.time.steps_a = spec.TIME["steps_operand"]
    cfg.time.steps_gap = spec.TIME["steps_gap"]
    cfg.time.steps_b = spec.TIME["steps_operand"]
    cfg.train.batch_size = spec.OPTIM["batch_size"]
    cfg.train.lr_gain = spec.OPTIM["lr_gain"]
    cfg.train.lr_readout = spec.OPTIM["lr_readout"]
    cfg.train.wd_readout = spec.OPTIM["wd_readout"]
    cfg.train.grad_clip = spec.OPTIM["grad_clip"]
    cfg.train.cosine = spec.OPTIM["schedule"] != "constant"
    cfg.seeds["model_seed"] = cell.seed
    return cfg


# --------------------------------------------------------------------------- #
def phase_columns(images: np.ndarray, retina: TorchRetina, plan: SequencePlan,
                  device) -> list[torch.Tensor]:
    """``(B, n_content_phases, S, S)`` images -> one column tensor per phase."""
    return [encode(images[:, i], retina, device) for i in range(images.shape[1])]


def sequence(columns: list[torch.Tensor], plan: SequencePlan) -> list[torch.Tensor]:
    """The full per-step stimulus list, with blanks materialised as zeros."""
    seq: list[torch.Tensor] = []
    c = 0
    for kind in plan.task.template:
        if kind == "blank":
            seq += [torch.zeros_like(columns[0])] * plan.steps_gap
        else:
            seq += [columns[c]] * (plan.steps_operand if kind == "dot" else plan.steps_cue)
            c += 1
    return seq


def run_batch(model, columns: list[torch.Tensor], plan: SequencePlan):
    """Forward one batch, returning (answer logits, history)."""
    hist = model._run(sequence(columns, plan), columns[0].shape[0])[1]
    return model.classify(model.pool(hist))[0], hist


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_split(model, retina: TorchRetina, split: dict, plan: SequencePlan, *,
                   batch_size: int = 256, device=DEVICE) -> dict:
    """Accuracy plus the continuous metrics that a threshold crossing would not show."""
    if split is None or split["n"] == 0:
        return {}
    model.eval()
    n = split["n"]
    logits_all, labels_all = [], []
    for i in range(0, n, batch_size):
        imgs = split["images"][i:i + batch_size]
        cols = phase_columns(imgs, retina, plan, device)
        logits, _ = run_batch(model, cols, plan)
        logits_all.append(logits.float().cpu())
        labels_all.append(torch.as_tensor(split["labels"][i:i + batch_size],
                                          dtype=torch.long))
    logits = torch.cat(logits_all)
    y = torch.cat(labels_all)
    logp = torch.log_softmax(logits, dim=1)
    p = logp.exp()
    ce = float(-logp.gather(1, y[:, None]).mean())
    correct = logits.argmax(1) == y
    margin = (logits.gather(1, y[:, None])
              - logits.masked_fill(torch.nn.functional.one_hot(y, logits.shape[1]).bool(),
                                   float("-inf")).max(dim=1, keepdim=True).values)
    ent = float(-(p * logp).sum(1).mean())
    return {
        "n": int(n),
        "acc": float(correct.float().mean()),
        "ce": ce,
        "p_correct": float(p.gather(1, y[:, None]).mean()),
        "margin": float(margin.mean()),
        "entropy": ent,
        "chance": 1.0 / split["n_classes"],
    }


@torch.no_grad()
def _phase_features(model, retina: TorchRetina, split: dict, plan: SequencePlan, *,
                    batch_size: int = 256, device=DEVICE) -> dict[int, torch.Tensor]:
    """Pooled readout features at the end of each content phase, for a whole split."""
    model.eval()
    ends = plan.content_ends()
    acc: dict[int, list] = {e: [] for e in ends}
    for i in range(0, split["n"], batch_size):
        cols = phase_columns(split["images"][i:i + batch_size], retina, plan, device)
        seq = sequence(cols, plan)
        hist = model._run(seq, cols[0].shape[0])[1]
        for e in ends:
            acc[e].append(model.pool(hist[:e]).float().cpu())
    return {e: torch.cat(v) for e, v in acc.items()}


def fit_probe(features: torch.Tensor, labels: torch.Tensor, n_classes: int, *,
              steps: int, n_components: int = 64, lr: float = 0.05,
              device=DEVICE) -> float:
    """Accuracy of a linear probe, fitted on half the rows and scored on the other half.

    Two things make this well posed rather than decorative.  The readout pool here is
    9,200 visual-projection neurons, so a raw linear probe has far more parameters than
    the rows it is fitted on and would report noise; the features are therefore projected
    onto the top ``n_components`` principal directions of the *training* half before the
    probe is fitted.  And the fit is split-half, so the probe cannot report decodability
    by memorising the rows it is scored on -- for a state measurement there is nothing
    else to hold out, since the question is whether the quantity is linearly present in
    the representation, not whether the probe generalises to new items.
    """
    n = features.shape[0]
    if n < 16 or len(torch.unique(labels)) < 2:
        return float("nan")
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    half = n // 2
    tr, te = perm[:half], perm[half:]
    x = features.to(device).float()
    y = labels.to(device)
    mu = x[tr].mean(0, keepdim=True)
    sd = x[tr].std(0, keepdim=True).clamp_min(1e-6)
    xt = (x[tr] - mu) / sd
    k = int(min(n_components, xt.shape[0] - 1, xt.shape[1]))
    # principal directions of the training half, from its (half x half) Gram matrix:
    # cheap even though the feature dimension is 9,200
    gram = xt @ xt.t()
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)[:k]
    basis = xt.t() @ evecs[:, order]
    basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-8)
    xt = xt @ basis
    xe = ((x[te] - mu) / sd) @ basis

    lin = nn.Linear(k, n_classes).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        lossf(lin(xt), y[tr]).backward()
        opt.step()
    with torch.no_grad():
        return float((lin(xe).argmax(1) == y[te]).float().mean())


def internal_metrics(model, retina: TorchRetina, split: dict, plan: SequencePlan, *,
                     probe_steps: int, device=DEVICE) -> dict:
    """What the state looks like, and what can be read out of it, at this moment."""
    out: dict[str, float] = {}
    if model.delta is not None:
        d = model.delta.detach()
        out["delta_abs_mean"] = float(d.abs().mean())
        out["delta_abs_max"] = float(d.abs().max())
    if model.bias is not None:
        out["bias_abs_mean"] = float(model.bias.detach().abs().mean())

    with torch.no_grad():
        cols = phase_columns(split["images"][:256], retina, plan, device)
        hist = model._run(sequence(cols, plan), cols[0].shape[0])[1]
        h = hist[-1]
        out["h_peak"] = float(h.abs().max())
        out["h_mean"] = float(h.abs().mean())
        # effective dimensionality of the state: participation ratio of the spectrum of
        # the (B x B) Gram matrix, which costs nothing at this batch size
        hc = h - h.mean(dim=1, keepdim=True)
        ev = torch.linalg.eigvalsh(hc.t() @ hc).clamp_min(0)
        out["activity_dim"] = float(ev.sum() ** 2 / ev.pow(2).sum().clamp_min(1e-30))

    feats = _phase_features(model, retina, split, plan, device=device)
    ends = plan.content_ends()
    item_index = split["item_index"]
    for name, content_idx, n_classes, fn in plan.task.probes():
        if content_idx >= len(ends):
            continue
        e = ends[content_idx]
        labels = torch.as_tensor([fn(tuple(it)) for it in split["items"][item_index]],
                                 dtype=torch.long)
        out[f"probe_{name}"] = fit_probe(feats[e], labels, n_classes, steps=probe_steps,
                                         device=device)
    return out


# --------------------------------------------------------------------------- #
def _checkpoint(model, path: Path) -> None:
    torch.save({"model": model.state_dict()}, path)


def _rotate(ckpt_dir: Path, keep: int = 3) -> None:
    """Keep the last ``keep`` rolling checkpoints and drop the rest.

    Rolling + periodic rather than every checkpoint kept: at 33k neurons a checkpoint is
    a few megabytes, so a 500,000-update run at one per 2,000 updates would write
    hundreds of them per cell for no analytical gain -- the probe log already records the
    trajectory, and the periodic checkpoints keep the coarse history on disk.
    """
    roll = sorted(ckpt_dir.glob("roll_*.pt"))
    for old in roll[:-keep]:
        old.unlink(missing_ok=True)



def run_id_for(cell_name: str, tag: str, seed: int) -> str:
    """The run directory name a cell will actually be given.

    This exists as a named function because getting it wrong is silent in the worst way:
    a warm start that looks in ``runs/C0`` while the run wrote ``runs/10_C0_s0`` simply
    does not find a checkpoint, and the first version of this code aborted the whole queue
    on the second cell.  :func:`run_cell` and :func:`find_checkpoint` both call it, and a
    test asserts it agrees with what :class:`RunContext` creates.
    """
    return f"{tag}_{cell_name}_s{seed}"


def find_checkpoint(cell_name: str, *, tag: str = "10", seed: int = 0) -> Path | None:
    """Locate a previous cell's weights, tolerating an untagged run directory.

    The untagged name is tried second so that a checkpoint produced outside the Phase I
    runner (the pilot study's ``countA01_src``, for instance) can still be used as a warm
    start deliberately.
    """
    from .. import paths

    for run in (run_id_for(cell_name, tag, seed), cell_name):
        for cand in (paths.run_dir(run) / "ckpt" / "final.pt",
                     paths.run_dir(run) / "ckpt" / "last.pt"):
            if cand.exists():
                return cand
    return None


def run_cell(cell: Cell, *, logger=None, cfg: ExperimentConfig | None = None,
             show_progress: bool = True) -> dict:
    """Train one cell of the plan and write its curves."""
    log = logger or get_logger("phase1")
    task = TASKS[cell.task]
    plan = SequencePlan(task)
    cfg = cfg or make_config(cell)
    sc = cfg.stimulus

    ctx = RunContext(run_id_for(cell.run, cell.tag, cell.seed), {
        **cfg.to_dict(),
        "phase1": {
            "cell": cell.run, "task": cell.task, "line": cell.line, "brain": cell.brain,
            "budget": cell.budget, "probes": list(cell.probes),
            "holdout": [list(p) for p in cell.holdout],
            "teach": [list(p) for p in cell.teach],
            "question": cell.question, "seed": cell.seed,
            "spec": {"dynamics": spec.DYNAMICS, "time": spec.TIME,
                     "stimulus": spec.STIMULUS, "optim": spec.OPTIM,
                     "warm_start": spec.WARM_START},
        },
    })
    log = ctx.log
    log.info("=" * 88)
    log.info("PHASE I %s | %s | brain=%s | %d updates | %d recurrent steps/update",
             cell.run, task.label, cell.brain, cell.budget, plan.n_steps())
    if cell.question:
        log.info("  question: %s", cell.question)

    taught, held = cell.resolved(task)
    reps = max(cell.reps_per_item, int(np.ceil(cell.min_train_rows / max(len(taught), 1))))
    pools = ph_data.build_pools(task, sc, taught=taught, holdout=held, reps=reps,
                                seed=cfg.seeds["data_seed"], eval_items=cell.eval_items,
                                logger=log, workers=cell.render_workers)
    splits = pools["splits"]

    prep = prepare(cfg, logger=log, n_classes=task.n_classes,
                   label_offset=task.label_offset)
    retina = TorchRetina(prep.encoder, device=cell.device or DEVICE)
    device = retina.device
    torch.manual_seed(cell.seed)
    model = prep.build_model(device=device, n_heads=1)
    warm = ""
    if cell.brain != "scratch":
        warm = cell.warm_start
        ckpt = find_checkpoint(warm, tag=cell.tag, seed=cell.seed)
        if ckpt is None:
            raise FileNotFoundError(
                f"{cell.run} needs the recurrent warm start {warm!r}, but neither "
                f"{paths.run_dir(run_id_for(warm, cell.tag, cell.seed))}/ckpt/final.pt "
                f"nor {paths.run_dir(warm)}/ckpt/final.pt exists")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        rep = model.load_recurrent_from(state)
        log.info("  warm start: %d recurrent tensors from %s; readout fresh "
                 "(%d tensors skipped)", len(rep["transferred"]), warm,
                 len(rep["skipped"]))
    else:
        log.info("  scratch: gains g=1, bias 0.1, readout freshly initialised")

    opt = make_optimizer(model, cfg)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(cell.seed + 10_000)
    train = splits["train"]
    n_train = train["n"]
    y_all = torch.as_tensor(train["labels"], dtype=torch.long)
    bs = cfg.train.batch_size
    log.info("  training on %d rows in conditions %s (%d taught items); %d withheld; "
             "%d unsupported; conditions %s are read out for reporting",
             n_train, "+".join(pools["train_modes"]), len(taught), len(held),
             len(pools["unsupported_items"]), "+".join(pools["eval_modes"]))
    log.info("  optimizer %s, constant lr %.0e, batch %d, %d updates = %.1f epochs",
             spec.OPTIM["optimizer"], cfg.train.lr_gain, bs, cell.budget,
             cell.budget * bs / max(n_train, 1))

    history: list[dict] = []
    t0 = time.time()

    def record(step: int, tag: str) -> None:
        rec: dict = {"updates": int(step), "tag": tag,
                     "elapsed": round(time.time() - t0, 1),
                     "epochs": round(step * bs / max(n_train, 1), 3)}
        # every split is evaluated in each condition; ``seen``/``hold``/``unsup`` are then
        # the average over the taught conditions, which is accuracy on the training
        # distribution, and the per-condition values are kept beside them
        by_mode: dict[str, list[float]] = {}
        for split_name in ("taught", "holdout", "unsupported"):
            for mode in pools["eval_modes"]:
                suffix = spec.MODE_SUFFIX.get(mode, f"_{mode.lower()}")
                m = evaluate_split(model, retina, splits[f"{split_name}{suffix}"], plan,
                                   device=device)
                if not m:
                    continue
                for k, v in m.items():
                    if k != "n":
                        rec[f"{split_name}{suffix}_{k}"] = v
                if mode in pools["train_modes"]:
                    by_mode.setdefault(split_name, []).append(m["acc"])
        for split_name in ("taught", "holdout", "unsupported"):
            if by_mode.get(split_name):
                short = {"taught": "seen", "holdout": "hold",
                         "unsupported": "unsup"}[split_name]
                rec[f"{short}_acc"] = float(np.mean(by_mode[split_name]))
        m = evaluate_split(model, retina, train, plan, device=device)
        for k, v in m.items():
            if k != "n":
                rec[f"train_{k}"] = v
        # the internal probes read the state in the primary taught condition
        rec.update(internal_metrics(model, retina, splits["taught"], plan,
                                    probe_steps=cell.probe_steps, device=device))
        ctx.metric(**rec)
        history.append(rec)
        if show_progress:
            probes = " ".join(f"{k[len('probe_'):]}={v:.3f}"
                              for k, v in rec.items()
                              if k.startswith("probe_") and v == v)
            taught = " ".join(
                "{}={:.3f}".format(mode, rec.get(
                    "taught{}_acc".format(spec.MODE_SUFFIX.get(mode, "")), float("nan")))
                for mode in pools["eval_modes"])
            held = " ".join(
                "{}={:.3f}".format(mode, rec.get(
                    "holdout{}_acc".format(spec.MODE_SUFFIX.get(mode, "")), float("nan")))
                for mode in pools["eval_modes"])
            log.info("  [%7d upd | %5.1f ep] train %s | seen %s [%s] | hold %s [%s] | %s",
                     step, rec["epochs"], _fmt(rec.get("train_acc")),
                     _fmt(rec.get("seen_acc")), taught, _fmt(rec.get("hold_acc")),
                     held, probes)

    def _fmt(v):
        return "  --  " if v is None else f"{v:.4f}"


    record(0, "initial")
    todo = [p for p in sorted(set(cell.probes)) if p > 0]
    ckpt_dir = ctx.dir / "ckpt"
    step = 0
    while step < cell.budget:
        perm = rng.permutation(n_train)
        for i in range(0, n_train, bs):
            if step >= cell.budget:
                break
            idx = perm[i:i + bs]
            cols = phase_columns(train["images"][idx], retina, plan, device)
            logits, _ = run_batch(model, cols, plan)
            loss = lossf(logits, y_all[idx].to(device))
            loss = loss + model.gain_penalty(cfg.model_cfg.lambda_gain)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip)
            opt.step()
            step += 1
            if todo and step >= todo[0]:
                record(step, f"updates={todo.pop(0)}")
            if step % cell.ckpt_every == 0:
                _checkpoint(model, ckpt_dir / f"roll_{step:08d}.pt")
                _rotate(ckpt_dir)
            if step % cell.permanent_every == 0:
                _checkpoint(model, ckpt_dir / f"step_{step:08d}.pt")
    _checkpoint(model, ckpt_dir / "final.pt")
    if history[-1]["updates"] != cell.budget:
        record(cell.budget, "final")

    final = history[-1]
    summary = {
        "run_id": ctx.run_id, "cell": cell.run, "line": cell.line, "task": cell.task,
        "brain": cell.brain, "model": "M1", "circuit": cfg.data.circuit,
        "graph": cfg.data.graph, "model_seed": cell.seed,
        "budget": cell.budget, "updates_run": int(step),
        "n_train_rows": n_train, "n_taught_items": len(taught),
        "n_holdout_items": len(held), "n_unsupported_items": len(pools["unsupported_items"]),
        "recurrent_steps_per_update": plan.n_steps(),
        "probes": list(cell.probes), "warm_start": warm or None,
        "pool_key": pools["pool_key"],
        "train_modes": pools["train_modes"], "eval_modes": pools["eval_modes"],
        "stimulus_fingerprint": pools["stimulus_fingerprint"],
        "spec_fingerprint": spec.fingerprint(),
        "optim": spec.OPTIM, "dynamics": spec.DYNAMICS,
        "taught_items": [list(p) for p in pools["taught_items"]],
        "holdout_items": [list(p) for p in pools["holdout_items"]],
        "unsupported_items": [list(p) for p in pools["unsupported_items"]],
        "history": history,
        "final": final,
        "wall_seconds": round(ctx.elapsed(), 1),
    }
    (ctx.dir / "full_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    ctx.finish("ok", stage=cell.tag, task=cell.task, model="M1",
               circuit=cfg.data.circuit, graph=cfg.data.graph, model_seed=cell.seed,
               wall_seconds=round(ctx.elapsed(), 1))
    log.info("PHASE I DONE %s | %d updates in %.1f min | train %s seen %s hold %s",
             cell.run, step, ctx.elapsed() / 60, _fmt(final.get("train_acc")),
             _fmt(final.get("seen_acc")), _fmt(final.get("hold_acc")))
    return summary
