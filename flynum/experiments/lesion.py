"""Virtual knockout of identified neuron types (LC11, LC10a, and controls).

The real fly literature reports that silencing **LC11** impairs numerical
discrimination whereas silencing LC10a, the mushroom body or the central complex
does not.  If the same manipulation in the connectome-constrained network also
degrades numerosity while a size-matched random knockout does not, the model is
reproducing a specific neural mechanism rather than merely fitting the task.

Two lesion modes are supported, and they answer different questions:

``readout``
    Remove the target neurons from the readout population only.  Asks whether the
    *decoder* depends on them.
``silence``
    Zero the target neurons' outgoing contribution every timestep.  Asks whether
    the *circuit* depends on them.  This is the closer analogue of the genetic
    silencing used in the fly experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..config import ExperimentConfig
from ..devices import pick_device
from ..logging_utils import RunContext
from ..pipeline import prepare
from ..retina.torch_encoder import TorchRetina
from ..train.data import build_count_data
from ..train.metrics import classification_metrics
from ..train.trainer import encode

DEVICE = pick_device()

#: Cell types used for the lesion panel.  ``lc11`` is the manipulation of
#: interest; ``lc10a`` is the visual control that the fly literature reports as
#: *not* affecting numerosity; ``t4t5`` is a large non-overlapping control.
#:
#: T4a/T5a are motion-pathway cells that live outside the ``core`` circuit, so in
#: the core panel they resolve to zero neurons and are simply not run (the reported
#: ``n_neuron_types`` records the zeros).  That makes LC10a and the size-matched
#: random knockout the operative controls in this study, which is the comparison
#: the user asked for; T4a/T5a are kept in the list so the gap is visible rather
#: than silently dropped when the panel is extended to ``full``.
TARGET_TYPES = ("LC11", "LC10a", "T4a", "T5a")


@dataclass
class LesionSpec:
    name: str
    cell_types: tuple[str, ...] = ()
    #: explicit neuron indices (used for the random size-matched control)
    indices: np.ndarray | None = None
    mode: str = "silence"  # silence | readout


@dataclass
class LesionResult:
    name: str
    n_lesioned: int
    mode: str
    per_mode: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n_lesioned": int(self.n_lesioned),
            "mode": self.mode,
            "per_mode": self.per_mode,
        }


# --------------------------------------------------------------------------- #
def type_indices(prep, cell_types: tuple[str, ...]) -> np.ndarray:
    """Indices (within the subgraph) of every neuron of the given cell types."""
    sub_ann = prep.ann.index_of(prep.subgraph.body_ids)
    types = prep.ann.cell_type[sub_ann]
    return np.nonzero(np.isin(types, list(cell_types)))[0].astype(np.int64)


@torch.no_grad()
def evaluate_with_lesion(
    model,
    retina: TorchRetina,
    data,
    *,
    lesion_idx: np.ndarray | None,
    mode: str = "silence",
    steps: int = 8,
    batch: int = 256,
) -> dict:
    """Accuracy on every test condition with a lesion applied.

    ``silence`` multiplies the lesioned neurons' activity by zero after every
    recurrent step and also removes them from the readout pool, which is what a
    genetic silencing experiment effectively does.
    """
    model.eval()
    # ``readout`` removal is implemented by zeroing the lesioned features rather
    # than by dropping them: the readout is a single linear layer, so
    # W[kept] @ z[kept] + b is identical either way, and zeroing keeps the input
    # width matching the layer (dropping would change the shape).
    zero_mask = None
    if mode == "readout" and lesion_idx is not None and len(lesion_idx):
        les = torch.as_tensor(lesion_idx, device=model.readout_idx.device)
        zero_mask = torch.isin(model.readout_idx, les)

    out: dict = {}
    # ``run_lesion_panel`` passes the condition dict directly while the count
    # experiment owns a ``CountData``; accept both rather than making every caller
    # wrap one in the other.
    tests = data["tests"] if isinstance(data, dict) else data.tests
    for cond, ds in tests.items():
        preds = []
        for i in range(0, len(ds["images"]), batch):
            cols = encode(ds["images"][i : i + batch], retina, DEVICE)
            b = cols.shape[0]
            h = torch.zeros(model.n_neurons, b, device=cols.device)
            bias = (model.bias if model.bias is not None else model.bias_const).unsqueeze(1)
            values = model.edge_values
            history = []
            for _ in range(steps):
                inj = model._inject(cols)
                drive = model.conn.matmul(values, h) + bias + inj
                h = (1.0 - model.alpha) * h + model.alpha * model._act(drive)
                if mode == "silence" and lesion_idx is not None and len(lesion_idx):
                    h[torch.as_tensor(lesion_idx, device=h.device)] = 0.0
                history.append(h)
            pooled = torch.stack(
                [hh.index_select(0, model.readout_idx) for hh in history[-model.readout_window:]]
            ).mean(dim=0)
            if zero_mask is not None:
                pooled = pooled.masked_fill(zero_mask.unsqueeze(1), 0.0)
            preds.append(model.classify(pooled.t())[0].argmax(1).cpu().numpy())
        pred = np.concatenate(preds)
        m = classification_metrics(ds["labels"], pred, model.n_classes)
        out[cond] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "confusion": m["confusion"]}
    return out


# --------------------------------------------------------------------------- #
def run_lesion_panel(
    cfg: ExperimentConfig,
    source_run: str,
    *,
    random_repeats: int = 5,
    seed: int = 0,
    logger=None,
) -> dict:
    """Evaluate a trained counting model under every lesion in the panel."""
    from .. import paths
    from ..logging_utils import get_logger

    log = logger or get_logger("lesion")
    ctx = RunContext(f"8_lesion_{source_run}", cfg.to_dict())
    log = ctx.log

    prep = prepare(cfg, logger=log)
    retina = TorchRetina(prep.encoder, device=DEVICE)
    model = prep.build_model(device=DEVICE)
    ckpt = paths.run_dir(source_run) / "ckpt" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint for {source_run}")
    model.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    )
    log.info("loaded %s (epoch %s)", source_run,
             torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch"))

    data = build_count_data(cfg, logger=log)
    sub_ann = prep.ann.index_of(prep.subgraph.body_ids)

    specs = [LesionSpec("intact", (), None, "silence")]
    for t in TARGET_TYPES:
        idx = type_indices(prep, (t,))
        if len(idx):
            specs.append(LesionSpec(t, (t,), idx, "silence"))
    # LC11 also under readout-only removal, to separate decoder from circuit
    lc11 = type_indices(prep, ("LC11",))
    if len(lc11):
        specs.append(LesionSpec("LC11_readout_only", ("LC11",), lc11, "readout"))

    # size-matched random knockouts
    rng = np.random.default_rng(seed)
    for r in range(random_repeats):
        n = max(len(lc11), 1)
        idx = rng.choice(prep.subgraph.n_neurons, size=n, replace=False).astype(np.int64)
        specs.append(LesionSpec(f"random_{n}_r{r}", (), idx, "silence"))

    results = []
    for spec in specs:
        idx = spec.indices if spec.indices is not None else np.array([], dtype=np.int64)
        res = evaluate_with_lesion(
            model, retina,
            {"tests": {k: v for k, v in data.tests.items()}},
            lesion_idx=idx, mode=spec.mode, steps=cfg.time.steps,
        )
        rr = LesionResult(spec.name, len(idx), spec.mode, res)
        results.append(rr)
        log.info(
            "  %-22s n=%5d  A %.4f  B %.4f  C %.4f  D %.4f",
            spec.name, len(idx), res["A"]["accuracy"], res["B"]["accuracy"],
            res["C"]["accuracy"], res["D"]["accuracy"],
        )

    intact = next(r for r in results if r.name == "intact")
    summary = {
        "run_id": ctx.run_id,
        "source_run": source_run,
        "graph": cfg.data.graph,
        "n_neuron_types": {
            t: int(len(type_indices(prep, (t,)))) for t in TARGET_TYPES
        },
        "lesions": [r.as_dict() for r in results],
        "deltas": {
            r.name: {
                c: round(
                    intact.per_mode[c]["accuracy"] - r.per_mode[c]["accuracy"], 4
                )
                for c in ("A", "B", "C", "D")
            }
            for r in results if r.name != "intact"
        },
    }
    import json as _json

    (paths.DATA_PROCESSED / f"lesion_{Path(source_run).name}.json").write_text(
        _json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (ctx.dir / "full_summary.json").write_text(
        _json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    ctx.finish("ok", stage="8", task="lesion", source_run=source_run,
               wall_seconds=round(ctx.elapsed(), 1))
    return summary
