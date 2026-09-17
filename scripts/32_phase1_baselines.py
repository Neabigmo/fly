"""The shortcut ceiling for every Phase I task: what a cue-follower could reach.

A trained accuracy is only evidence about computation if it is compared against what can be
reached *without* computing.  This script builds that comparison from the stimulus
statistics themselves:

* per-group **ink** (total brightness) and per-group **mean radius**, the two statistics
  any of the three conditions leaks;
* a **nearest-centroid** classifier on the arithmetic combination of those scalars that
  matches the task's own answer (sum for addition, signed difference for rule selection,
  ``a + b - c`` for the two-step task) -- this is what a network that reads the cue and
  then does the arithmetic on it would score;
* a **split-half ridge regression** on the raw per-group scalars, which is a stronger
  linear cue-follower and does not need to guess the right combination.

The larger of the two is reported as the ceiling.  A cell below its ceiling learned
nothing the cue did not already give away; a cell above it in *every* condition cannot be
explained by any of these statistics.

    python scripts/32_phase1_baselines.py            # every task, CPU only
    python scripts/32_phase1_baselines.py --json data/processed/phase1_cue_ceilings.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from flynum import paths  # noqa: E402
from flynum.config import StimulusConfig  # noqa: E402
from flynum.phase1 import spec, tasks  # noqa: E402
from flynum.stimuli import dots  # noqa: E402


def group_scalars(item: tuple[int, ...], sc, rng, mode: str, n_rep: int,
                  content: tuple[str, ...]) -> dict[int, tuple[float, float]]:
    """``{content index: (ink, radius_mean)}`` for every dot group of one item.

    Keyed by the *content* index so that the operator positions stay addressable, which is
    what the arithmetic combination below needs; collapsing the groups into a bare list
    silently shifted the second operand of a rule-selection item onto the operator.
    """
    out: dict[int, tuple[float, float]] = {}
    for i, kind in enumerate(content):
        if kind != "dot":
            continue
        inks, radii = [], []
        for _ in range(n_rep):
            st = dots.make_count_stimulus(int(item[i]), sc, rng, mode=mode)
            inks.append(st.ink)
            radii.append(st.radius_mean)
        out[i] = (float(np.mean(inks)), float(np.mean(radii)))
    return out


def combine(values: dict[int, float], task_name: str, item: tuple[int, ...]) -> float:
    """Apply the task's own arithmetic to a per-group scalar.

    ``add`` -> a + b, ``addsub`` -> a + b or a - b following the glyph, ``two_step`` ->
    a + b - c, counting -> the single group.  This is what a network that reads the cue and
    then does the arithmetic on it would score, which is the number a trained accuracy has
    to be compared against.
    """
    if task_name == "count":
        return values[0]
    if task_name in ("add", "cyc7"):
        return values[0] + values[1]
    if task_name == "addsub":
        return values[0] + values[2] if item[1] == 0 else values[0] - values[2]
    if task_name == "two_step":
        return values[0] + values[2] - values[4]
    raise ValueError(task_name)


def centroid_accuracy(feature: np.ndarray, y: np.ndarray) -> float:
    classes = np.unique(y)
    means = np.array([feature[y == c].mean() for c in classes])
    pred = classes[np.abs(feature[:, None] - means[None, :]).argmin(1)]
    return float((pred == y).mean())


def ridge_accuracy(x: np.ndarray, y: np.ndarray, n_classes: int, *,
                   lam: float = 1e-2) -> float:
    """Split-half linear classifier, fitted in closed form on one-hot targets.

    Solved in the dual when there are more features than rows (the two-step task
    concatenates five phases of 1,771 columns, so the primal system is 8,855 squared and
    the dual is 1,200 squared; both give the same predictions).
    """
    n = len(y)
    g = np.random.default_rng(0)
    perm = g.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    xs = (x - mu) / sd
    ytr = np.eye(n_classes)[y[tr]]
    xtr = xs[tr]
    if xtr.shape[1] <= xtr.shape[0]:
        w = np.linalg.solve(xtr.T @ xtr + lam * np.eye(xtr.shape[1]), xtr.T @ ytr)
    else:
        dual = xtr @ xtr.T + lam * np.eye(xtr.shape[0])
        w = xtr.T @ np.linalg.solve(dual, ytr)
    return float((np.argmax(xs[te] @ w, 1) == y[te]).mean())


def retina_readout(sc, *, per_item: int, seed: int = 7, logger=None,
                   mlp_steps: int = 400) -> dict:
    """Best accuracy reachable from the fixed retina *without* the recurrence.

    Two decoders of the same column activations: a split-half ridge (linear) and a small
    multilayer perceptron, both fitted on all three conditions pooled and scored per
    condition.  This is the reference that says whether the recurrent circuit is doing
    anything: a trained network that scores no better than this has learned nothing the
    retina did not already expose.

    ``per_item`` matters here and not in the cue table: with 1,771 features per phase, a
    few dozen rows cannot fit a seven-way decoder at all, and the first version of this
    measurement reported 0.14 for the natural counting condition -- where brightness is a
    linear function of the pixels and therefore trivially available -- purely because it
    had 35 training rows.
    """
    import torch
    import torch.nn as nn

    from flynum.phase1.run import Cell, make_config
    from flynum.pipeline import prepare
    from flynum.retina.torch_encoder import TorchRetina

    cfg = make_config(Cell(run="baseline", task="count", brain="scratch", budget=1,
                           probes=(0,)))
    for k, v in spec.STIMULUS.items():
        setattr(cfg.stimulus, k, v)
    prep = prepare(cfg, logger=logger, n_classes=7, label_offset=1)
    retina = TorchRetina(prep.encoder, device="cpu")

    out: dict = {}
    for name, task in tasks.TASKS.items():
        rng = np.random.default_rng(seed)
        feats, labels, modes = [], [], []
        for mode in spec.EVAL_MODES:
            for item in task.items:
                for _ in range(per_item):
                    imgs = [tasks.render_phase(task, i, item, sc, rng, mode=mode)
                            for i in range(len(task.content))]
                    feats.append(np.stack(imgs))
                    labels.append(task.answer(item))
                    modes.append(mode)
        x = torch.as_tensor(np.asarray(feats, dtype=np.float32))
        with torch.no_grad():
            enc = retina.encode(x.reshape(-1, sc.image_size, sc.image_size))
        enc = enc.reshape(len(x), -1).numpy()
        y = np.asarray(labels)
        mod = np.asarray(modes)

        acc: dict = {}
        for mode in spec.EVAL_MODES:
            acc[mode] = {"linear": ridge_accuracy(enc[mod == mode], y[mod == mode],
                                                  task.n_classes)}
        # one MLP on all conditions, scored per condition; split by row so training never
        # sees the rows it is scored on
        g = np.random.default_rng(seed)
        perm = g.permutation(len(enc))
        tr, te = perm[: int(0.7 * len(perm))], perm[int(0.7 * len(perm)):]
        mu, sd = enc[tr].mean(0), enc[tr].std(0) + 1e-8
        xt = torch.as_tensor((enc - mu) / sd, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.long)
        net = nn.Sequential(nn.Linear(xt.shape[1], 256), nn.ReLU(),
                            nn.Linear(256, task.n_classes))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        for _ in range(mlp_steps):
            opt.zero_grad(set_to_none=True)
            lossf(net(xt[tr]), yt[tr]).backward()
            opt.step()
        with torch.no_grad():
            pred = net(xt).argmax(1).numpy()
        for mode in spec.EVAL_MODES:
            sel = (mod == mode) & np.isin(np.arange(len(enc)), te)
            acc[mode]["mlp"] = float((pred[sel] == y[sel]).mean()) if sel.any() else None
        out[name] = acc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8, help="layouts averaged per group")
    ap.add_argument("--per-item", type=int, default=40, help="layout samples per item")
    ap.add_argument("--json", default="")
    ap.add_argument("--retina-per-item", type=int, default=150)
    ap.add_argument("--no-retina", action="store_true",
                    help="skip the fixed-retina read-out baseline (it needs the circuit)")
    args = ap.parse_args()

    sc = StimulusConfig()
    for k, v in spec.STIMULUS.items():
        setattr(sc, k, v)
    rng = np.random.default_rng(2024)
    area = float(spec.STIMULUS["area_target"])
    table: dict = {}

    print(f"{'task':9s} {'cond':4s} {'items':>5s} {'chance':>7s} {'ink-NC':>7s} "
          f"{'size-NC':>8s} {'ridge':>7s} {'ceiling':>8s}")
    for name, task in tasks.TASKS.items():
        table[name] = {}
        for mode in spec.EVAL_MODES:
            rows, labels, feats = [], [], []
            for item in task.items:
                for _ in range(args.per_item):
                    gs = group_scalars(item, sc, rng, mode, args.reps, task.content)
                    inks = {i: v[0] for i, v in gs.items()}
                    counts = {i: area / (np.pi * max(v[1], 1e-6) ** 2)
                              for i, v in gs.items()}
                    rows.append((combine(inks, name, item), combine(counts, name, item)))
                    feats.append([v for i in sorted(gs) for v in gs[i]])
                    labels.append(task.answer(item))
            y = np.asarray(labels)
            rows_a = np.asarray(rows)
            feats_a = np.asarray(feats)
            ink_nc = centroid_accuracy(rows_a[:, 0], y)
            size_nc = centroid_accuracy(rows_a[:, 1], y)
            ridge = ridge_accuracy(feats_a, y, task.n_classes)
            ceiling = max(ink_nc, size_nc, ridge)
            table[name][mode] = {
                "n_items": len(task.items), "chance": 1.0 / task.n_classes,
                "ink_centroid": ink_nc, "size_centroid": size_nc,
                "ridge_on_scalars": ridge, "ceiling": ceiling,
            }
            print(f"{name:9s} {mode:4s} {len(task.items):5d} {1 / task.n_classes:7.3f} "
                  f"{ink_nc:7.3f} {size_nc:8.3f} {ridge:7.3f} {ceiling:8.3f}")
    for name in table:
        taught = [table[name][m]["ceiling"] for m in spec.TRAIN_MODES]
        print(f"  {name:9s} taught-mixture ceiling: {np.mean(taught):.3f}")

    if not args.no_retina:
        print(f"\nfixed-retina decoders (no recurrence at all), "
              f"{args.retina_per_item} layouts per item:")
        print(f"{'task':9s} {'cond':4s} {'linear':>8s} {'MLP':>8s} {'chance':>8s}")
        retina = retina_readout(sc, per_item=args.retina_per_item)
        for name in table:
            for mode in spec.EVAL_MODES:
                row = retina[name][mode]
                table[name][mode]["retina_linear"] = row["linear"]
                table[name][mode]["retina_mlp"] = row["mlp"]
                print(f"{name:9s} {mode:4s} {row['linear']:8.3f} "
                      f"{(row['mlp'] if row['mlp'] is not None else float('nan')):8.3f} "
                      f"{1 / tasks.TASKS[name].n_classes:8.3f}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
