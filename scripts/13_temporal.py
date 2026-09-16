"""Stage 5 -- temporal decoding of the addition task.

Loads a trained addition model, records the recurrent state at every timestep and
fits an independent linear probe at each one to decode ``a`` (first count),
``b`` (second count) and ``a+b``.

The shapes of those three curves are the evidence for how the task is solved:

* if ``a`` is decodable during the first epoch and **stays** decodable across the
  blank delay, the network holds a persistent representation of the first count;
* if ``a`` decays to chance during the delay, the network is not using memory and
  any success on the addition task would be a shortcut;
* if ``a+b`` rises only after the second epoch, the sum is being computed rather
  than read off the second stimulus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.analysis.temporal import collect_states, decode_over_time, summarise_decoding  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.logging_utils import RunContext  # noqa: E402
from flynum.pipeline import prepare  # noqa: E402
from flynum.retina.torch_encoder import TorchRetina  # noqa: E402
from flynum.train.data import build_add_data  # noqa: E402


def find_addition_runs() -> list[dict]:
    out = []
    for p in sorted(paths.RUNS.glob("*/summary.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if s.get("task") == "add" and s.get("status") == "ok":
            out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="", help="addition run to analyse")
    ap.add_argument("--n-samples", type=int, default=1200)
    ap.add_argument("--graph", default="real", choices=["real", "shuffled"])
    args = ap.parse_args()

    ctx = RunContext("temporal_decoding")
    log = ctx.log
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runs = find_addition_runs()
    if not runs:
        log.error("no completed addition runs found - run scripts/08_run_addition.py first")
        return 1
    if args.run_id:
        runs = [r for r in runs if r["run_id"] == args.run_id]
    else:
        runs = [r for r in runs if r.get("graph") == args.graph]
    if not runs:
        log.error("no matching addition run")
        return 1
    run = runs[0]
    log.info("analysing addition run %s (graph=%s)", run["run_id"], run.get("graph"))

    cfg = ExperimentConfig()
    cfg.task = "add"
    cfg.data.circuit = run.get("circuit", "core")
    cfg.data.graph = run.get("graph", "real")
    cfg.model = run.get("model", "M1")
    cfg.stimulus.holdout_pairs = tuple(tuple(p) for p in run.get("holdout_pairs", [[2, 3], [3, 2]]))
    ckpt = paths.run_dir(run["run_id"]) / "ckpt" / "best.pt"
    if not ckpt.exists():
        log.error("checkpoint missing: %s", ckpt)
        return 1

    prep = prepare(cfg, logger=log)
    retina = TorchRetina(prep.encoder, device=device)
    model = prep.build_model(device=device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    log.info("loaded checkpoint from epoch %s", state.get("epoch"))

    data = build_add_data(cfg, logger=log)
    tr = data["train"]
    # a balanced slice of every ordered pair, so all sums are represented
    states, meta = collect_states(
        model, retina, tr["image_a"], tr["image_b"], cfg,
        n_samples=args.n_samples, device=device,
    )
    idx = meta["a"][0]
    a_vals = tr["a"][idx]
    b_vals = tr["b"][idx]
    sums = a_vals + b_vals
    log.info(
        "states %s | timesteps: A=%s gap=%s B=%s",
        states.shape, meta["epoch_a"], meta["gap"], meta["epoch_b"],
    )

    dec = decode_over_time(states, {"a": a_vals, "b": b_vals, "sum": sums})
    full = summarise_decoding(dec, meta)

    log.info("-" * 78)
    log.info("linear decodability by timestep (chance = %.3f)", dec["a"]["chance"])
    log.info("%-6s %8s %8s %8s", "t", "a", "b", "a+b")
    for t in range(dec["T"]):
        phase = (
            "A" if t + 1 in meta["epoch_a"]
            else ("gap" if t + 1 in meta["gap"] else "B")
        )
        log.info(
            "%-4d%-2s %8.4f %8.4f %8.4f",
            t + 1, phase, dec["a"]["curve"][t], dec["b"]["curve"][t], dec["sum"]["curve"][t],
        )
    log.info("-" * 78)
    log.info("phase means: %s", json.dumps(
        {k: v for k, v in full.items() if k.endswith("_phases")}, default=str))
    if "memory_retention" in full:
        log.info(
            "first-count memory retention across the delay: %.3f "
            "(1.0 = perfectly held, 0.0 = lost)",
            full["memory_retention"],
        )

    out = paths.DATA_PROCESSED / f"temporal_decoding_{run['run_id']}.json"
    out.write_text(json.dumps({"run_id": run["run_id"], "meta": meta, **full},
                              indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out)

    try:
        from flynum.analysis.figures import plot_temporal_decoding

        fig = plot_temporal_decoding(
            full,
            meta,
            paths.FIGURES / f"fig5_temporal_decoding_{run.get('graph', 'real')}.png",
            title=f"Addition decoding ({run.get('graph', 'real')} connectome)",
        )
        log.info("wrote %s", fig)
    except Exception as exc:  # a plotting problem must not lose the analysis
        log.warning("could not draw the temporal figure: %s", exc)
    ctx.finish("ok", run_id=run["run_id"], memory_retention=full.get("memory_retention"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
