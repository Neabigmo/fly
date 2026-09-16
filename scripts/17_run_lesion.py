"""Stage 8b -- virtual knockout panel on a trained counting model.

Reports accuracy on every test condition after silencing LC11, LC10a, T4a/T5a, or
a size-matched random set of neurons, plus LC11 removed from the readout only.

The scientific question is specificity: the fly literature reports that LC11
silencing impairs numerical discrimination while LC10a silencing does not.  If the
model reproduces that pattern -- and a random knockout of the same size does
nothing -- the connectome is carrying a specific mechanism rather than an
undifferentiated mass of recurrence.

Usage::

    python scripts/17_run_lesion.py --source 7_rep_real_s0 --config runs/7_rep_real_s0/config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.experiments.lesion import run_lesion_panel  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="run id of a trained counting model")
    ap.add_argument("--from-config", default="",
                    help="reuse the source run's config.json (recommended: the lesion "
                         "must be evaluated under exactly the training configuration)")
    ap.add_argument("--random-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log = get_logger("lesion_cli")
    cfg_path = Path(args.from_config) if args.from_config else (
        paths.run_dir(args.source) / "config.json"
    )
    if not cfg_path.exists():
        log.error("config not found: %s", cfg_path)
        return 1
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    # strict=False: a run's config.json carries annotation keys (curriculum,
    # sign_report) alongside the configuration itself.
    cfg = ExperimentConfig.from_dict(
        {k: v for k, v in raw.items() if not k.startswith("_")}, strict=False
    )
    log.info(
        "lesion panel | source=%s | graph=%s circuit=%s signed=%s w_scale=%.3g",
        args.source, cfg.data.graph, cfg.data.circuit,
        cfg.model_cfg.signed_synapses, cfg.model_cfg.w_scale,
    )
    summary = run_lesion_panel(
        cfg, args.source, random_repeats=args.random_repeats, seed=args.seed, logger=log
    )
    log.info("neuron counts: %s", summary["n_neuron_types"])
    log.info("delta (intact - lesioned), positive = impairment:")
    for name, d in summary["deltas"].items():
        log.info("  %-22s A %+.4f  B %+.4f  C %+.4f  D %+.4f",
                 name, d["A"], d["B"], d["C"], d["D"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
