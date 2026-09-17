"""How much of each counting condition is answerable without counting.

This is a thin wrapper around :mod:`flynum.phase1.cues`, which is also what the pre-flight
check uses, so the table quoted in the plan and the table quoted in the report cannot drift
apart.  Run it after any change to the stimulus generator.

    python scripts/30_condition_cues.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum.config import StimulusConfig  # noqa: E402
from flynum.phase1 import cues, spec  # noqa: E402


def main() -> int:
    sc = StimulusConfig()
    for k, v in spec.STIMULUS.items():
        setattr(sc, k, v)
    table = cues.cue_table(sc)
    print("{:6s} {:>7s} {:>7s} {:>7s} {:>7s}".format("mode", "chance", *cues.CUE_FIELDS))
    for mode, row in table.items():
        print("{:6s} {:7.3f} {:7.3f} {:7.3f} {:7.3f}".format(
            mode, row["chance"], row["ink"], row["spread"], row["radius_mean"]))
    print("\nnearest-centroid accuracy from that single scalar; 'radius' is the mean "
          "per-dot radius")
    ceiling = cues.mixture_ceiling(table, spec.TRAIN_MODES)
    print("taught mixture {}: best single scalar {} at {:.3f} (chance {:.3f})".format(
        "+".join(spec.TRAIN_MODES), ceiling["cue"], ceiling["accuracy"],
        ceiling["chance"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
