"""Print where the warm-start lookup resolves for each Phase I cell."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.phase1 import spec  # noqa: E402
from flynum.phase1.run import find_checkpoint, run_id_for  # noqa: E402


def main() -> int:
    tag, seed = "10", 0
    for t in spec.FOUNDATION + spec.LINE_A + spec.LINE_B:
        src = spec.WARM_START.get(t.brain, "")
        if not src:
            print(f"{t.run:7s} scratch")
            continue
        found = find_checkpoint(src, tag=tag, seed=seed)
        want = paths.run_dir(run_id_for(src, tag, seed)) / "ckpt" / "final.pt"
        print(f"{t.run:7s} <- {src:7s} expected {want.name} in {want.parent.parent.name}: "
              f"{'FOUND' if found else 'MISSING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
