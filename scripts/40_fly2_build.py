"""Build the fly2 whole-brain cache from the public release.

    python scripts/40_fly2_build.py            # needs data/raw and data/processed/edges.npz
    python scripts/40_fly2_build.py --verify   # also print the populations and the MB loop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402
from fly2 import data as f2data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    log = get_logger("fly2.build")
    cache = paths.DATA_PROCESSED / "fly2"
    d = f2data.build(cache, logger=log)
    log.info("=" * 78)
    for k, v in d.summary().items():
        log.info("  %-26s %s", k, v)
    if args.verify:
        import numpy as np

        e = d
        def count(src, dst):
            return int(np.isin(e.pre, src).__and__(np.isin(e.post, dst)).sum())

        log.info("--- mushroom-body loop, in the cached substrate ---")
        log.info("  VPN -> KC   %8d synapses", count(e.readout_visual, e.kenyon))
        log.info("  KC  -> MBON %8d", count(e.kenyon, e.readout_mb))
        log.info("  DAN -> KC   %8d", count(e.dopamine, e.kenyon))
        log.info("  OA  -> KC   %8d", count(e.octopamine, e.kenyon))
        log.info("  MBON -> VPN %8d", count(e.readout_mb, e.readout_visual))
        log.info("  MB output synapses flagged for lesion: %d", int(d.meta["mb_output_synapses"]))
    log.info("wrote %s", cache / f2data.CACHE_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
