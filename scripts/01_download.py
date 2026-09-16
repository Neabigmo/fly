"""Stage 0b -- download the MaleCNS v1.0 files needed for the core experiment.

Every file is HEAD-probed first and its size printed.  Anything above 5 GiB is
refused unless ``--approve`` is given, implementing the user's rule that large
downloads need explicit approval.

Core set (all below the limit, ~1.06 GB total):
    body-annotations        13.81 MB
    body-neurotransmitters  41.28 MB
    connectome-weights    1002.54 MB
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.data import download as dl  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument(
        "--approve",
        action="store_true",
        help="user has approved downloads larger than 5 GiB",
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="optional keys to fetch as well (e.g. tbar_neurotransmitters)",
    )
    args = ap.parse_args()
    log = get_logger("download")

    log.info("=" * 78)
    log.info("MaleCNS v1.0 file inventory (HEAD probe, proxy %s)", paths.PROXY)
    log.info("=" * 78)
    infos = dl.probe_all()
    for info in infos:
        log.info("  %s", info.line())
    required_total = sum(
        i.size_bytes for i in infos if i.key in paths.REQUIRED_FILES and i.size_bytes > 0
    )
    log.info(
        "  %-58s %9.2f MB  <-- core set, NO approval needed",
        "TOTAL (required)",
        required_total / 1024**2,
    )

    over = [i for i in infos if i.over_limit]
    if over and not args.approve:
        log.warning("-" * 78)
        log.warning("NOT downloading (over 5 GiB, needs approval):")
        for i in over:
            log.warning("   %s  (%.2f GiB)", i.name, i.size_bytes / 1024**3)
        log.warning("   -> re-run with --approve if the user has approved these.")

    log.info("-" * 78)
    paths_to = dl.download_required(
        force=args.force,
        logger=log,
    )
    for key, p in paths_to.items():
        log.info("  %-22s %s (%.2f MB)", key, p.name, p.stat().st_size / 1024**2)

    for key in args.include:
        try:
            p = dl.download_file(key, force=args.force, allow_over_limit=args.approve, logger=log)
            log.info("  %-22s %s (%.2f MB)", key, p.name, p.stat().st_size / 1024**2)
        except dl.ApprovalRequired as exc:
            log.warning("%s", exc)

    dl.write_download_manifest(logger=log)
    log.info("DOWNLOAD: DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
