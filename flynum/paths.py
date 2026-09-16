"""Canonical filesystem layout.

Everything heavy lives on the project drive (I:), *never* on C: (which had only
~13 GB free).  Importing this module also pins temp/cache environment variables
so that no third-party library silently writes to C:.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
CONFIGS = ROOT / "configs"
LOGS = ROOT / "logs"
CACHE = ROOT / ".cache"

ALL_DIRS = (
    DATA,
    DATA_RAW,
    DATA_PROCESSED,
    RUNS,
    REPORTS,
    FIGURES,
    CONFIGS,
    LOGS,
    CACHE,
)

# --------------------------------------------------------------------------- #
# MaleCNS v1.0 public release
# --------------------------------------------------------------------------- #
MALECNS_BASE = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
)

#: Files required for the core experiment (Stages 0-5).  Total ~1.06 GB, every
#: single file is below the 5 GiB approval threshold.
REQUIRED_FILES: dict[str, str] = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome_weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}

#: Files that exist upstream but are deliberately NOT fetched by default.
#: Anything over 5 GiB requires explicit user approval.
OPTIONAL_FILES: dict[str, str] = {
    "body_stats": "body-stats-male-cns-v1.0-minconf-0.5.feather",
    "tbar_neurotransmitters": "tbar-neurotransmitters-male-cns-v1.0.feather",
    "syn_points": "syn-points-male-cns-v1.0-minconf-0.5.feather",
    "syn_partners": "syn-partners-male-cns-v1.0-minconf-0.5.feather",
}

#: Hard gate: refuse to download any single file larger than this without an
#: explicit approval flag.
SIZE_APPROVAL_LIMIT = 5 * 1024**3  # 5 GiB

PROXY = "http://127.0.0.1:7897"


def ensure_dirs() -> None:
    """Create every project directory (idempotent)."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def pin_temp_env() -> None:
    """Redirect temp/cache locations off the small system drive."""
    tmp = CACHE / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for var in ("TMP", "TEMP", "TMPDIR"):
        os.environ[var] = str(tmp)
    os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "mpl"))
    os.environ.setdefault("PYTHONUTF8", "1")


def raw_path(key: str) -> Path:
    """Absolute path of a raw MaleCNS file by short key."""
    try:
        name = REQUIRED_FILES[key]
    except KeyError:
        name = OPTIONAL_FILES[key]
    return DATA_RAW / name


def processed_path(name: str) -> Path:
    return DATA_PROCESSED / name


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


ensure_dirs()
pin_temp_env()
