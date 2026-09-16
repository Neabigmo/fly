"""Structured logging and run-directory management.

Every experiment writes exactly one directory::

    runs/<run_id>/
        config.json     fully-resolved configuration + code/env fingerprint
        env.json        python / torch / CUDA / GPU / driver versions
        metrics.jsonl   one JSON object per epoch (and per evaluation)
        log.txt         human-readable UTF-8 log
        ckpt/           best.pt, last.pt
        figures/        per-run figures
        summary.json    final metrics, confusion matrices, status

and appends one row to ``runs/index.csv`` so the whole study can be queried
from a single table.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_INDEX_FIELDS = [
    "run_id",
    "stage",
    "task",
    "model",
    "circuit",
    "graph",
    "n_train",
    "model_seed",
    "shuffle_seed",
    "status",
    "best_val_acc",
    "test_acc_a",
    "test_acc_b",
    "test_acc_c",
    "n_params",
    "epochs_run",
    "wall_seconds",
    "finished_at",
]


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


def environment_fingerprint() -> dict[str, Any]:
    """Collect reproducible environment facts (written into every run)."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_hash": _git_hash(),
        "cwd": str(paths.ROOT),
    }
    try:
        import torch

        env.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
            }
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env.update(
                {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_capability": list(torch.cuda.get_device_capability(0)),
                    "gpu_total_memory_gb": round(props.total_memory / 1e9, 2),
                }
            )
    except ImportError:
        env["torch"] = "missing"
    for mod in ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "matplotlib"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except ImportError:
            env[mod] = "missing"
    return env


class RunContext:
    """Owns one run directory and its loggers.

    If a directory for this configuration already holds metrics, a numbered
    suffix is used instead.  Without this, re-running a configuration silently
    *appends* to the previous ``metrics.jsonl`` -- and two concurrent processes
    with the same configuration would interleave their rows into one file,
    making the logs unusable and hiding the fact that two models were training
    at once.
    """

    def __init__(self, run_id: str, config: dict[str, Any] | None = None):
        base = run_id
        directory = paths.run_dir(base)
        attempt = 1
        while (
            (directory / "metrics.jsonl").exists()
            and (directory / "metrics.jsonl").stat().st_size > 0
        ):
            attempt += 1
            directory = paths.run_dir(f"{base}_r{attempt}")
        self.run_id = directory.name
        self.attempt = attempt
        self.dir = directory
        (self.dir / "ckpt").mkdir(parents=True, exist_ok=True)
        (self.dir / "figures").mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        self._metrics_fh = (self.dir / "metrics.jsonl").open("a", encoding="utf-8")
        if config is not None:
            config = dict(config)
            config["_run_id"] = self.run_id
            config["_attempt"] = attempt
            self.write_config(config)
        self.write_env()
        self.log = self._build_logger()
        if attempt > 1:
            self.log.warning(
                "configuration already has results; using a fresh directory %s "
                "(attempt %d)", self.run_id, attempt,
            )
        self.log.info("run %s started -> %s", self.run_id, self.dir)

    # ------------------------------------------------------------------ #
    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"flynum.run.{self.run_id}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if logger.handlers:
            return logger
        fh = logging.FileHandler(self.dir / "log.txt", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    # ------------------------------------------------------------------ #
    def write_config(self, config: dict[str, Any]) -> None:
        self._dump("config.json", config)

    def write_env(self) -> None:
        self._dump("env.json", environment_fingerprint())

    def _dump(self, name: str, obj: Any) -> None:
        with (self.dir / name).open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------ #
    def metric(self, **kw: Any) -> None:
        """Append one structured metrics record."""
        rec = {"t": round(time.time() - self._t0, 3), "run_id": self.run_id}
        rec.update(kw)
        self._metrics_fh.write(json.dumps(rec, default=str) + "\n")
        self._metrics_fh.flush()

    def elapsed(self) -> float:
        return time.time() - self._t0

    # ------------------------------------------------------------------ #
    def finish(self, status: str, **summary: Any) -> dict[str, Any]:
        """Write summary.json and append a row to runs/index.csv."""
        payload = {
            "run_id": self.run_id,
            "status": status,
            "wall_seconds": round(self.elapsed(), 2),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        payload.update(summary)
        self._dump("summary.json", payload)
        self.log.info("run %s finished: %s (%.1fs)", self.run_id, status, self.elapsed())
        self._metrics_fh.close()
        for h in self.log.handlers:
            if isinstance(h, logging.FileHandler):
                h.close()
                self.log.removeHandler(h)
        append_index_row(payload)
        return payload


def append_index_row(row: dict[str, Any]) -> None:
    """Append one row to the study-wide runs/index.csv (creating it if needed)."""
    index = paths.RUNS / "index.csv"
    exists = index.exists()
    with index.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_INDEX_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _INDEX_FIELDS})


def get_logger(name: str) -> logging.Logger:
    """Plain module logger for scripts that do not own a RunContext."""
    logger = logging.getLogger(f"flynum.{name}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(sh)
    return logger
