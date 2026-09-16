"""Stage 0a -- environment self-check.

Verifies the GPU, the conda environment, disk headroom and the proxy, and
writes the fingerprint to data/processed/env_report.json.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.logging_utils import environment_fingerprint, get_logger  # noqa: E402


def main() -> int:
    log = get_logger("env_check")
    env = environment_fingerprint()

    log.info("=" * 78)
    log.info("ENVIRONMENT")
    log.info("=" * 78)
    for key in (
        "python", "executable", "platform", "torch", "cuda_available",
        "cuda_version", "gpu_name", "gpu_capability", "gpu_total_memory_gb",
        "numpy", "pandas", "pyarrow", "scipy", "sklearn", "matplotlib",
    ):
        if key in env:
            log.info("  %-22s %s", key, env[key])

    ok = True
    if not env.get("torch") or env["torch"] == "missing":
        log.error("torch is not importable -- wrong python interpreter?")
        ok = False
    if not env.get("cuda_available"):
        log.error("CUDA is not available")
        ok = False

    # disk headroom: never let heavy artefacts land on a full drive
    log.info("-" * 78)
    for drive in ("I:\\", "C:\\"):
        try:
            usage = shutil.disk_usage(drive)
            log.info(
                "  disk %-4s free %7.1f GB / %7.1f GB",
                drive,
                usage.free / 1024**3,
                usage.total / 1024**3,
            )
            if drive.startswith("I") and usage.free < 50 * 1024**3:
                log.error("project drive I: has less than 50 GB free")
                ok = False
        except OSError:
            pass
    log.info("  temp dir               %s", paths.CACHE / "tmp")

    # proxy reachability
    log.info("-" * 78)
    try:
        from flynum.data.download import head_size

        size, ranges = head_size(paths.MALECNS_BASE + paths.REQUIRED_FILES["annotations"])
        log.info(
            "  proxy %s -> OK (annotations file %.2f MB, ranges=%s)",
            paths.PROXY,
            size / 1024**2,
            ranges,
        )
    except Exception as exc:
        log.error("proxy %s unreachable: %s", paths.PROXY, exc)
        ok = False

    out = paths.DATA_PROCESSED / "env_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2), encoding="utf-8")
    log.info("-" * 78)
    log.info("wrote %s", out)
    log.info("ENV CHECK: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
