"""Run the remaining study blocks back to back, one at a time, on one GPU.

Why a queue rather than parallel workers: measured throughput on this card is
0.0357 epochs/s with one worker, 0.0284 with two and 0.0288 with four.  The GPU
is compute-saturated at a single run (100% utilisation at 1 and 4 workers), so
extra workers only add contention -- four workers made an epoch 5x slower, not
4x faster.  Blocks therefore run strictly sequentially, and the parallelism that
does pay off (independent shuffled graphs) lives on the CPU side.

Each block writes a marker with its status, so the queue is resumable: a rerun
skips blocks that already succeeded unless ``--force`` is given.  A failing block
is recorded and the queue moves on, because the remaining blocks are independent
and one bad block should not cost the whole night.

Usage::

    python scripts/18_queue.py --dry-run
    python scripts/18_queue.py                       # every block, in order
    python scripts/18_queue.py --blocks fxu signed
    python scripts/18_queue.py --force --blocks lesion
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flynum import paths  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402

#: Trained ``core`` + 20k real counting model used as (a) the reference point for
#: the fixed-update control and (b) the warm start and lesion source.  It is the
#: seed-0 run of the block that was verified byte-identical to the replication
#: configuration, so it is the same model the replication averages over.
REF_REAL = "s2_count_M1_core_real_N20000_m0_sh0_7ab4bc3c46"
REF_SHUFFLED = "s2_count_M1_core_shuffled_N20000_m0_sh0_22058eb561"

#: The lesion panel runs on all three seeds of the headline condition, not one.  A
#: single-seed knockout is exactly the kind of result that turns out to be seed luck,
#: and the claim it supports -- LC11 silencing impairs numerosity while LC10a does not
#: -- is a double dissociation, so it needs a spread across models rather than one
#: model's number.  Each panel costs ~20 min.
LESION_SOURCES = (
    REF_REAL,
    "s2_count_M1_core_real_N20000_m1_sh0_cc44762d6a",
    "s2_count_M1_core_real_N20000_m2_sh0_758f75b4ab",
)


def _blocks() -> dict[str, dict]:
    """Ordered block table.  Order follows the priority the user set: get ``core``
    correct first, then the one biological upgrade, then the curriculum, then the
    lesion, and only afterwards anything that costs full-circuit compute."""
    return {
        # ---- 1. the confound the user identified ------------------------------ #
        # ---- 1. is the headline comparison even valid? --------------------- #
        # Cheapest foundational check, so it runs first: 3 runs, 1.6 h.  The
        # degree/weight-preserving shuffle sits 29% below the real circuit in spectral
        # radius, so the ordinary real-vs-shuffled gap may be an operating-point
        # difference rather than a structural one.  If so, that changes how the
        # headline, the curriculum's shuffled arm and the fixed-update control are all
        # read -- information worth having in 1.6 h rather than 14 h in.
        "spectral": {
            "why": "spectrally matched shuffled control (real rho=3.946, shuffled 2.797)",
            "cmd": ["scripts/15_run_parallel.py", "--grid", "spectral_control",
                    "--workers", "1", "--seeds", "3"],
            "est_min": 95,
        },
        # ---- 2. the confound the user identified ----------------------------- #
        "fxu": {
            "why": "equal optimiser-update control for the sample-efficiency knee",
            "cmd": ["scripts/15_run_parallel.py", "--grid", "fixed_updates",
                    "--workers", "1"],
            "est_min": 470,
        },
        # ---- 3. Fly-v2: signed synapses ------------------------------------- #
        "signed": {
            "why": "signed synapses bound h(t); counting under them, plus a "
                   "scale-matched unsigned control",
            "cmd": ["scripts/15_run_parallel.py", "--grid", "signed_count",
                    "--workers", "1", "--seeds", "3", "--w-scale", "0.5"],
            "est_min": 270,
        },
        # ---- 4. curriculum addition: the 2x2 transfer table ------------------ #
        "curr_real_pre": {
            "why": "real + count-pretrained, held-out pair 2+3/3+2",
            "cmd": ["scripts/16_run_curriculum.py", "--graph", "real",
                    "--pretrain", REF_REAL, "--seeds", "0", "1", "2"],
            "est_min": 75,
        },
        "curr_real_scr": {
            "why": "real + scratch (does count training transfer at all?)",
            "cmd": ["scripts/16_run_curriculum.py", "--graph", "real",
                    "--scratch", "--seeds", "0", "1", "2"],
            "est_min": 75,
        },
        "curr_shuf_pre": {
            "why": "shuffled + count-pretrained (is the transfer topology specific?)",
            "cmd": ["scripts/16_run_curriculum.py", "--graph", "shuffled",
                    "--pretrain", REF_SHUFFLED, "--seeds", "0", "1", "2"],
            "est_min": 75,
        },
        "curr_shuf_scr": {
            "why": "shuffled + scratch",
            "cmd": ["scripts/16_run_curriculum.py", "--graph", "shuffled",
                    "--scratch", "--seeds", "0", "1", "2"],
            "est_min": 75,
        },
        # ---- 5. is the unseen-pair result specific to which pair is held out? - #
        "lpo": {
            "why": "same 2x2 at held-out pairs 1+3, 1+4, 2+4 (one seed each)",
            "cmd": None,  # expanded below into several invocations
            "est_min": 150,
        },
        # ---- 5. LC11 / LC10a / random knockout ------------------------------ #
        "lesion": {
            "why": "virtual knockout panel on all three seeds of the headline model",
            "cmd": None,  # expanded below: one panel per source model
            "est_min": 60,
        },
        # ---- plumbing self-test --------------------------------------------- #
        # Runs on the CPU behind an isolated stimulus cache, so it validates this
        # script's subprocess/log/marker machinery without touching the GPU:
        #   python scripts/18_queue.py --blocks smoke --force
        "smoke": {
            "why": "queue self-test (CPU only; verifies subprocess + markers)",
            "cmd": ["scripts/19_smoke.py"],
            "est_min": 5,
        },
    }


def _expand(name: str, spec: dict) -> list[list[str]]:
    """A block may expand to several sequential invocations."""
    if name == "lesion":
        return [["scripts/17_run_lesion.py", "--source", src,
                 "--random-repeats", "5"] for src in LESION_SOURCES]
    if name != "lpo":
        return [spec["cmd"]]
    cmds = []
    for holdout in ("13", "14", "24"):
        for graph, pre in (("real", REF_REAL), ("shuffled", REF_SHUFFLED)):
            cmds.append(["scripts/16_run_curriculum.py", "--graph", graph,
                         "--pretrain", pre, "--holdout", holdout, "--seeds", "0"])
    return cmds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="rerun blocks that succeeded")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log = get_logger("queue")
    table = _blocks()
    names = args.blocks or list(table)
    unknown = [n for n in names if n not in table]
    if unknown:
        log.error("unknown blocks: %s (known: %s)", unknown, list(table))
        return 2

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    markers = paths.DATA_PROCESSED / "queue"
    markers.mkdir(parents=True, exist_ok=True)
    status_path = markers / "status.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {}
    )

    total_est = sum(table[n]["est_min"] for n in names)
    log.info("queue: %d blocks, ~%.1f h estimated", len(names), total_est / 60)
    for n in names:
        done = status.get(n, {}).get("status") == "ok"
        log.info("  %-15s %-6s est %4d min | %s",
                 n, "DONE" if done else "todo", table[n]["est_min"], table[n]["why"])
    if args.dry_run:
        prereq_ok = True
        for n in names:
            for cmd in _expand(n, table[n]):
                log.info("    %s %s", sys.executable, " ".join(cmd))
                # A block that warm-starts or lesions needs a real checkpoint; a
                # missing one would only surface hours later, after the earlier
                # blocks had already burned the GPU.
                for flag in ("--pretrain", "--source"):
                    if flag in cmd:
                        ref = cmd[cmd.index(flag) + 1]
                        ck = paths.run_dir(ref) / "ckpt" / "best.pt"
                        if not ck.exists():
                            prereq_ok = False
                            log.error("      MISSING PREREQUISITE: %s has no %s",
                                      ref, ck.relative_to(paths.ROOT))
        log.info("prerequisites: %s", "all present" if prereq_ok else "INCOMPLETE (see above)")
        return 0 if prereq_ok else 1

    t_all = time.time()
    for n in names:
        if status.get(n, {}).get("status") == "ok" and not args.force:
            log.info("skip %s (already ok)", n)
            continue
        cmds = _expand(n, table[n])
        log.info("=" * 78)
        log.info("BLOCK %s | %d invocation(s) | %s", n, len(cmds), table[n]["why"])
        t0 = time.time()
        rc = 0
        # One file handle per block, appended to: the log survives a crash and can
        # be tailed while the queue runs.
        block_log = logs / f"queue_{n}.log"
        with block_log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n\n===== queue block {n} started "
                     f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            for cmd in cmds:
                fh.write(f"\n--- {' '.join(cmd)}\n")
                fh.flush()
                proc = subprocess.run(
                    [sys.executable, *cmd], cwd=str(ROOT), stdout=fh,
                    stderr=subprocess.STDOUT, env=None,
                )
                if proc.returncode != 0:
                    rc = proc.returncode
                    fh.write(f"\n--- FAILED rc={rc}\n")
                    break
        mins = (time.time() - t0) / 60
        status[n] = {
            "status": "ok" if rc == 0 else "error",
            "returncode": rc,
            "minutes": round(mins, 1),
            "log": str(block_log.relative_to(ROOT)),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        log.info("BLOCK %s %s in %.1f min -> %s", n, "ok" if rc == 0 else f"FAILED rc={rc}",
                 mins, block_log.name)

    bad = [n for n in names if status.get(n, {}).get("status") != "ok"]
    log.info("=" * 78)
    log.info("queue finished in %.1f h | %d/%d blocks ok",
             (time.time() - t_all) / 3600, len(names) - len(bad), len(names))
    if bad:
        log.error("blocks needing attention: %s", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
