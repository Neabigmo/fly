"""Edge-list construction from ``connectome-weights-...-minconf-0.5.feather``.

Measured facts about the source file (recorded in the QC report so every run
can be audited against them):

===========================  =============
total rows                   151,856,684
unique raw segments           88,384,522
rows with both endpoints
present in the annotation
table (**usable universe**)    26,028,386
synapses in that universe     125,365,933
self-loops                           112
===========================  =============

Rows whose pre- or post-synaptic partner is not an annotated body are dropped:
they are unlabelled fragments and cannot be assigned to a cell type.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

from .. import paths
from .annotations import Annotations

EDGES_NPZ = paths.DATA_PROCESSED / "edges.npz"
EDGES_QC = paths.DATA_PROCESSED / "edges_qc.json"

# Reference values from the feasibility study; a run whose counts differ is
# either using a different release or has a bug.
EXPECTED_TOTAL_ROWS = 151_856_684
#: rows whose pre- and post-synaptic partners are both annotated bodies
#: (measured **before** self-loop removal)
EXPECTED_ANNOTATED_EDGES = 26_028_386
#: self-loops (pre == post) among those rows
EXPECTED_SELF_LOOPS = 112
#: total synapses in the annotated-annotated rows (before self-loop removal)
EXPECTED_ANNOTATED_SYNAPSES = 125_365_933
#: derived, not guessed
EXPECTED_USABLE_EDGES = EXPECTED_ANNOTATED_EDGES - EXPECTED_SELF_LOOPS  # 26_028_274


def _reader(path: Path) -> ipc.RecordBatchFileReader:
    return ipc.open_file(pa.memory_map(str(path), "r"))


def stream_edges(path: Path, batch_rows: int = 1 << 20):
    """Yield ``(body_pre, body_post, weight)`` numpy chunks from the feather file."""
    reader = _reader(path)
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        yield (
            batch.column("body_pre").to_numpy(),
            batch.column("body_post").to_numpy(),
            batch.column("weight").to_numpy(),
        )


def build_edge_list(
    ann: Annotations,
    *,
    cache: bool = True,
    logger=None,
    strict: bool = True,
    path: Path | None = None,
) -> dict:
    """Stream the connectivity file and keep only annotated-annotated edges.

    Returns a dict with ``pre``, ``post`` (int32 positional indices into ``ann``)
    and ``weight`` (float32), plus a QC summary.
    """
    path = path or paths.raw_path("connectome_weights")
    if not path.exists():
        raise FileNotFoundError(f"connectome weights not found: {path}")

    # Body ids span a very wide range (max 1,571,825,087 for only ~211k rows),
    # so map them with a binary search rather than a 12.6 GB dense table.
    order = np.argsort(ann.body_ids, kind="stable")
    sorted_ids = ann.body_ids[order]
    sorted_pos = order.astype(np.int64)

    def to_index(ids: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(sorted_ids, ids)
        clipped = np.clip(pos, 0, len(sorted_ids) - 1)
        hit = sorted_ids[clipped] == ids
        return np.where(hit, sorted_pos[clipped], -1).astype(np.int64)

    pre_l, post_l, w_l = [], [], []
    total_rows = 0
    n_batches = 0
    n_annotated = 0
    synapses_annotated = 0
    self_loops = 0
    for bp, bq, bw in stream_edges(path):
        n_batches += 1
        total_rows += len(bp)
        p = to_index(bp)
        q = to_index(bq)
        keep = (p >= 0) & (q >= 0)
        if not keep.any():
            continue
        p, q, w = p[keep], q[keep], bw[keep]
        n_annotated += len(p)
        synapses_annotated += int(w.sum())
        sl = p == q
        self_loops += int(sl.sum())
        p, q, w = p[~sl], q[~sl], w[~sl]
        pre_l.append(p.astype(np.int32))
        post_l.append(q.astype(np.int32))
        w_l.append(w.astype(np.float32))
        if logger and n_batches % 400 == 0:
            logger.info(
                "  read %5d batches | %d rows | %d annotated-annotated | %d kept",
                n_batches,
                total_rows,
                n_annotated,
                sum(len(a) for a in pre_l),
            )

    pre = np.concatenate(pre_l)
    post = np.concatenate(post_l)
    weight = np.concatenate(w_l)
    del pre_l, post_l, w_l

    # duplicate check: connectome-weights is expected to be aggregated already
    keys = pre.astype(np.int64) * ann.n + post
    n_unique = len(np.unique(keys))
    del keys
    is_aggregated = n_unique == len(pre)

    qc = {
        "source_file": path.name,
        "source_size_bytes": path.stat().st_size,
        "total_rows": int(total_rows),
        "annotated_edges": int(n_annotated),
        "annotated_synapses": int(synapses_annotated),
        "usable_edges": int(len(pre)),
        "usable_synapses": int(weight.sum()),
        "self_loops_removed": int(self_loops),
        "self_loop_synapses_removed": int(synapses_annotated - weight.sum()),
        "aggregated_pre_post": bool(is_aggregated),
        "fraction_usable": round(len(pre) / total_rows, 6),
        "weight_min": float(weight.min()),
        "weight_max": float(weight.max()),
        "weight_mean": float(weight.mean()),
        "weight_median": float(np.median(weight)),
        "n_batches": n_batches,
    }
    if strict:
        if total_rows != EXPECTED_TOTAL_ROWS:
            raise AssertionError(
                f"row count {total_rows} != expected {EXPECTED_TOTAL_ROWS}"
            )
        if n_annotated != EXPECTED_ANNOTATED_EDGES:
            raise AssertionError(
                f"annotated-annotated edges {n_annotated} != {EXPECTED_ANNOTATED_EDGES}"
            )
        if synapses_annotated != EXPECTED_ANNOTATED_SYNAPSES:
            raise AssertionError(
                f"annotated synapses {synapses_annotated} != {EXPECTED_ANNOTATED_SYNAPSES}"
            )
        if self_loops != EXPECTED_SELF_LOOPS:
            raise AssertionError(f"self-loops {self_loops} != {EXPECTED_SELF_LOOPS}")
        if not is_aggregated:
            raise AssertionError("edge list contains duplicate (pre, post) pairs")

    if cache:
        np.savez(
            EDGES_NPZ,
            pre=pre,
            post=post,
            weight=weight,
            n_neurons=np.int64(ann.n),
            body_ids=ann.body_ids,
        )
        EDGES_QC.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    if logger:
        logger.info(
            "edges: %d usable / %d rows (%.2f%%)  |  %d synapses  |  aggregated=%s",
            len(pre),
            total_rows,
            qc["fraction_usable"] * 100,
            int(weight.sum()),
            is_aggregated,
        )
    return {"pre": pre, "post": post, "weight": weight, "qc": qc}


def load_edge_list(verify: bool = True) -> dict:
    """Load the cached edge list (building it must have happened already)."""
    if not EDGES_NPZ.exists():
        raise FileNotFoundError(
            f"{EDGES_NPZ} missing - run scripts/02_build_connectome.py first"
        )
    with np.load(EDGES_NPZ) as z:
        out = {
            "pre": z["pre"],
            "post": z["post"],
            "weight": z["weight"],
            "n_neurons": int(z["n_neurons"]),
            "body_ids": z["body_ids"],
        }
    if verify and len(out["pre"]) != EXPECTED_USABLE_EDGES:
        raise AssertionError(
            f"cached edge list has {len(out['pre'])} edges, expected {EXPECTED_USABLE_EDGES}"
        )
    return out


def degree_tables(pre: np.ndarray, post: np.ndarray, weight: np.ndarray, n: int) -> dict:
    """In/out degree and weighted strength statistics."""
    out_deg = np.bincount(pre, minlength=n)
    in_deg = np.bincount(post, minlength=n)
    out_w = np.bincount(pre, weights=weight.astype(np.float64), minlength=n)
    in_w = np.bincount(post, weights=weight.astype(np.float64), minlength=n)
    return {
        "in_degree": in_deg,
        "out_degree": out_deg,
        "in_weight": in_w,
        "out_weight": out_w,
    }
