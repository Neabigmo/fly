"""Stage 0c -- build the neuron-level edge list.

Streams ``connectome-weights-...-minconf-0.5.feather`` (151,856,684 rows), keeps
only edges whose pre- and post-synaptic partners are both annotated bodies,
drops self-loops, and writes a compact ``data/processed/edges.npz``.

The expected counts are asserted so that a silent upstream change or a coding
error cannot propagate into the results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.data import connectome as cx  # noqa: E402
from flynum.data.annotations import load_annotations, write_qc  # noqa: E402
from flynum.data.manifest import Manifest  # noqa: E402
from flynum.logging_utils import RunContext, get_logger  # noqa: E402


def main() -> int:
    ctx = RunContext("stage0_build_connectome")
    log = ctx.log
    log.info("=" * 78)
    log.info("BUILD CONNECTOME")
    log.info("=" * 78)

    ann = load_annotations()
    qc_ann = ann.qc()
    write_qc(ann)
    log.info("annotations: %d bodies", ann.n)
    for key in ("hex_neurons", "hex_columns", "hex_columns_L", "hex_columns_R", "lc11_neurons"):
        log.info("  %-16s %s", key, qc_ann.get(key))
    log.info("  superclasses: %s", json.dumps(qc_ann["superclass_counts"], default=str))
    log.info("  hex types   : %s", json.dumps(qc_ann["hex_types"], default=str))

    log.info("-" * 78)
    res = cx.build_edge_list(ann, cache=True, logger=log, strict=True)
    qc = res["qc"]
    log.info("edge QC:")
    for k, v in qc.items():
        log.info("  %-24s %s", k, v)

    man = Manifest()
    man.add("edges", paths.processed_path("edges.npz"), **qc)
    man.add("annotations_qc", paths.processed_path("annotations_qc.json"), **qc_ann)
    man.save()

    ctx.finish(
        "ok",
        stage="0c",
        total_rows=qc["total_rows"],
        usable_edges=qc["usable_edges"],
        usable_synapses=qc["usable_synapses"],
        hex_neurons=qc_ann["hex_neurons"],
        hex_columns=qc_ann["hex_columns"],
    )
    log.info("BUILD CONNECTOME: DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
