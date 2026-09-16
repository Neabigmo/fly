"""Stage 0d -- build the circuits and validate the retinotopic map.

Builds the ``core`` (33.5k neurons / 1.17M edges) and ``full``
(106.9k neurons / 12.5M edges) circuits, constructs the fixed retinal encoder,
and writes a full QC bundle including a figure of the hexagonal visual field
with the stimulus footprint overlaid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.data.manifest import Manifest  # noqa: E402
from flynum.logging_utils import RunContext  # noqa: E402
from flynum.pipeline import prepare  # noqa: E402


def main() -> int:
    ctx = RunContext("stage0_circuits")
    log = ctx.log
    log.info("=" * 78)
    log.info("BUILD CIRCUITS + RETINAL MAP")
    log.info("=" * 78)

    results = {}
    for circuit in ("core", "full"):
        cfg = ExperimentConfig()
        cfg.data.circuit = circuit
        prep = prepare(cfg, logger=log, strict_circuit=True)
        results[circuit] = prep.info

        # ---- retinal map sanity ------------------------------------- #
        qc = prep.field.qc()
        assert abs(qc["nn_distance_mean"] - 1.0) < 1e-4, (
            f"hex lattice broke: NN distance {qc['nn_distance_mean']}"
        )
        assert 1.1 < qc["density"] < 1.3, f"field is not a filled hex region: {qc['density']}"
        assert prep.encoder.qc.n_driven_columns > 100, "too few image-driven columns"

        # round-trip: an image must produce a non-trivial column pattern
        img = np.zeros((32, 32), dtype=np.float32)
        img[14:18, 14:18] = 1.0
        cols = prep.encoder.sample(img)
        assert cols.max() > 0.5, "bright stimulus produced no column activation"
        log.info(
            "  map check: input-columns=%d  image-driven=%d/%d (%.1f%%) | "
            "max activation=%.3f | input neurons=%d",
            prep.encoder.qc.n_columns_with_input,
            prep.encoder.qc.n_driven_columns,
            prep.encoder.qc.n_columns,
            100 * prep.encoder.qc.driven_fraction,
            float(cols.max()),
            prep.encoder.qc.n_input_neurons,
        )

        sub_ann = prep.ann.index_of(prep.subgraph.body_ids)
        type_counts = {}
        for t in np.unique(prep.ann.cell_type[sub_ann]):
            type_counts[str(t)] = int((prep.ann.cell_type[sub_ann] == t).sum())
        results[circuit]["top_cell_types"] = dict(
            sorted(type_counts.items(), key=lambda kv: -kv[1])[:25]
        )
        log.info("-" * 78)

    out = paths.DATA_PROCESSED / "circuit_report.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    man = Manifest()
    for circuit in ("core", "full"):
        man.add(
            f"circuit_{circuit}",
            paths.DATA_PROCESSED / "circuits" / f"{circuit}.npz",
            **{k: v for k, v in results[circuit].items() if not isinstance(v, dict)},
        )
    man.save()

    ctx.finish("ok", circuits=list(results))
    log.info("wrote %s", out)
    log.info("BUILD CIRCUITS: DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
