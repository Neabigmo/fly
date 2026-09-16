"""Annotation-table invariants.

These need the downloaded MaleCNS files; they are skipped when the data is not
present so the rest of the suite still runs on a fresh checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from flynum import paths


def _require_annotations():
    if not paths.raw_path("annotations").exists():
        pytest.skip("MaleCNS annotations not downloaded")
    from flynum.data.annotations import load_annotations

    return load_annotations()


def test_body_ids_are_unique_and_hex_columns_match_measured_values() -> None:
    ann = _require_annotations()
    assert ann.n == 211_577
    assert len(np.unique(ann.body_ids)) == ann.n

    qc = ann.qc()
    # values established in the feasibility study and re-verified by
    # scripts/02_build_connectome.py
    assert qc["hex_neurons"] == 23_720
    assert qc["hex_columns"] == 892
    assert qc["hex_columns_R"] == 892
    assert qc["hex_columns_L"] == 879
    assert qc["lc11_neurons"] == 143
    assert len(qc["hex_types"]) == 15


def test_index_of_matches_a_dense_lookup_table() -> None:
    """Regression guard for the 12.6 GB dense table.

    MaleCNS body ids reach 1,571,825,087 while the table has only 211,577 rows,
    so a dense ``np.full(max_id)`` mapping allocated 12.6 GB and repeatedly
    exhausted memory.  The binary-search implementation must agree with an
    independent reference on every probe id.

    The reference is a dict, not the old dense array: rebuilding that array here
    would reintroduce the very allocation this test guards against, and it fails
    whenever other work is holding RAM.
    """
    ann = _require_annotations()
    # round trip over every row
    assert np.array_equal(ann.index_of(ann.body_ids), np.arange(ann.n))

    probe = np.array(
        [-1, 0, 1, int(ann.body_ids.max()), int(ann.body_ids.max()) + 1, 123456789]
        + ann.body_ids[:20].tolist()
    )
    lut = {int(b): i for i, b in enumerate(ann.body_ids)}
    expected = np.array([lut.get(int(i), -1) for i in probe], dtype=np.int64)
    assert np.array_equal(ann.index_of(probe), expected)


def test_hex_types_are_the_expected_columnar_cells() -> None:
    from flynum.data.annotations import HEX_TYPES

    ann = _require_annotations()
    observed = set(np.unique(ann.cell_type[ann.mask_hex]).tolist())
    assert observed == set(HEX_TYPES)


def test_input_type_sets_are_subsets_of_hex_types() -> None:
    from flynum.data.annotations import HEX_TYPES, INPUT_TYPE_SETS

    for name, types in INPUT_TYPE_SETS.items():
        assert set(types) <= set(HEX_TYPES), f"{name} contains unknown types"
