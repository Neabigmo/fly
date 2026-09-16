"""Shuffled-control invariants and the hexagonal retinotopic map."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull, cKDTree  # noqa: F401  (ConvexHull kept for parity)

from flynum.data.shuffle import edge_overlap, shuffle_edges
from flynum.retina.hexmap import build_hex_field, hex_to_cartesian


# --------------------------------------------------------------------------- #
# shuffled-fly control
# --------------------------------------------------------------------------- #
def _random_edges(n: int, e: int, seed: int = 0):
    """A duplicate-free, self-loop-free edge list (the MaleCNS format)."""
    rng = np.random.default_rng(seed)
    keys = rng.choice(n * n, size=int(e * 1.4), replace=False)
    pre = (keys // n).astype(np.int32)
    post = (keys % n).astype(np.int32)
    keep = pre != post
    pre, post = pre[keep][:e], post[keep][:e]
    w = rng.integers(1, 20, len(pre)).astype(np.float32)
    return pre, post, w


def test_shuffle_rejects_non_aggregated_edge_list() -> None:
    """A duplicate (pre, post) pair would make degree preservation ill-defined."""
    pre = np.array([0, 1, 0], dtype=np.int32)
    post = np.array([1, 2, 1], dtype=np.int32)   # (0,1) appears twice
    w = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError, match="aggregated"):
        shuffle_edges(pre, post, w, 4, rounds_per_edge=1.0, seed=0)


def test_shuffle_preserves_degrees_and_weight_multiset() -> None:
    n = 300
    pre, post, w = _random_edges(n, 4000, seed=1)
    res = shuffle_edges(pre, post, w, n, rounds_per_edge=20.0, seed=42)

    assert np.array_equal(np.bincount(post, minlength=n), np.bincount(res.post, minlength=n))
    assert np.array_equal(np.bincount(pre, minlength=n), np.bincount(res.pre, minlength=n))
    assert np.array_equal(np.sort(w), np.sort(res.weight))


def test_shuffle_removes_self_loops_and_duplicates() -> None:
    n = 300
    pre, post, w = _random_edges(n, 4000, seed=2)
    res = shuffle_edges(pre, post, w, n, rounds_per_edge=20.0, seed=7)
    assert not np.any(res.pre == res.post)
    keys = res.pre.astype(np.int64) * n + res.post
    assert len(np.unique(keys)) == len(keys)


def test_shuffle_actually_randomises() -> None:
    n = 300
    pre, post, w = _random_edges(n, 4000, seed=3)
    res = shuffle_edges(pre, post, w, n, rounds_per_edge=20.0, seed=11)
    ov = edge_overlap(pre, post, res.pre, res.post, n)
    assert ov < 0.2, f"edge overlap {ov:.3f} is too high - swap did not mix"
    assert res.stats["swaps"] > 0


def test_shuffle_is_deterministic_under_seed() -> None:
    n = 200
    pre, post, w = _random_edges(n, 2000, seed=4)
    a = shuffle_edges(pre, post, w, n, rounds_per_edge=5.0, seed=99)
    b = shuffle_edges(pre, post, w, n, rounds_per_edge=5.0, seed=99)
    assert np.array_equal(a.pre, b.pre) and np.array_equal(a.post, b.post)
    c = shuffle_edges(pre, post, w, n, rounds_per_edge=5.0, seed=100)
    assert not np.array_equal(a.post, c.post)


# --------------------------------------------------------------------------- #
# hex lattice
# --------------------------------------------------------------------------- #
def test_hex_to_cartesian_is_the_hexagonal_lattice() -> None:
    """Neighbouring axial cells must be exactly 1 unit apart (hex, not square)."""
    q, r = np.meshgrid(np.arange(-12, 13), np.arange(-12, 13))
    q, r = q.ravel(), r.ravel()
    x, y = hex_to_cartesian(q.astype(float), r.astype(float))
    pts = np.stack([x, y], axis=1)
    d, _ = cKDTree(pts).query(pts, k=7)
    assert np.allclose(d[:, 1], 1.0, atol=1e-9), "nearest-neighbour spacing must be 1"
    # exactly six neighbours at distance 1 for an interior cell
    n_at_one = (d[:, 1:] < 1.0 + 1e-9).sum(axis=1)
    interior = (q > -10) & (q < 10) & (r > -10) & (r < 10) & (np.abs(q + r) < 10)
    assert np.all(n_at_one[interior] == 6)

    # A square lattice would put only 4 neighbours at distance 1: this is the
    # discriminating test between "hexagonal axial" and "row/column integers".
    square_n = np.full(len(q), 4)
    assert not np.array_equal(n_at_one[interior], square_n[interior])

    # Local density at radius 8 sits a few percent above the asymptotic 1.1547
    # because a finite sampling disc always over-counts its own boundary.  The
    # test is therefore a coarse sanity check; the exact characterisation of the
    # lattice is the "exactly six neighbours at distance 1" assertion above,
    # which a square lattice fails.
    centre = pts.mean(0)
    within = int((((pts - centre) ** 2).sum(1) <= 64.0).sum())
    local_density = within / (np.pi * 64.0)
    assert abs(local_density - 2 / np.sqrt(3)) < 0.06, (
        f"local density {local_density:.4f} differs from hexagonal 1.1547"
    )
    assert local_density > 1.1, "a square lattice would give 1.0"

    # exact: the six neighbours of an interior point are spaced 60 degrees apart
    qc, rc = 0, 0
    dist = np.linalg.norm(pts - np.array([qc + rc / 2, rc * np.sqrt(3) / 2]), axis=1)
    ring = np.argsort(dist)[1:7]
    angles = np.sort(np.arctan2(pts[ring, 1], pts[ring, 0]))
    gaps = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))
    assert np.allclose(gaps, np.pi / 3, atol=1e-6), "interior angles are not 60 degrees"


def test_build_hex_field_separates_the_two_eyes() -> None:
    rng = np.random.default_rng(0)
    q = rng.integers(1, 12, 80).astype(float)
    r = rng.integers(1, 12, 80).astype(float)
    side = np.where(np.arange(80) % 2 == 0, "L", "R").astype(object)
    field = build_hex_field(q, r, side)
    expected = len(np.unique(np.stack([q[side == "L"], r[side == "L"]], 1), axis=0)) + len(
        np.unique(np.stack([q[side == "R"], r[side == "R"]], 1), axis=0)
    )
    assert field.n_cols == expected
    assert field.qc()["per_side"]["L"] > 0 and field.qc()["per_side"]["R"] > 0


def test_hex_field_qc_survives_degenerate_input() -> None:
    """A one-column field must not crash the convex-hull density estimate."""
    h1 = np.array([1.0, 2.0])
    h2 = np.array([1.0, 1.0])
    side = np.array(["L", "L"], dtype=object)
    qc = build_hex_field(h1, h2, side).qc()
    assert qc["n_columns"] == 2
    assert np.isnan(qc["hull_area"])


def test_column_lookup_round_trip() -> None:
    from flynum.retina.hexmap import column_lookup

    rng = np.random.default_rng(0)
    q = rng.integers(1, 20, 200).astype(float)
    r = rng.integers(1, 20, 200).astype(float)
    side = rng.choice(["L", "R"], 200).astype(object)
    field = build_hex_field(q, r, side)
    idx = column_lookup(field, q, r, side)
    assert np.all(idx >= 0)
    back = field.columns[idx]
    assert np.allclose(back[:, 0], q)
    assert np.allclose(back[:, 1], r)
