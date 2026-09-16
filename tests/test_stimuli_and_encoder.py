"""Stimulus controls: the anti-shortcut guarantees, and the retinal map."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from flynum.config import StimulusConfig
from flynum.retina.encoder import RetinaEncoder, select_input_neurons
from flynum.retina.hexmap import build_hex_field, column_lookup
from flynum.stimuli.dots import (
    make_add_dataset,
    make_count_dataset,
    make_count_stimulus,
    render_dots,
)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_rendered_ink_matches_geometric_area() -> None:
    """Total intensity of an anti-aliased disc equals its geometric area."""
    r = 3.0
    img = render_dots(32, np.array([[15.5, 15.5]]), np.array([r]), supersample=4)
    assert abs(img.sum() - np.pi * r * r) / (np.pi * r * r) < 0.03
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_dots_do_not_overlap() -> None:
    cfg = StimulusConfig()
    rng = np.random.default_rng(0)
    for n in (2, 3, 4, 5):
        for _ in range(25):
            s = make_count_stimulus(n, cfg, rng, mode="A")
            # count connected components of the thresholded image
            from scipy import ndimage

            lab, k = ndimage.label(s.image > 0.35)
            assert k == n, f"expected {n} blobs, found {k}"


def test_area_controlled_condition_has_constant_ink() -> None:
    cfg = StimulusConfig()
    ds = make_count_dataset(600, cfg, seed=1, mode="B")
    ink_by_n = {n: ds["ink"][ds["ns"] == n] for n in range(cfg.n_min, cfg.n_max + 1)}
    for n, vals in ink_by_n.items():
        assert vals.std() / vals.mean() < 0.05, f"ink varies too much for n={n}"
    means = np.array([v.mean() for v in ink_by_n.values()])
    assert means.std() / means.mean() < 0.02, "Test B fails to equalise total area"


def test_natural_condition_leaks_ink_as_expected() -> None:
    """Sanity check: condition A *should* differ in ink (that is why B exists)."""
    cfg = StimulusConfig()
    ds = make_count_dataset(500, cfg, seed=2, mode="A")
    means = np.array([ds["ink"][ds["ns"] == n].mean() for n in range(cfg.n_min, cfg.n_max + 1)])
    assert np.all(np.diff(means) > 0), "natural condition should increase with N"


def test_envelope_controlled_condition_matches_spread() -> None:
    cfg = StimulusConfig()
    ds = make_count_dataset(500, cfg, seed=3, mode="C")
    ns = [n for n in range(cfg.n_min, cfg.n_max + 1) if n >= 2]
    spreads = np.array([ds["spread"][ds["ns"] == n].mean() for n in ns])
    assert spreads.std() / spreads.mean() < 0.10, "Test C fails to equalise spread"


def test_unseen_layout_mode_is_regular() -> None:
    """Condition D must be grid-like: pairwise distances take a few discrete values."""
    from scipy import ndimage

    cfg = StimulusConfig()
    s = make_count_stimulus(4, cfg, np.random.default_rng(0), mode="D")
    lab, k = ndimage.label(s.image > 0.35)
    assert k == 4
    centres = np.array(ndimage.center_of_mass(s.image > 0.35, lab, range(1, k + 1)))
    d = np.linalg.norm(centres[:, None] - centres[None, :], axis=2)[np.triu_indices(4, 1)]
    ratios = d / d.min()
    # a 3x3 square-grid subset can only produce these distance ratios
    allowed = np.array([1.0, np.sqrt(2.0), 2.0, np.sqrt(5.0), 2 * np.sqrt(2.0)])
    residual = np.min(np.abs(ratios[:, None] - allowed[None, :]), axis=1)
    assert residual.max() < 0.15, f"layout is not regular: ratio residuals {residual}"


def test_condition_D_does_not_leak_spread() -> None:
    """Randomising the grid orientation stops spacing from predicting the count."""
    from flynum.stimuli.dots import shortcut_report

    cfg = StimulusConfig()
    rep = shortcut_report(make_count_dataset(800, cfg, seed=5, mode="D"))
    assert rep["spread_centroid_accuracy"] < 0.6, (
        f"condition D leaks the count through dot spacing "
        f"({rep['spread_centroid_accuracy']:.2f})"
    )


def test_generated_datasets_respect_the_blob_count() -> None:
    """Every image must really contain N disconnected dots."""
    cfg = StimulusConfig()
    for mode in ("A", "B", "C", "D"):
        ds = make_count_dataset(120, cfg, seed=7, mode=mode)
        assert ds["blob_ok"].all(), f"mode {mode}: {int((~ds['blob_ok']).sum())} mislabelled"


def test_shortcut_baselines_are_at_chance_on_controlled_conditions() -> None:
    """If total ink predicts N on Test B, the control has failed."""
    from flynum.stimuli.dots import shortcut_report

    cfg = StimulusConfig()
    for mode in ("B", "C"):
        ds = make_count_dataset(1000, cfg, seed=4, mode=mode)
        rep = shortcut_report(ds)
        assert rep["ink_centroid_accuracy"] < 0.45, (
            f"mode {mode}: total ink still predicts the count "
            f"({rep['ink_centroid_accuracy']:.2f}) - control failed"
        )


# --------------------------------------------------------------------------- #
# addition task
# --------------------------------------------------------------------------- #
def test_addition_holdout_pairs_are_excluded_from_training() -> None:
    cfg = StimulusConfig()
    tr = make_add_dataset(cfg, seed=5, reps_per_pair=4, split="train")
    te = make_add_dataset(cfg, seed=5, reps_per_pair=4, split="test")
    tr_pairs = set(zip(tr["a"].tolist(), tr["b"].tolist()))
    te_pairs = set(zip(te["a"].tolist(), te["b"].tolist()))
    assert te_pairs == set(cfg.holdout_pairs)
    assert not (tr_pairs & te_pairs)
    # every sum must still be reachable from the training pairs alone
    tr_sums = set((tr["a"] + tr["b"]).tolist())
    te_sums = set((te["a"] + te["b"]).tolist())
    assert te_sums <= tr_sums, "held-out pairs must not create unseen sums"


# --------------------------------------------------------------------------- #
# retinal encoder
# --------------------------------------------------------------------------- #
def _tiny_encoder(map_mode="stretch"):
    q, r = np.meshgrid(np.arange(1, 12), np.arange(1, 12))
    q, r = q.ravel().astype(float), r.ravel().astype(float)
    side = np.array(["R"] * len(q), dtype=object)
    field = build_hex_field(q, r, side)
    col = column_lookup(field, q, r, side)
    types = np.array(["L1"] * len(q), dtype=object)
    mask = select_input_neurons(types, side, col, input_types="lamina", input_eye="both")
    return RetinaEncoder(field, col, mask, image_size=32, map_mode=map_mode), col


def test_encoder_drives_expected_columns() -> None:
    enc, col = _tiny_encoder("stretch")
    assert enc.qc.n_columns == len(np.unique(col))
    assert enc.qc.n_input_neurons > 0
    assert 0 < enc.qc.driven_fraction <= 1.0


def test_numpy_and_torch_encoders_agree() -> None:
    torch = pytest.importorskip("torch")
    from flynum.retina.torch_encoder import TorchRetina

    enc, _ = _tiny_encoder("stretch")
    rng = np.random.default_rng(0)
    imgs = rng.random((5, 32, 32)).astype(np.float32)

    np_out = np.stack([enc.sample(im) for im in imgs])
    tr = TorchRetina(enc, device="cpu")
    t_out = tr.encode(torch.as_tensor(imgs)).numpy()
    assert np.allclose(np_out, t_out, atol=1e-5)


def test_isotropic_map_preserves_lattice_spacing() -> None:
    """In ``isotropic`` mode one lattice step must equal exactly one pixel."""
    enc, _ = _tiny_encoder("isotropic")
    uv = enc.col_uv.astype(np.float64)
    d, _ = cKDTree(uv).query(uv, k=2)
    nn = d[:, 1]
    assert np.allclose(nn, 1.0, atol=1e-5), (
        f"lattice spacing is not preserved: min={nn.min():.4f} max={nn.max():.4f}"
    )


def test_stretch_mode_covers_every_column() -> None:
    enc, _ = _tiny_encoder("stretch")
    assert enc.coverage_mask().all(), "stretch mode should drive every column"
    x0, x1, y0, y1 = enc.field.bounds
    assert enc.col_uv[:, 0].max() > 20 and enc.col_uv[:, 1].max() > 20


def test_image_orientation_matches_the_visual_field() -> None:
    """Image top-left must drive the *top-left* of the visual field.

    This is a stated design contract ("the top-left of the picture stimulates the
    top-left of the fly's visual field").  An earlier version flipped the vertical
    axis; that is harmless for counting but breaks the contract and makes the
    encoder figure misleading.

    The contract is checked on the mapping itself rather than with image patches,
    because the field is a tilted diamond whose bounding-box corners are
    unoccupied -- a corner patch would legitimately drive nothing.
    """
    enc, _ = _tiny_encoder("stretch")
    xy = enc.field.xy.astype(np.float64)
    uv = enc.col_uv.astype(np.float64)

    rho_x = float(np.corrcoef(xy[:, 0], uv[:, 0])[0, 1])
    rho_y = float(np.corrcoef(xy[:, 1], uv[:, 1])[0, 1])
    assert rho_x > 0.99, f"field +x must map to image +u (correlation {rho_x:.3f})"
    assert rho_y < -0.99, (
        f"field +y must map to image -v, because image row 0 is the top "
        f"(correlation {rho_y:.3f})"
    )

    # behavioural spot check on a region the diamond actually covers
    field = enc.field
    cx, cy = field.centroid
    S = enc.image_size
    img = np.zeros((S, S), dtype=np.float32)
    img[S - 10 : S, 0:10] = 1.0        # bottom-left of the image
    vals = enc.sample(img)
    assert vals.max() > 0.9, "bottom-left patch produced no column activation"
    best = np.flatnonzero(vals > 0.9 * vals.max())
    dx = float(np.mean(field.xy[best, 0]) - cx)
    dy = float(np.mean(field.xy[best, 1]) - cy)
    assert dx < 0, f"bottom-left patch landed on the right of the field (dx={dx:.2f})"
    assert dy < 0, f"bottom-left patch landed at the top of the field (dy={dy:.2f})"
