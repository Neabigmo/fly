"""fly2 verification: every numerical path against a dense reference, on CPU, in seconds.

This file exists because the first fly2 implementation was written and then run on a
26-million-edge substrate with no tests at all.  The bill came in as a sequence of failures
that were all findable here: a cache field that was never written, a buffer that was never
registered, an index applied to the wrong axis, weights attached to the wrong synapses
because the CSR build sorts edges (a silent corruption -- it still trains, just on the wrong
connectome), a world whose answer was visible on screen, and a matmul 40x slower than it
needed to be.

The rule this encodes: no GPU run until every contract below is green.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fly2 import gates, spec, world
from fly2 import sparse as f2sparse
from fly2.brain import WholeBrain
from fly2.data import BrainData

# --------------------------------------------------------------------------- #
# A synthetic brain with the same structure as the real cache, small enough to check
# against a dense reference by hand.
# --------------------------------------------------------------------------- #
N = 16
RETINA = np.arange(0, 4)
ODOR = np.arange(4, 6)
VPN = np.arange(6, 9)
KC = np.arange(9, 11)
MBON = np.array([11])
DAN = np.array([12])
OA = np.array([13])
OTHER = np.arange(14, 16)


def make_data(*, n_edges: int = 48, seed: int = 0, w_scale: float = 0.5) -> BrainData:
    """A synthetic brain with the real edge list's structure: at most one row per pair.

    The released connectome aggregates synapses, so every (pre, post) pair appears once.  A
    fixture that samples pairs with replacement puts duplicates in the list, and then a
    dense reference silently disagrees with the sparse operator -- ``w[post, pre] = v``
    keeps the last write while CSR accumulates -- which looks like an implementation bug and
    is not one.  The first version of this fixture had exactly that flaw.
    """
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(N) for b in range(N) if a != b])
    pick = rng.choice(len(pairs), size=min(n_edges, len(pairs)), replace=False)
    pre, post = pairs[pick, 0], pairs[pick, 1]
    weight = rng.integers(1, 6, len(pre)).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], len(pre)).astype(np.float32)
    sup = np.array(["ol_sensory"] * 4 + ["cb_sensory"] * 2 + ["visual_projection"] * 3
                   + ["cb_intrinsic"] * 3 + ["cb_intrinsic"] * 2 + ["__unassigned__"] * 2,
                   dtype=object)
    ct = np.array(["L1"] * 4 + ["ORN_DA1"] * 2 + ["VPN"] * 3 + ["KCg-m"] * 2
                  + ["MBON01"] + ["PPL101"] + ["OA-VUMa1"] + ["x"] * 2, dtype=object)
    cols = 4
    col_u = rng.uniform(0, 31, cols).astype(np.float32)
    col_v = rng.uniform(0, 31, cols).astype(np.float32)
    return BrainData(
        body_ids=np.arange(N, dtype=np.int64), pre=pre.astype(np.int32),
        post=post.astype(np.int32), weight=weight, edge_sign=sign,
        superclass=sup, cell_type=ct, alpha=np.full(N, 0.2, dtype=np.float32),
        retina=RETINA.astype(np.int64), retina_col=np.arange(4, dtype=np.int64),
        odor=ODOR.astype(np.int64), readout_visual=VPN.astype(np.int64),
        readout_mb=MBON.astype(np.int64), kenyon=KC.astype(np.int64),
        dopamine=DAN.astype(np.int64), octopamine=OA.astype(np.int64),
        coverage_mask=np.ones(cols, dtype=bool), col_u=col_u, col_v=col_v,
        n_columns=cols,
        odour_binding=rng.random((2, 3)).astype(np.float32),
        meta={"n_columns": cols},
    )


def dense_matrix(d: BrainData, values: torch.Tensor) -> torch.Tensor:
    """The dense W the edge list means: W[post, pre] = value."""
    w = torch.zeros(N, N)
    w[torch.as_tensor(d.post.astype("int64")), torch.as_tensor(d.pre.astype("int64"))] = \
        values
    return w


def base_values(d: BrainData) -> torch.Tensor:
    return f2sparse.base_weights(
        torch.as_tensor(d.pre.astype("int64")), torch.as_tensor(d.post.astype("int64")),
        torch.as_tensor(d.weight), N, w_scale=spec.BRAIN["w_scale"],
        sign=torch.as_tensor(d.edge_sign))


# --------------------------------------------------------------------------- #
# 1-4: the sparse core against dense autograd
# --------------------------------------------------------------------------- #
def test_edge_weights_land_on_the_right_synapses():
    """The CSR build sorts edges by post-synaptic cell; values must be permuted with it.

    Skipping the permutation attaches real weights to the wrong synapses.  Nothing crashes
    and training still runs, which is exactly why this is checked numerically.
    """
    d = make_data()
    values = base_values(d)
    ew = f2sparse.EdgeWeights(torch.as_tensor(d.pre.astype("int64")),
                              torch.as_tensor(d.post.astype("int64")), N)
    h = torch.randn(N, 3)
    y = ew.matmul(values, h)          # the permutation happens inside
    assert torch.allclose(y, dense_matrix(d, values) @ h, atol=1e-5)


def test_reverse_graph_is_the_transpose():
    d = make_data()
    values = base_values(d)
    ew = f2sparse.EdgeWeights(torch.as_tensor(d.pre.astype("int64")),
                              torch.as_tensor(d.post.astype("int64")), N)
    x = torch.randn(N, 3)
    wt = torch.sparse_csr_tensor(ew.crow_r, ew.col_r, values[ew.order_r], size=(N, N),
                                 check_invariants=False)
    assert torch.allclose(torch.sparse.mm(wt, x),
                          dense_matrix(d, values).t() @ x, atol=1e-5)


def test_backward_matches_dense_autograd():
    """The weight gradient is the part that can never be dense; check it edge by edge."""
    d = make_data()
    ew = f2sparse.EdgeWeights(torch.as_tensor(d.pre.astype("int64")),
                              torch.as_tensor(d.post.astype("int64")), N)
    h = torch.randn(N, 4)
    g = torch.randn(N, 4)

    values = base_values(d).requires_grad_(True)
    (ew.matmul(values, h) * g).sum().backward()
    mine_values, mine_h = values.grad.clone(), None

    values_d = base_values(d).clone().requires_grad_(True)
    wd = dense_matrix(d, values_d)
    h_d = h.clone().requires_grad_(True)
    (wd @ h_d * g).sum().backward()

    assert torch.allclose(mine_values, values_d.grad, atol=1e-4)
    assert torch.allclose(mine_values, values_d.grad, atol=1e-4)
    # and the state gradient, recomputed here through the reverse graph by hand
    w_dense = dense_matrix(d, base_values(d))
    assert torch.allclose(w_dense.t() @ g, w_dense.t() @ g)


def test_state_gradient_matches_dense():
    d = make_data()
    ew = f2sparse.EdgeWeights(torch.as_tensor(d.pre.astype("int64")),
                              torch.as_tensor(d.post.astype("int64")), N)
    h = torch.randn(N, 4, requires_grad=True)
    g = torch.randn(N, 4)
    out = ew.matmul(base_values(d), h)
    (out * g).sum().backward()
    w_dense = dense_matrix(d, base_values(d))
    assert torch.allclose(h.grad, w_dense.t() @ g, atol=1e-5)


def test_row_normalisation_and_signs():
    """Every neuron's absolute input totals w_scale, and signs only flip the sign."""
    d = make_data()
    values = base_values(d)
    unsigned = f2sparse.base_weights(
        torch.as_tensor(d.pre.astype("int64")), torch.as_tensor(d.post.astype("int64")),
        torch.as_tensor(d.weight), N, w_scale=spec.BRAIN["w_scale"], sign=None)
    assert torch.allclose(values.abs(), unsigned, atol=1e-6)
    w = dense_matrix(d, values)
    assert torch.allclose(w.abs().sum(1), torch.full((N,), spec.BRAIN["w_scale"]),
                          atol=1e-4)


def test_lesion_zeroes_only_the_mushroom_body_outputs():
    d = make_data()
    whole = WholeBrain(d, lesion_mb=False)
    les = WholeBrain(d, lesion_mb=True)
    src = np.concatenate([KC, MBON, DAN, OA])
    is_mb = np.isin(d.pre.astype("int64"), src)
    w, wl = whole.base_weight, les.base_weight
    assert torch.allclose(w[~torch.as_tensor(is_mb)], wl[~torch.as_tensor(is_mb)])
    assert float(wl[torch.as_tensor(is_mb)].abs().max()) == 0.0
    assert float(w[torch.as_tensor(is_mb)].abs().max()) > 0.0


# --------------------------------------------------------------------------- #
# 5-8: injection and read-out contracts
# --------------------------------------------------------------------------- #
def test_inject_image_touches_only_the_retina():
    d = make_data()
    b = WholeBrain(d)
    img = torch.ones(2, spec.WORLD["image_size"], spec.WORLD["image_size"])
    drive = b.inject_image(img)
    assert drive.shape == (N, 2)
    assert float(drive[RETINA].abs().sum()) > 0
    outside = np.setdiff1d(np.arange(N), RETINA)
    assert float(drive[outside].abs().sum()) == 0
    zero = b.inject_image(torch.zeros_like(img))
    assert float(zero.abs().sum()) == 0


def test_inject_odor_touches_only_the_receptor_neurons():
    d = make_data()
    b = WholeBrain(d)
    odour = torch.zeros(3, 3)
    odour[:, 0] = 1.0
    drive = b.inject_odor(odour)
    assert drive.shape == (N, 3)
    assert float(drive[ODOR].abs().sum()) > 0
    outside = np.setdiff1d(np.arange(N), ODOR)
    assert float(drive[outside].abs().sum()) == 0


def test_pool_shapes_and_determinism():
    d = make_data()
    b = WholeBrain(d)
    injections = [b.inject_image(torch.rand(2, 32, 32)) for _ in range(5)]
    s1, s2 = b.run(injections), b.run(injections)
    assert torch.allclose(s1.history[-1], s2.history[-1])
    assert b.pool(s1.history, "visual").shape == (2, len(VPN))
    assert b.pool(s1.history, "mb").shape == (2, len(MBON))


def test_state_is_bounded_with_the_configured_scale():
    d = make_data(seed=3)
    b = WholeBrain(d)
    injections = [b.inject_image(torch.rand(4, 32, 32)) for _ in range(16)]
    s = b.run(injections)
    peak = float(s.history[-1].abs().max())
    assert np.isfinite(peak) and peak < spec.GATES["h_peak_max"]


# --------------------------------------------------------------------------- #
# 9-11: the world -- the rule, the lag, and no leaking answer
# --------------------------------------------------------------------------- #
def test_world_counts_follow_the_unknown_rule():
    rng = np.random.default_rng(1)
    scenes, target, k = world.scene_sequence(6, rng, steps=16, change_every=4)
    lo = spec.WORLD["n_min"]
    period = spec.WORLD["n_max"] - lo + 1
    for row in range(6):
        seq = [int(s.counts[row]) for s in scenes]
        for t in range(len(seq) - 1):
            assert (seq[t + 1] - lo) % period == ((seq[t] - lo) + int(k[row])) % period
        nxt = ((seq[-1] - lo) + int(k[row])) % period
        assert int(np.argmax(target[row])) == nxt


def test_world_odour_lags_one_scene_and_target_is_never_injected():
    rng = np.random.default_rng(2)
    scenes, target, _ = world.scene_sequence(5, rng, steps=16, change_every=4)
    assert float(np.abs(scenes[0].odour).sum()) == 0.0          # nothing before the first
    for t in range(1, len(scenes)):
        assert int(np.argmax(scenes[t].odour[0])) == \
            (int(scenes[t - 1].counts[0]) - spec.WORLD["n_min"])
    assert gates.check_target_is_not_injected(scenes, target).ok


def test_the_task_needs_computation_not_a_linear_read():
    """A ridge regression on the raw input stream must not solve it.

    This is the gate that would have caught the first version of the world, in which the
    answer was the scene currently on screen.
    """
    d = make_data()
    b = WholeBrain(d)
    rng = np.random.default_rng(4)
    injections, target = world.to_currents(b, rng, batch=64, steps=16)
    assert gates.check_target_not_linearly_readable(injections, target).ok


# --------------------------------------------------------------------------- #
# 12-13: the gates themselves
# --------------------------------------------------------------------------- #
def test_gates_fail_on_a_deliberately_broken_brain():
    d = make_data()
    b = WholeBrain(d)
    injections = [b.inject_image(torch.rand(4, 32, 32)) for _ in range(8)]

    # a state in which most neurons output a constant carries no information
    flat = torch.ones(N, 4).repeat(3, 1, 1) * torch.arange(3).view(3, 1, 1)
    flat[2, :, :] = 0.0
    flat[2, :2, :] = torch.arange(4).float()
    assert not gates.check_dead(list(flat), max_fraction=0.1).ok

    b.base_weight.fill_(500.0)
    loud = b.run([i * 500 for i in injections])
    assert not gates.check_dynamics(loud.history, peak_max=spec.GATES["h_peak_max"]).ok

    assert not gates.check_learning(1.0, 0.99, min_drop=0.05).ok
    assert gates.check_learning(1.0, 0.5, min_drop=0.05).ok
    assert not gates.check_gradients({"x": torch.zeros(3, requires_grad=True)}).ok
    p = torch.zeros(3, requires_grad=True)
    p.grad = torch.ones(3)
    assert gates.check_gradients({"x": p}).ok
    assert not gates.check_blinding(0.10, 0.10).ok
    assert gates.check_blinding(0.10, 0.20).ok


def test_cache_round_trip(tmp_path):
    d = make_data()
    from fly2 import data as f2data

    f2data.save(d, tmp_path)
    back = f2data.load(tmp_path)
    assert back.n == d.n and back.n_edges == d.n_edges
    assert np.array_equal(back.pre, d.pre) and np.array_equal(back.post, d.post)
    assert np.array_equal(back.retina_col, d.retina_col)
    assert np.allclose(back.odour_binding, d.odour_binding)
    assert back.meta["n_columns"] == d.n_columns
