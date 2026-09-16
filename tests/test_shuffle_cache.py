"""The shuffled control must actually be shuffled -- including on cache hits.

A previous version cached only the shuffle *statistics* and then returned the
real edge list, so every "shuffled" run trained on the real connectome and the
two conditions produced byte-identical metrics.  These tests exercise the
cache round-trip explicitly, because the failure mode is invisible in the
metrics: it looks exactly like a genuine null result.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from flynum.config import ExperimentConfig
from flynum.data.shuffle import edge_overlap
from flynum.data.subgraph import Subgraph


def _synthetic_subgraph(n: int = 400, e: int = 3000, seed: int = 0) -> Subgraph:
    """A duplicate-free, self-loop-free edge list, like the real connectome."""
    rng = np.random.default_rng(seed)
    keys = rng.choice(n * n, size=int(e * 1.4), replace=False)
    pre = (keys // n).astype(np.int32)
    post = (keys % n).astype(np.int32)
    keep = pre != post
    pre, post = pre[keep][:e], post[keep][:e]
    w = rng.integers(1, 25, len(pre)).astype(np.float32)
    return Subgraph(
        name="tiny",
        body_ids=np.arange(n, dtype=np.int64),
        pre=pre,
        post=post,
        weight=w,
        selection_mask=np.ones(n, dtype=bool),
        isolated_removed=0,
    )


@pytest.fixture()
def shuffle_env(tmp_path, monkeypatch):
    import flynum.pipeline as pipeline

    monkeypatch.setattr(pipeline, "CIRCUIT_DIR", tmp_path)
    return pipeline


def _cfg(seed: int, rounds: float = 6.0) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.data.graph = "shuffled"
    cfg.data.shuffle_rounds_per_edge = rounds
    cfg.seeds["shuffle_seed"] = seed
    return cfg


def test_shuffled_graph_differs_from_real_and_survives_caching(shuffle_env) -> None:
    sg = _synthetic_subgraph()
    cfg = _cfg(seed=3)

    first = shuffle_env.get_graph(sg, cfg)
    assert not np.array_equal(first.pre, sg.pre) or not np.array_equal(first.post, sg.post)

    overlap = edge_overlap(sg.pre, sg.post, first.pre, first.post, sg.n_neurons)
    assert overlap < 0.5, f"shuffle barely changed the graph (overlap {overlap:.2f})"

    # second call must hit the cache and return the *shuffled* edges, not the real ones
    second = shuffle_env.get_graph(sg, cfg)
    assert np.array_equal(first.pre, second.pre)
    assert np.array_equal(first.post, second.post)
    assert np.array_equal(first.weight, second.weight)
    assert edge_overlap(sg.pre, sg.post, second.pre, second.post, sg.n_neurons) < 0.5


def test_cached_shuffle_preserves_degrees_and_weights(shuffle_env) -> None:
    sg = _synthetic_subgraph()
    res = shuffle_env.get_graph(sg, _cfg(seed=7))
    n = sg.n_neurons
    assert np.array_equal(np.bincount(res.post, minlength=n), np.bincount(sg.post, minlength=n))
    assert np.array_equal(np.bincount(res.pre, minlength=n), np.bincount(sg.pre, minlength=n))
    assert np.array_equal(np.sort(res.weight), np.sort(sg.weight))
    assert not np.any(res.pre == res.post)


def test_statistics_only_cache_is_rebuilt(shuffle_env, tmp_path) -> None:
    """A cache written by the buggy version must be detected and regenerated."""
    sg = _synthetic_subgraph()
    cfg = _cfg(seed=5, rounds=4.0)
    path = (
        tmp_path
        / f"{sg.name}_shuffled_seed{cfg.seeds['shuffle_seed']}"
        f"_r{cfg.data.shuffle_rounds_per_edge:g}.npz"
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.savez(path, stats=json.dumps({"final_edge_overlap": 0.02, "edges": len(sg.pre)}))

    res = shuffle_env.get_graph(sg, cfg)
    overlap = edge_overlap(sg.pre, sg.post, res.pre, res.post, sg.n_neurons)
    assert overlap < 0.5, "statistics-only cache was trusted and returned the real graph"
    assert np.array_equal(np.sort(res.weight), np.sort(sg.weight))


def test_validation_rejects_an_unshuffled_graph(shuffle_env) -> None:
    """The guard must refuse a 'shuffled' graph that is really the real one."""
    sg = _synthetic_subgraph()
    with pytest.raises(AssertionError, match="shares"):
        shuffle_env._validate_shuffle(sg, sg.pre, sg.post, sg.weight, {}, None)


def test_two_shuffle_seeds_give_different_graphs(shuffle_env) -> None:
    sg = _synthetic_subgraph()
    a = shuffle_env.get_graph(sg, _cfg(seed=1))
    b = shuffle_env.get_graph(sg, _cfg(seed=2))
    assert not np.array_equal(a.post, b.post)
