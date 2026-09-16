"""Subgraph extraction -- which neurons make up the model.

Two named circuits, both defined by *rules over the annotation table* so they
are exactly reproducible.  Counts were measured in the feasibility study and are
asserted at build time.

``core``  (33,484 neurons / 1,170,024 edges)
    The columnar optic-lobe core that receives the image, plus the optic lobe's
    output neurons and the centrifugal feedback cells::

        assignedOlHex1.notna()  |  superclass == visual_projection
                                |  superclass == visual_centrifugal

    Small and fast (about 13 ms per training step at batch 64).  It retains
    26.4% of the outgoing synaptic weight of its own neurons, so it is a
    *lesioned* preparation: it tests whether connectome topology is a useful
    inductive bias, not whether the full fly circuit computes numerosity.

``full``  (99,170 neurons / 12,392,758 edges)
    A circuit-complete preparation that retains 91.3% of the outgoing synaptic
    weight of its own neurons::

        ol_intrinsic | hop1 | visual_projection | visual_centrifugal

    where ``hop1`` is every annotated neuron of superclass
    {ol_intrinsic, visual_projection, cb_intrinsic, visual_centrifugal}
    receiving at least one synapse from a hexagonal-column neuron.

    Restricting ``hop1`` by superclass keeps out 4,353 photoreceptors (only
    partially traced) and 3,357 bodies with no superclass annotation; the
    unrestricted variant would add those and reach 106,901 neurons / 91.7%
    retention for no scientific gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .annotations import Annotations

# Measured on MaleCNS v1.0 after self-loop removal, in the feasibility study and
# re-verified by ``scripts/03_build_circuits.py``.  Asserting these values means
# an upstream data change or a coding mistake cannot slip through silently.
# "neurons" is the count *selected by the rule*, before isolated neurons are
# pruned from the induced graph.
EXPECTED: dict[str, dict[str, int]] = {
    "core": {"neurons": 33_484, "edges": 1_170_024},
    "full": {"neurons": 99_170, "edges": 12_392_758},
}

KEEP_SUPERCLASSES = (
    "ol_intrinsic",
    "visual_projection",
    "cb_intrinsic",
    "visual_centrifugal",
)


def hop_targets(
    seed: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    n: int,
    keep: np.ndarray | None = None,
) -> np.ndarray:
    """Neurons receiving at least one synapse from ``seed`` (optionally filtered)."""
    src = seed[pre]
    targets = np.zeros(n, dtype=bool)
    np.logical_or.at(targets, post[src], True)
    if keep is not None:
        targets &= keep
    return targets


def select_neurons(
    ann: Annotations,
    pre: np.ndarray,
    post: np.ndarray,
    circuit: str,
) -> np.ndarray:
    """Rule-based neuron selection -> boolean mask over the annotation table."""
    hex_mask = ann.mask_hex
    vpn = ann.mask_vpn
    cen = ann.mask_cen

    if circuit == "core":
        return hex_mask | vpn | cen

    keep = np.isin(ann.superclass, list(KEEP_SUPERCLASSES))
    hop1 = hop_targets(hex_mask, pre, post, ann.n, keep=keep)

    if circuit == "c3":
        return hex_mask | hop1 | vpn | cen
    if circuit == "full":
        return ann.mask_ol_intrinsic | hop1 | vpn | cen

    raise ValueError(f"unknown circuit: {circuit!r} (core | c3 | full)")


@dataclass
class Subgraph:
    """An induced subgraph with compact 0..N-1 indices."""

    name: str
    body_ids: np.ndarray       # int64 (N,)  original MaleCNS ids
    pre: np.ndarray            # int32 (E,)  reindexed presynaptic
    post: np.ndarray           # int32 (E,)  reindexed postsynaptic
    weight: np.ndarray         # float32 (E,) synapse counts
    selection_mask: np.ndarray  # bool (ann.n,) neurons selected by the rule
    isolated_removed: int
    qc_extra: dict = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return len(self.body_ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    def qc(self) -> dict:
        n = self.n_neurons
        in_deg = np.bincount(self.post, minlength=n)
        out_deg = np.bincount(self.pre, minlength=n)
        base = {
            "circuit": self.name,
            "neurons": int(n),
            "edges": int(self.n_edges),
            "edges_per_neuron": round(self.n_edges / max(n, 1), 2),
            "isolated_removed": int(self.isolated_removed),
            "self_loops": int((self.pre == self.post).sum()),
            "synapses": float(self.weight.sum()),
            "in_degree_median": float(np.median(in_deg)),
            "in_degree_mean": float(in_deg.mean()),
            "in_degree_max": int(in_deg.max()),
            "out_degree_median": float(np.median(out_deg)),
            "out_degree_max": int(out_deg.max()),
            "weight_mean_log1p": float(np.log1p(self.weight).mean()),
        }
        base.update(self.qc_extra)
        return base

    # ------------------------------------------------------------------ #
    def save(self, path) -> "Path":  # noqa: F821
        import json
        from pathlib import Path as _P

        path = _P(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            body_ids=self.body_ids,
            pre=self.pre,
            post=self.post,
            weight=self.weight,
        )
        path.with_suffix(".qc.json").write_text(
            json.dumps(self.qc(), indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path) -> "Subgraph":
        """Load a cached subgraph, restoring the QC details saved alongside it.

        The weight-retention fractions are computed at build time; without
        restoring them a cache hit silently reports ``nan`` in every report.
        """
        import json
        from pathlib import Path as _P

        path = _P(path)
        with np.load(path) as z:
            sg = cls(
                name=path.stem,
                body_ids=z["body_ids"],
                pre=z["pre"],
                post=z["post"],
                weight=z["weight"],
                selection_mask=np.zeros(0, dtype=bool),
                isolated_removed=0,
            )
        qc_path = path.with_suffix(".qc.json")
        if qc_path.exists():
            try:
                stored = json.loads(qc_path.read_text(encoding="utf-8"))
                sg.isolated_removed = int(stored.get("isolated_removed", 0))
                for key in ("retained_out_weight_fraction", "retained_in_weight_fraction"):
                    if key in stored:
                        sg.qc_extra[key] = stored[key]
            except (json.JSONDecodeError, OSError):
                pass
        return sg


def induce(
    ann: Annotations,
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    *,
    name: str,
    prune_isolated: bool = True,
    logger=None,
) -> Subgraph:
    """Induce the subgraph on ``mask`` and reindex to 0..N-1."""
    keep_edge = mask[pre] & mask[post]
    e_pre = pre[keep_edge]
    e_post = post[keep_edge]
    e_w = weight[keep_edge]

    body_ids = ann.body_ids[mask]
    remap = np.full(ann.n, -1, dtype=np.int32)
    remap[mask] = np.arange(mask.sum(), dtype=np.int32)
    s_pre = remap[e_pre]
    s_post = remap[e_post]

    n_before = int(mask.sum())
    isolated_removed = 0
    if prune_isolated:
        touched = np.zeros(n_before, dtype=bool)
        touched[s_pre] = True
        touched[s_post] = True
        drop = ~touched
        isolated_removed = int(drop.sum())
        if isolated_removed:
            shift = np.cumsum(~drop) - 1
            s_pre = shift[s_pre].astype(np.int32)
            s_post = shift[s_post].astype(np.int32)
            body_ids = body_ids[~drop]

    sg = Subgraph(
        name=name,
        body_ids=np.asarray(body_ids, dtype=np.int64),
        pre=s_pre.astype(np.int32),
        post=s_post.astype(np.int32),
        weight=e_w.astype(np.float32),
        selection_mask=mask,
        isolated_removed=isolated_removed,
    )
    if logger:
        q = sg.qc()
        logger.info(
            "subgraph %-5s: %d neurons (%d selected, %d isolated pruned) | "
            "%d edges | E/N=%.1f | synapses=%.0f",
            name,
            q["neurons"],
            n_before,
            isolated_removed,
            q["edges"],
            q["edges_per_neuron"],
            q["synapses"],
        )
    return sg


def retention(pre, post, weight, mask: np.ndarray) -> dict:
    """Fraction of synaptic weight retained inside the subgraph."""
    inside = mask[pre] & mask[post]
    out_total = weight[mask[pre]].sum()
    in_total = weight[mask[post]].sum()
    inside_w = weight[inside].sum()
    return {
        "retained_out_weight_fraction": round(float(inside_w / max(out_total, 1.0)), 4),
        "retained_in_weight_fraction": round(float(inside_w / max(in_total, 1.0)), 4),
    }


def build_subgraph(
    ann: Annotations,
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    circuit: str,
    *,
    strict: bool = True,
    logger=None,
) -> Subgraph:
    """Select, induce, prune and validate one circuit."""
    mask = select_neurons(ann, pre, post, circuit)
    expect = EXPECTED.get(circuit)
    if strict and expect is not None:
        if int(mask.sum()) != expect["neurons"]:
            raise AssertionError(
                f"{circuit}: {int(mask.sum())} neurons selected, expected {expect['neurons']}"
            )
        induced = int((mask[pre] & mask[post]).sum())
        if induced != expect["edges"]:
            raise AssertionError(
                f"{circuit}: {induced} induced edges, expected {expect['edges']}"
            )
    ret = retention(pre, post, weight, mask)
    sg = induce(ann, pre, post, weight, mask, name=circuit, logger=logger)
    sg.qc_extra.update(ret)
    if logger:
        logger.info(
            "  retention: %.1f%% of outgoing, %.1f%% of incoming synaptic weight",
            ret["retained_out_weight_fraction"] * 100,
            ret["retained_in_weight_fraction"] * 100,
        )
    return sg
