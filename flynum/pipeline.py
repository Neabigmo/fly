"""Assemble a complete, ready-to-train experiment from a configuration.

This module is the single place where the pieces are wired together, so that
training scripts stay thin and every run is assembled identically:

    annotations -> edge list -> circuit (real or shuffled) -> retina map
                -> readout population -> :class:`ConnectomeRNN`
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import paths
from .config import ExperimentConfig
from .data import connectome as cx
from .data.annotations import Annotations, load_annotations
from .data.manifest import Manifest
from .data.shuffle import ShuffleResult, shuffle_edges
from .data.subgraph import Subgraph, build_subgraph
from .models.connectome_rnn import ConnectomeRNN, make_base_weights
from .models.sparse_lin import SparseConnectome
from .retina.encoder import RetinaEncoder, select_input_neurons
from .retina.hexmap import build_hex_field, column_lookup

CIRCUIT_DIR = paths.DATA_PROCESSED / "circuits"


# --------------------------------------------------------------------------- #
def get_subgraph(
    ann: Annotations,
    circuit: str,
    *,
    force: bool = False,
    strict: bool = True,
    logger=None,
) -> Subgraph:
    """Build (and cache) the induced subgraph for a named circuit."""
    path = CIRCUIT_DIR / f"{circuit}.npz"
    if path.exists() and not force:
        sg = Subgraph.load(path)
        if logger:
            q = json.loads(path.with_suffix(".qc.json").read_text(encoding="utf-8"))
            logger.info(
                "circuit %s loaded from cache: %s neurons / %s edges",
                circuit,
                q["neurons"],
                q["edges"],
            )
        return sg
    edges = cx.load_edge_list()
    sg = build_subgraph(
        ann, edges["pre"], edges["post"], edges["weight"], circuit,
        strict=strict, logger=logger,
    )
    sg.save(path)
    return sg


def get_graph(
    sg: Subgraph, cfg: ExperimentConfig, *, force: bool = False, logger=None
) -> ShuffleResult:
    """Return the real edge list or a degree/weight-preserving shuffle of it.

    The shuffled *edges* are cached alongside their statistics, and the result is
    always validated against the real graph before use.  An earlier version
    cached only the statistics and then returned ``sg.pre/sg.post`` -- the real
    connectome -- so every "shuffled" run silently trained on the real graph and
    the two conditions produced identical numbers.  That is exactly the kind of
    silent failure that fabricates a null result, so it is now checked
    explicitly on every load.
    """
    if cfg.data.graph == "real":
        return ShuffleResult(sg.pre, sg.post, sg.weight, {"graph": "real"})

    path = (
        CIRCUIT_DIR
        / f"{sg.name}_shuffled_seed{cfg.seeds['shuffle_seed']}"
        f"_r{cfg.data.shuffle_rounds_per_edge:g}.npz"
    )
    if path.exists() and not force:
        with np.load(path) as z:
            if "pre" not in z.files:
                # statistics-only cache written by the buggy version: rebuild
                if logger:
                    logger.warning(
                        "cached shuffle %s has no edges (written by an older "
                        "version); regenerating", path.name,
                    )
            else:
                stats = json.loads(str(z["stats"]))
                pre, post, weight = z["pre"], z["post"], z["weight"]
                _validate_shuffle(sg, pre, post, weight, stats, logger)
                if logger:
                    logger.info(
                        "shuffled graph loaded from cache (seed=%d, overlap=%.2f%%)",
                        cfg.seeds["shuffle_seed"],
                        100 * stats["final_edge_overlap"],
                    )
                return ShuffleResult(pre, post, weight, stats)

    res = shuffle_edges(
        sg.pre,
        sg.post,
        sg.weight,
        sg.n_neurons,
        rounds_per_edge=cfg.data.shuffle_rounds_per_edge,
        seed=cfg.seeds["shuffle_seed"],
        logger=logger,
    )
    _validate_shuffle(sg, res.pre, res.post, res.weight, res.stats, logger)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pre=res.pre,
        post=res.post,
        weight=res.weight,
        stats=json.dumps(res.stats),
    )
    return res


def _validate_shuffle(
    sg: Subgraph, pre, post, weight, stats: dict, logger
) -> None:
    """Fail loudly if a 'shuffled' graph is not actually a rewiring of the real one."""
    from .data.shuffle import edge_overlap

    n = sg.n_neurons
    if not np.array_equal(np.sort(weight), np.sort(sg.weight)):
        raise AssertionError("shuffled control does not preserve the weight multiset")
    if not np.array_equal(
        np.bincount(post, minlength=n), np.bincount(sg.post, minlength=n)
    ):
        raise AssertionError("shuffled control does not preserve in-degrees")
    if not np.array_equal(
        np.bincount(pre, minlength=n), np.bincount(sg.pre, minlength=n)
    ):
        raise AssertionError("shuffled control does not preserve out-degrees")
    overlap = edge_overlap(sg.pre, sg.post, pre, post, n)
    if overlap > 0.5:
        raise AssertionError(
            f"'shuffled' graph still shares {100 * overlap:.1f}% of its edges with "
            "the real connectome - the control would be meaningless"
        )
    if logger:
        logger.info(
            "  shuffle verified: overlap with real graph %.2f%%, degrees and "
            "weights preserved", 100 * overlap,
        )


# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    """Everything a training run needs."""

    cfg: ExperimentConfig
    ann: Annotations
    subgraph: Subgraph
    graph: ShuffleResult
    field: object                     # HexField
    encoder: RetinaEncoder
    readout_name: str
    readout_idx: np.ndarray           # indices into the subgraph
    input_rows: np.ndarray
    input_cols: np.ndarray
    n_classes: int
    label_offset: int
    edge_sign: np.ndarray | None = None
    sign_report: dict | None = None
    info: dict = field(default_factory=dict)

    def build_model(
        self, *, base_weight: torch.Tensor | None = None,
        device: torch.device | str = "cpu", seed: int = 0, n_heads: int = 1,
        head_classes: tuple[int, ...] | None = None,
    ) -> ConnectomeRNN:
        dev = torch.device(device)
        conn = SparseConnectome(
            torch.as_tensor(self.graph.pre.astype(np.int64)),
            torch.as_tensor(self.graph.post.astype(np.int64)),
            self.subgraph.n_neurons,
            device=dev,
        )
        if base_weight is None:
            edge_sign = (
                torch.as_tensor(self.edge_sign) if self.edge_sign is not None else None
            )
            base_weight, _ = make_base_weights(
                torch.as_tensor(self.graph.pre.astype(np.int64)),
                torch.as_tensor(self.graph.post.astype(np.int64)),
                torch.as_tensor(self.graph.weight.astype(np.float32)),
                self.subgraph.n_neurons,
                w_scale=self.cfg.model_cfg.w_scale,
                normalization=self.cfg.model_cfg.weight_normalization,
                edge_sign=edge_sign,
                device=dev,
            )
        model = ConnectomeRNN(
            conn,
            base_weight,
            n_neurons=self.subgraph.n_neurons,
            n_classes=self.n_classes,
            n_columns=self.field.n_cols,
            input_rows=torch.as_tensor(self.input_rows.astype(np.int64)),
            input_cols=torch.as_tensor(self.input_cols.astype(np.int64)),
            readout_idx=torch.as_tensor(self.readout_idx.astype(np.int64)),
            alpha=self.cfg.model_cfg.alpha,
            nonlinearity=self.cfg.model_cfg.nonlinearity,
            readout_window=self.cfg.model_cfg.readout_window,
            standardize=self.cfg.model_cfg.readout_standardize,
            learn_gains=(self.cfg.model == "M1") and self.cfg.model_cfg.learn_gains,
            n_heads=n_heads,
            head_classes=head_classes,
            device=dev,
        )
        return model


# --------------------------------------------------------------------------- #
def choose_readout(
    ann: Annotations,
    subgraph: Subgraph,
    kind: str,
    input_body_ids: set[int],
) -> tuple[np.ndarray, str]:
    """Index set of the neurons the linear readout reads from."""
    body = subgraph.body_ids
    sub_ann = ann.index_of(body)
    superclass = ann.superclass[sub_ann]
    cell_type = ann.cell_type[sub_ann]

    is_vpn = superclass == "visual_projection"
    is_input = np.isin(body, list(input_body_ids))

    if kind == "vpn":
        idx = np.nonzero(is_vpn)[0]
        name = "visual_projection (VPN)"
    elif kind == "all_except_input":
        idx = np.nonzero(~is_input)[0]
        name = "all neurons except the injected input cells"
    elif kind == "all":
        idx = np.arange(subgraph.n_neurons)
        name = "all neurons"
    elif kind == "lc11":
        idx = np.nonzero(cell_type == "LC11")[0]
        name = "LC11 only"
    else:
        raise ValueError(f"unknown readout population: {kind!r}")
    if len(idx) == 0:
        raise ValueError(f"readout population {kind!r} is empty in this circuit")
    return idx.astype(np.int64), name


def prepare(
    cfg: ExperimentConfig,
    *,
    logger=None,
    force: bool = False,
    strict_circuit: bool = True,
    n_classes: int | None = None,
    label_offset: int | None = None,
) -> Prepared:
    """Wire up everything for one experiment configuration.

    ``n_classes`` and ``label_offset`` override the task head.  The Phase I tasks
    (rule selection, two-step arithmetic, cyclic addition) each have their own answer
    space, and a task name is not enough to derive it -- ``addsub`` shares a 0-8 space
    between a sum and a difference, so the caller that owns the task table states the
    head explicitly instead of this function guessing from a string.
    """
    if cfg.retina.image_size != cfg.stimulus.image_size:
        raise ValueError(
            "retina.image_size and stimulus.image_size must match "
            f"({cfg.retina.image_size} != {cfg.stimulus.image_size})"
        )

    if logger:
        logger.info("-" * 78)
        logger.info(
            "preparing: circuit=%s graph=%s retina=%s/%s/%s readout=%s",
            cfg.data.circuit, cfg.data.graph, cfg.retina.map_mode,
            cfg.retina.input_types, cfg.retina.input_eye, cfg.model_cfg.readout,
        )
    ann = load_annotations()
    sg = get_subgraph(
        ann, cfg.data.circuit, force=force, strict=strict_circuit, logger=logger
    )
    graph = get_graph(sg, cfg, force=force, logger=logger)

    # ---- retinotopy -------------------------------------------------- #
    sub_ann = ann.index_of(sg.body_ids)
    h1 = ann.hex1[sub_ann]
    h2 = ann.hex2[sub_ann]
    side = ann.soma_side[sub_ann]
    has_hex = ~np.isnan(h1)
    field = build_hex_field(h1[has_hex], h2[has_hex], side[has_hex])
    column_of_neuron = np.full(sg.n_neurons, -1, dtype=np.int32)
    column_of_neuron[has_hex] = column_lookup(field, h1[has_hex], h2[has_hex], side[has_hex])
    if logger:
        q = field.qc()
        logger.info(
            "hex field: %d columns | bounds x[%.1f,%.1f] y[%.1f,%.1f] | "
            "NN=%.4f | density=%.3f (ideal %.3f)",
            q["n_columns"], *q["bounds"], q["nn_distance_mean"], q["density"],
            q["ideal_hex_density"],
        )

    input_mask = select_input_neurons(
        ann.cell_type[sub_ann], side, column_of_neuron,
        input_types=cfg.retina.input_types, input_eye=cfg.retina.input_eye,
    )
    encoder = RetinaEncoder(
        field,
        column_of_neuron,
        input_mask,
        image_size=cfg.retina.image_size,
        map_mode=cfg.retina.map_mode,
        input_types=cfg.retina.input_types,
        input_eye=cfg.retina.input_eye,
    )
    if logger:
        logger.info(
            "retina map: %d/%d columns carry input neurons; %d of them fall "
            "inside the image window (%.1f%%) | %d input neurons",
            encoder.qc.n_columns_with_input,
            encoder.qc.n_columns,
            encoder.qc.n_driven_columns,
            100 * encoder.qc.driven_fraction,
            encoder.qc.n_input_neurons,
        )

    # ---- signed synapses (Fly-v2) ------------------------------------- #
    edge_sign = None
    sign_report = None
    if cfg.model_cfg.signed_synapses:
        from .data.neurotransmitters import edge_signs, load_neuron_signs

        neuron_signs, rep = load_neuron_signs(
            ann, unknown_sign=cfg.model_cfg.unknown_nt_sign
        )
        sub_signs = neuron_signs[sub_ann]
        edge_sign = edge_signs(graph.pre, sub_signs)
        sign_report = rep.as_dict()
        if logger:
            logger.info(
                "signed synapses: %d/%d neurons carry a prediction, "
                "%d excitatory / %d inhibitory / %d defaulted to %+d "
                "(%.1f%% signed)",
                rep.n_with_prediction, rep.n_neurons, rep.n_excitatory,
                rep.n_inhibitory, rep.n_unknown, rep.unknown_sign,
                100 * rep.fraction_signed,
            )

    # ---- readout population ------------------------------------------ #
    input_body_ids = set(sg.body_ids[input_mask].tolist())
    readout_idx, readout_name = choose_readout(
        ann, sg, cfg.model_cfg.readout, input_body_ids
    )
    if logger:
        logger.info("readout: %s -> %d neurons", readout_name, len(readout_idx))

    # ---- task head ---------------------------------------------------- #
    # An explicit head always wins: Phase I's addition task uses operands 1-7 (13 sum
    # classes) while the legacy ``add`` task derives 7 classes from ``a_max = 4``, so
    # the name alone does not determine the head.
    if n_classes is None or label_offset is None:
        if cfg.task == "count":
            n_classes = cfg.stimulus.n_max - cfg.stimulus.n_min + 1
            label_offset = cfg.stimulus.n_min
        elif cfg.task == "add":
            n_classes = 2 * cfg.stimulus.a_max - 2 * cfg.stimulus.a_min + 1
            label_offset = 2 * cfg.stimulus.a_min
        else:
            raise ValueError(f"unknown task {cfg.task!r}")

    info = {
        "circuit": sg.name,
        # ``graph`` belongs here: run summaries are keyed on it (the learning curve,
        # the headline table and the fixed-update table all select by graph), and its
        # absence silently removed every run that recorded it as null from those
        # analyses while still leaving them in the aggregate counts.
        "graph": cfg.data.graph,
        "n_neurons": sg.n_neurons,
        "n_edges": sg.n_edges,
        "n_classes": n_classes,
        "label_offset": label_offset,
        "readout_population": cfg.model_cfg.readout,
        "readout_name": readout_name,
        "n_readout_neurons": int(len(readout_idx)),
        "n_input_neurons": int(encoder.n_input_neurons),
        "n_columns_with_input": encoder.qc.n_columns_with_input,
        "n_driven_columns": encoder.qc.n_driven_columns,
        "n_columns": field.n_cols,
        "hex_bounds": field.qc()["bounds"],
        "hex_density": field.qc()["density"],
        "circuit_qc": sg.qc(),
        "shuffle_stats": graph.stats,
        "signed_synapses": bool(cfg.model_cfg.signed_synapses),
        "sign_report": sign_report,
    }
    return Prepared(
        cfg=cfg,
        ann=ann,
        subgraph=sg,
        graph=graph,
        field=field,
        encoder=encoder,
        readout_name=readout_name,
        readout_idx=readout_idx,
        input_rows=encoder.input_rows,
        input_cols=encoder.input_cols,
        n_classes=n_classes,
        label_offset=label_offset,
        edge_sign=edge_sign,
        sign_report=sign_report,
        info=info,
    )


def cache_path(name: str) -> Path:
    return paths.DATA_PROCESSED / name


def load_w_scale(circuit: str = "core") -> float:
    """Recurrent weight scale calibrated for ``circuit``.

    ``w_scale`` is circuit specific (the degree heterogeneity differs by an order
    of magnitude between ``core`` and ``full``: reusing the core value on ``full``
    diverges within one epoch).  The legacy single-file name is only accepted for
    ``core`` -- falling back to it for another circuit would silently apply the
    wrong calibration.
    """
    names = [f"calibration_{circuit}.json"]
    if circuit == "core":
        names.append("calibration.json")
    for name in names:
        path = paths.DATA_PROCESSED / name
        if path.exists():
            try:
                return float(json.loads(path.read_text(encoding="utf-8"))["chosen_w_scale"])
            except (json.JSONDecodeError, KeyError):
                continue
    return 1.0
