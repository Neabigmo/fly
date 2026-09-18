"""Data build for fly2: one cached whole-brain circuit, everything else reads it.

Parsing the public MaleCNS release is the one job this module borrows the previous
project for: the edge-list parsing, the annotation table and the retinotopic field are
already validated against the release, and rewriting them would only add a way to be
subtly wrong about the data.  From the cache onward -- dynamics, plasticity, probes --
``fly2`` is self-contained.

What the cache holds (all arrays indexed 0..N-1 over the kept neurons):

``pre``, ``post``, ``weight``     the connectome
``superclass``, ``cell_type``     strings, plus ``alpha`` per neuron
``edge_sign``                     transmitter sign of the presynaptic cell
``retina``                        lamina input neurons, and their image sampling map
``odor``                          olfactory receptor neurons and their binding matrix
``readout_visual``, ``readout_mb``, ``kenyon``, ``dopamine``, ``octopamine``
``lesion_mask``                   MB output synapses, for the causal control
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import spec

CACHE_NAME = "fly2_brain.npz"


# --------------------------------------------------------------------------- #
@dataclass
class BrainData:
    """The cached substrate, loaded for use."""

    body_ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    edge_sign: np.ndarray
    superclass: np.ndarray
    cell_type: np.ndarray
    alpha: np.ndarray
    retina: np.ndarray
    retina_col: np.ndarray
    odor: np.ndarray
    readout_visual: np.ndarray
    readout_mb: np.ndarray
    kenyon: np.ndarray
    dopamine: np.ndarray
    octopamine: np.ndarray
    coverage_mask: np.ndarray
    col_u: np.ndarray
    col_v: np.ndarray
    n_columns: int
    odour_binding: np.ndarray
    meta: dict

    @property
    def n(self) -> int:
        return len(self.body_ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    def summary(self) -> dict:
        return {
            "neurons": int(self.n), "edges": int(self.n_edges),
            "retina": int(len(self.retina)), "odor": int(len(self.odor)),
            "readout_visual": int(len(self.readout_visual)),
            "readout_mb": int(len(self.readout_mb)),
            "kenyon": int(len(self.kenyon)), "dopamine": int(len(self.dopamine)),
            "octopamine": int(len(self.octopamine)),
            "columns": int(self.n_columns),
            "negative_edges_fraction": float((self.edge_sign < 0).mean()),
        }


# --------------------------------------------------------------------------- #
def _pools(ann_idx, superclass, cell_type) -> dict[str, np.ndarray]:
    """Resolve every named population against the cached neurons."""
    out: dict[str, np.ndarray] = {}
    for name, rule in spec.POOLS.items():
        if rule["kind"] == "superclass":
            m = superclass == rule["value"]
        elif rule["kind"] == "celltype_prefix":
            pre = tuple(str(p).lower() for p in rule["value"])
            m = np.array([any(str(t).lower().startswith(p) for p in pre)
                          for t in cell_type])
        else:
            raise ValueError(rule["kind"])
        out[name] = np.nonzero(m)[0].astype(np.int64)
    return out


def build(cache_dir: Path, *, logger=None) -> BrainData:
    """Parse the release once and write ``fly2_brain.npz``."""
    # imported here only: the build step is allowed to read the release through the
    # previous project's validated parsers, the runtime never touches it
    import pandas as pd

    from flynum import paths
    from flynum.data import connectome as cx
    from flynum.data.annotations import load_annotations
    from flynum.data.neurotransmitters import NT_SIGN
    from flynum.retina.encoder import RetinaEncoder, select_input_neurons
    from flynum.retina.hexmap import build_hex_field, column_lookup
    from flynum.config import ExperimentConfig

    log = logger
    ann = load_annotations()
    edges = cx.load_edge_list()
    pre_raw, post_raw, w_raw = edges["pre"], edges["post"], edges["weight"]

    # ---- neuron set ---------------------------------------------------------- #
    degree = np.bincount(pre_raw, minlength=ann.n) + np.bincount(post_raw, minlength=ann.n)
    keep = np.ones(ann.n, dtype=bool) if spec.BRAIN["keep_isolated"] else degree > 0
    remap = np.full(ann.n, -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    n = int(keep.sum())
    edge_keep = keep[pre_raw] & keep[post_raw]
    pre, post, weight = remap[pre_raw[edge_keep]], remap[post_raw[edge_keep]], \
        w_raw[edge_keep].astype(np.float32)
    if log:
        log.info("kept %d of %d annotated neurons (with >=1 edge); %d edges",
                 n, ann.n, len(pre))

    sub = np.nonzero(keep)[0]
    superclass = np.asarray(ann.superclass[sub], dtype=object)
    cell_type = np.asarray(ann.cell_type[sub], dtype=object)

    # ---- rich cell types from the transmitter file --------------------------- #
    nt = pd.read_feather(paths.raw_path("neurotransmitters"))[["body", "cell_type",
                                                               "predicted_nt"]]
    per_body_type = (nt.dropna(subset=["cell_type"]).groupby("body")["cell_type"].first())
    per_body_nt = nt.groupby("body")["predicted_nt"].first()
    body_pos = {int(b): i for i, b in enumerate(ann.body_ids[sub])}
    n_typed = 0
    for body, t in per_body_type.items():
        i = body_pos.get(int(body))
        if i is not None:
            cell_type[i] = str(t)
            n_typed += 1
    if log:
        log.info("cell types enriched for %d neurons from the transmitter file", n_typed)

    # ---- transmitter sign of each presynaptic cell --------------------------- #
    sign_of_neuron = np.full(n, float(spec.BRAIN["unknown_sign"]), dtype=np.float32)
    known = 0
    for body, nt_type in per_body_nt.items():
        i = body_pos.get(int(body))
        if i is None:
            continue
        s = NT_SIGN.get(str(nt_type).lower())
        if s is not None:
            sign_of_neuron[i] = float(s)
            known += 1
    if log:
        log.info("transmitter sign resolved for %d neurons (%.1f%% inhibitory)",
                 known, 100 * float((sign_of_neuron < 0).mean()))
    edge_sign = sign_of_neuron[pre]

    # ---- per-superclass leak -------------------------------------------------- #
    alpha = np.full(n, spec.ALPHA_DEFAULT, dtype=np.float32)
    for sup, a in spec.ALPHA_BY_SUPERCLASS.items():
        alpha[superclass == sup] = a

    # ---- populations --------------------------------------------------------- #
    pools = _pools(sub, superclass, cell_type)
    for name, cells in pools.items():
        if log:
            log.info("population %-16s %6d", name, len(cells))

    # ---- retinotopy ---------------------------------------------------------- #
    h1, h2 = ann.hex1[sub], ann.hex2[sub]
    side = ann.soma_side[sub]
    has_hex = ~np.isnan(h1)
    field = build_hex_field(h1[has_hex], h2[has_hex], side[has_hex])
    col_of = np.full(n, -1, dtype=np.int32)
    col_of[has_hex] = column_lookup(field, h1[has_hex], h2[has_hex], side[has_hex])
    cfg = ExperimentConfig()
    cfg.retina.image_size = spec.WORLD["image_size"]
    input_mask = select_input_neurons(cell_type, side, col_of,
                                      input_types=cfg.retina.input_types,
                                      input_eye=cfg.retina.input_eye)
    encoder = RetinaEncoder(field, col_of, input_mask,
                            image_size=spec.WORLD["image_size"],
                            map_mode=cfg.retina.map_mode,
                            input_types=cfg.retina.input_types,
                            input_eye=cfg.retina.input_eye)
    retina = np.nonzero(input_mask)[0].astype(np.int64)
    retina_col = np.clip(col_of[retina], 0, field.n_cols - 1).astype(np.int64)
    if log:
        log.info("retinotopy: %d columns, %d driven, %d lamina input neurons",
                 field.n_cols, len(encoder.driven_columns), len(retina))

    # ---- odour binding matrix ------------------------------------------------ #
    rng = np.random.default_rng(spec.WORLD.setdefault("odour_seed", 2024))
    odor = pools["odor"]
    binding = rng.random((len(odor), spec.WORLD["n_odour_channels"])).astype(np.float32)
    binding[binding < (1 - spec.WORLD["odour_sparsity"])] = 0.0

    # ---- MB output synapses, for the causal lesion --------------------------- #
    kc, mbon, dan, oa = (pools["kenyon"], pools["readout_mb"], pools["dopamine"],
                         pools["octopamine"])
    mb_sources = np.concatenate([kc, mbon, dan, oa])
    lesion_mask = np.isin(pre, mb_sources)

    data = BrainData(
        body_ids=ann.body_ids[sub].astype(np.int64), pre=pre.astype(np.int32),
        post=post.astype(np.int32), weight=weight.astype(np.float32),
        edge_sign=edge_sign.astype(np.float32), superclass=superclass,
        cell_type=cell_type, alpha=alpha, retina=retina.astype(np.int64),
        retina_col=retina_col,
        odor=odor.astype(np.int64), readout_visual=pools["readout_visual"],
        readout_mb=mbon, kenyon=kc, dopamine=dan, octopamine=oa,
        coverage_mask=encoder.coverage_mask().astype(bool),
        col_u=np.asarray(encoder.col_uv[:, 0], dtype=np.float32),
        col_v=np.asarray(encoder.col_uv[:, 1], dtype=np.float32),
        n_columns=int(field.n_cols), odour_binding=binding,
        meta={"columns_with_input": int(encoder.qc.n_columns_with_input),
              "driven_columns": int(encoder.qc.n_driven_columns),
              "mb_output_synapses": int(lesion_mask.sum()),
              "sign_resolved": int(known), "typed": int(n_typed)},
    )
    save(data, cache_dir)
    return data


# --------------------------------------------------------------------------- #
def save(data: BrainData, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_NAME
    np.savez(
        path, pre=data.pre, post=data.post, weight=data.weight,
        edge_sign=data.edge_sign, superclass=data.superclass.astype("U64"),
        cell_type=data.cell_type.astype("U64"), alpha=data.alpha,
        body_ids=data.body_ids, retina=data.retina, retina_col=data.retina_col,
        odor=data.odor,
        readout_visual=data.readout_visual, readout_mb=data.readout_mb,
        kenyon=data.kenyon, dopamine=data.dopamine, octopamine=data.octopamine,
        coverage_mask=data.coverage_mask, col_u=data.col_u, col_v=data.col_v,
        odour_binding=data.odour_binding,
        meta=json.dumps({**data.meta, "n_columns": data.n_columns}),
    )
    return path


def load(cache_dir: Path) -> BrainData:
    path = cache_dir / CACHE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/40_fly2_build.py")
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return BrainData(
            body_ids=z["body_ids"], pre=z["pre"], post=z["post"], weight=z["weight"],
            edge_sign=z["edge_sign"], superclass=z["superclass"],
            cell_type=z["cell_type"], alpha=z["alpha"], retina=z["retina"],
            retina_col=z["retina_col"], odor=z["odor"],
            readout_visual=z["readout_visual"],
            readout_mb=z["readout_mb"], kenyon=z["kenyon"], dopamine=z["dopamine"],
            octopamine=z["octopamine"], coverage_mask=z["coverage_mask"],
            col_u=z["col_u"], col_v=z["col_v"], n_columns=int(meta["n_columns"]),
            odour_binding=z["odour_binding"], meta=meta,
        )
