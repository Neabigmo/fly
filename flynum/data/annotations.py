"""Neuron-level annotations from ``body-annotations-...-minconf-0.5.feather``.

The annotation table is the *universe* of the study: only bodies present here
are eligible neurons, and its columns supply cell type, superclass and -- most
importantly -- the ``assignedOlHex1/2`` retinotopic column coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths

#: Cell types carrying ``assignedOlHex1/2`` (verified: exactly these 15).
HEX_TYPES: tuple[str, ...] = (
    "L1", "L2", "L3", "L5", "T1", "C2", "C3",
    "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9", "Tm20",
)

#: Lamina cell types that receive direct photoreceptor input -> image injection
#: targets for ``input_types="lamina"``.
LAMINA_TYPES: tuple[str, ...] = ("L1", "L2", "L3", "L5", "T1")
#: Add the lamina centrifugal cells.
LAMINA_ALL_TYPES: tuple[str, ...] = LAMINA_TYPES + ("C2", "C3")

INPUT_TYPE_SETS: dict[str, tuple[str, ...]] = {
    "lamina": LAMINA_TYPES,
    "lamina_all": LAMINA_ALL_TYPES,
    "all_hex": HEX_TYPES,
}


@dataclass
class Annotations:
    """Immutable, array-oriented view of the MaleCNS annotation table."""

    df: pd.DataFrame
    body_ids: np.ndarray          # int64, length N
    superclass: np.ndarray        # object, length N
    cell_class: np.ndarray        # object
    cell_type: np.ndarray         # object
    soma_side: np.ndarray         # object
    hex1: np.ndarray              # float32 with NaN
    hex2: np.ndarray              # float32 with NaN
    type_codes: np.ndarray        # int32 type index
    type_vocab: tuple[str, ...]

    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return len(self.body_ids)

    def _sorted_index(self) -> tuple[np.ndarray, np.ndarray]:
        """Cached (sorted body ids, original positions) for id -> row lookup."""
        cached = getattr(self, "_sorted_cache", None)
        if cached is None:
            order = np.argsort(self.body_ids, kind="stable")
            cached = (self.body_ids[order], order.astype(np.int64))
            object.__setattr__(self, "_sorted_cache", cached)
        return cached

    def index_of(self, body_ids: np.ndarray | list[int]) -> np.ndarray:
        """Map bodyId -> positional index (-1 when absent).

        MaleCNS body ids are sparse over a very wide range (max id 1,571,825,087
        against only ~211k rows), so a dense lookup table would need 12.6 GB.
        A binary search over the sorted ids is both tiny and faster to build.
        """
        sorted_ids, sorted_pos = self._sorted_index()
        ids = np.asarray(body_ids, dtype=np.int64)
        pos = np.searchsorted(sorted_ids, ids)
        pos_clipped = np.clip(pos, 0, len(sorted_ids) - 1)
        hit = sorted_ids[pos_clipped] == ids
        return np.where(hit, sorted_pos[pos_clipped], -1).astype(np.int64)

    # --- masks -------------------------------------------------------- #
    def mask_superclass(self, *names: str) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        for nm in names:
            m |= self.superclass == nm
        return m

    def mask_types(self, types) -> np.ndarray:
        return np.isin(self.cell_type, list(types))

    @property
    def mask_hex(self) -> np.ndarray:
        return ~np.isnan(self.hex1)

    @property
    def mask_vpn(self) -> np.ndarray:
        """Visual projection neurons -- the optic-lobe output / readout pool."""
        return self.mask_superclass("visual_projection")

    @property
    def mask_cen(self) -> np.ndarray:
        return self.mask_superclass("visual_centrifugal")

    @property
    def mask_ol_intrinsic(self) -> np.ndarray:
        return self.mask_superclass("ol_intrinsic")

    @property
    def mask_photoreceptor(self) -> np.ndarray:
        return self.mask_superclass("ol_sensory")

    # ------------------------------------------------------------------ #
    def qc(self) -> dict:
        hex_mask = self.mask_hex
        q: dict = {
            "n_bodies": int(self.n),
            "superclass_counts": {
                str(k): int(v)
                for k, v in pd.Series(self.superclass).value_counts().items()
            },
            "hex_neurons": int(hex_mask.sum()),
            "hex_types": {
                str(k): int(v)
                for k, v in pd.Series(self.cell_type[hex_mask]).value_counts().items()
            },
            "hex_columns": int(
                len(np.unique(np.stack([self.hex1[hex_mask], self.hex2[hex_mask]], 1), axis=0))
            ),
            "lc11_neurons": int((self.cell_type == "LC11").sum()),
        }
        for side in ("L", "R"):
            m = hex_mask & (self.soma_side == side)
            if m.sum():
                cols = np.unique(
                    np.stack([self.hex1[m], self.hex2[m]], 1), axis=0
                )
                q[f"hex_columns_{side}"] = int(len(cols))
                q[f"hex_types_{side}"] = int(len(np.unique(self.cell_type[m])))
        return q


def _str_col(series: pd.Series, fill: str = "") -> np.ndarray:
    """Convert an object/string column to ``dtype=object`` with NaN -> ``fill``.

    pandas 3.0 keeps NaN as a float inside string columns, which makes
    ``np.unique(..., return_inverse=True)`` fail with a float/str comparison.
    Normalising here keeps everything downstream orderable.
    """
    out = np.empty(len(series), dtype=object)
    for i, v in enumerate(series.to_numpy(dtype=object)):
        if v is None:
            out[i] = fill
        elif isinstance(v, float) and np.isnan(v):
            out[i] = fill
        else:
            out[i] = str(v)
    return out


def load_annotations(path: Path | None = None) -> Annotations:
    """Load and index the annotation table."""
    path = path or paths.raw_path("annotations")
    if not path.exists():
        raise FileNotFoundError(f"annotations not found: {path} (run download first)")

    df = pd.read_feather(path)
    body_ids = df["bodyId"].to_numpy().astype(np.int64)
    if len(np.unique(body_ids)) != len(body_ids):
        raise ValueError("duplicate bodyId values in annotation table")

    cell_type = _str_col(df["type"], fill="__untyped__")
    vocab, codes = np.unique(cell_type, return_inverse=True)
    return Annotations(
        df=df,
        body_ids=body_ids,
        superclass=_str_col(df["superclass"], fill="__unassigned__"),
        cell_class=_str_col(df["class"]),
        cell_type=cell_type,
        soma_side=_str_col(df["somaSide"], fill="NA"),
        hex1=df["assignedOlHex1"].to_numpy().astype(np.float32),
        hex2=df["assignedOlHex2"].to_numpy().astype(np.float32),
        type_codes=codes.astype(np.int32),
        type_vocab=tuple(vocab.tolist()),
    )


def write_qc(ann: Annotations, out: Path | None = None) -> Path:
    out = out or (paths.DATA_PROCESSED / "annotations_qc.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ann.qc(), indent=2), encoding="utf-8")
    return out
