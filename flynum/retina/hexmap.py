"""Retinotopic mapping from the ``assignedOlHex`` columns.

Verified in the feasibility study
---------------------------------
``assignedOlHex1/2`` cover exactly 23,720 neurons, all ``ol_intrinsic``, in
15 columnar cell types.  Occupancy forms a diamond in ``(hex1, hex2)`` space
with nearest-neighbour distance exactly 1.0000 and convex-hull density 1.193
(ideal hexagonal 2/sqrt(3) = 1.1547, ideal square 1.0).  The columns are
therefore **axial coordinates of a hexagonal lattice**, and the correct
Cartesian embedding is::

    x = q + r / 2
    y = r * sqrt(3) / 2

with ``q = hex1``, ``r = hex2``.  For the right optic lobe this yields 892
columns spanning x in [4.5, 53.5] and y in [0.9, 33.8] -- an elliptical visual
field containing ~892 ommatidial columns, consistent with a male
Drosophila eye.  The left lobe uses the same grid with 879 of the 892 columns
occupied (it lacks C2, L3 and Tm4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SQRT3_2 = np.sqrt(3.0) / 2.0


def hex_to_cartesian(hex1: np.ndarray, hex2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axial hexagonal coordinates -> Cartesian (validated in the feasibility study)."""
    x = hex1 + hex2 / 2.0
    y = hex2 * SQRT3_2
    return x, y


@dataclass
class HexField:
    """The visual field as a set of hexagonal columns."""

    columns: np.ndarray        # (n_cols, 2) float32 axial (q, r)
    xy: np.ndarray             # (n_cols, 2) float32 Cartesian
    side: np.ndarray           # (n_cols,) object: 'L' or 'R'
    col_index: dict            # (side, q, r) -> column row index

    @property
    def n_cols(self) -> int:
        return len(self.columns)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self.xy[:, 0].min()),
            float(self.xy[:, 0].max()),
            float(self.xy[:, 1].min()),
            float(self.xy[:, 1].max()),
        )

    @property
    def centroid(self) -> np.ndarray:
        return self.xy.mean(axis=0)

    def qc(self) -> dict:
        """Geometric sanity check.

        The left and right optic lobes share the same ``(hex1, hex2)`` grid and
        therefore the same Cartesian coordinates, so the geometry has to be
        measured on the *unique* coordinate set -- otherwise every column is
        counted twice and the nearest-neighbour distance collapses to zero.
        """
        from scipy.spatial import ConvexHull, cKDTree

        xy_unique = np.unique(self.xy.astype(np.float64), axis=0)
        d, _ = cKDTree(xy_unique).query(xy_unique, k=2)
        nn = d[:, 1]
        x0, x1, y0, y1 = self.bounds

        # The hull-based density is biased upwards for small or degenerate
        # fields (boundary points own cells that stick out of the hull), so a
        # local (radius-8) density is reported alongside it.  ``ideal_hex_density``
        # is 2/sqrt(3) = 1.1547.
        hull_area = float("nan")
        density = float("nan")
        try:
            if len(xy_unique) >= 3:
                hull_area = float(ConvexHull(xy_unique).volume)
                if hull_area > 1e-9:
                    density = float(len(xy_unique) / hull_area)
        except Exception:  # degenerate point set
            pass

        local_density = float("nan")
        if len(xy_unique) >= 4:
            centre = xy_unique.mean(0)
            r = 8.0
            n_within = int((((xy_unique - centre) ** 2).sum(1) <= r * r).sum())
            local_density = n_within / (np.pi * r * r)

        return {
            "n_columns": int(self.n_cols),
            "n_unique_positions": int(len(xy_unique)),
            "bounds": [x0, x1, y0, y1],
            "nn_distance_mean": float(nn.mean()),
            "nn_distance_min": float(nn.min()),
            "nn_distance_max": float(nn.max()),
            "hull_area": hull_area,
            "density": density,
            "local_density_r8": local_density,
            "ideal_hex_density": float(2 / np.sqrt(3)),
            "per_side": {str(s): int((self.side == s).sum()) for s in np.unique(self.side)},
        }


def build_hex_field(
    hex1: np.ndarray, hex2: np.ndarray, side: np.ndarray
) -> HexField:
    """Build a :class:`HexField` from per-neuron (hex1, hex2, side) arrays.

    Columns are de-duplicated within each side, because the left and right optic
    lobes are separate visual fields that happen to share the same index grid.
    """
    keys = np.stack([hex1, hex2], axis=1).astype(np.float64)

    records: dict[tuple[str, int, int], int] = {}
    col_list: list[tuple[float, float]] = []
    side_list: list[str] = []
    for s in ("L", "R"):
        sel = side == s
        if not sel.any():
            continue
        pairs = np.unique(keys[sel], axis=0)
        for q, r in pairs:
            records[(s, int(q), int(r))] = len(col_list)
            col_list.append((float(q), float(r)))
            side_list.append(s)

    columns = np.asarray(col_list, dtype=np.float32)
    xy = np.stack(hex_to_cartesian(columns[:, 0], columns[:, 1]), axis=1).astype(np.float32)
    return HexField(
        columns=columns,
        xy=xy,
        side=np.asarray(side_list, dtype=object),
        col_index=records,
    )


def column_lookup(field: HexField, hex1: np.ndarray, hex2: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Map each neuron to its column row index (-1 when unknown)."""
    out = np.full(len(hex1), -1, dtype=np.int32)
    for i, (q, r, s) in enumerate(zip(hex1, hex2, side)):
        out[i] = field.col_index.get((str(s), int(q), int(r)), -1)
    return out
