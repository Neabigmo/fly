"""Fixed (never trained) image -> neuron injection map.

Design
------
The stimulus is a ``32 x 32`` image.  Each hexagonal column of the visual field
is assigned a position in image coordinates; the image is bilinearly sampled at
that position and the resulting scalar is delivered to every *input neuron* of
that column (default: the lamina cell types ``L1, L2, L3, L5, T1``, i.e. the
cells that receive direct photoreceptor input).

Two mapping modes are available:

``isotropic`` (default)
    One image pixel spans exactly one lattice unit, so a round dot stays round
    on the hexagonal lattice.  A 32x32 window centred on the field centroid is
    used; columns outside the window receive zero input.  This is the
    physically honest choice because the fly eye samples the world with
    roughly uniform angular spacing.

``stretch``
    The image is stretched linearly over the whole field bounding box so that
    every column receives drive.  The anisotropy is a *constant* linear warp
    (about 1.48:1), so it scales all areas by the same factor and therefore
    cannot create an area shortcut.

The resulting injection matrix ``B_in`` is a sparse boolean matrix of shape
``(n_neurons, n_columns)`` and is frozen into the model as a buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hexmap import HexField


@dataclass
class EncoderQC:
    n_columns: int
    n_columns_with_input: int
    n_driven_columns: int
    n_input_neurons: int
    map_mode: str
    input_types: str
    input_eye: str
    driven_fraction: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class RetinaEncoder:
    """Maps a 32x32 image onto per-neuron input currents (fixed, not trained)."""

    def __init__(
        self,
        field: HexField,
        neuron_column: np.ndarray,
        input_neuron_mask: np.ndarray,
        *,
        image_size: int = 32,
        map_mode: str = "isotropic",
        input_types: str = "lamina",
        input_eye: str = "both",
    ):
        self.field = field
        self.image_size = int(image_size)
        self.map_mode = map_mode
        self.input_types = input_types
        self.input_eye = input_eye

        n_neurons = len(input_neuron_mask)
        #: image-space (u, v) of every column, in pixel units
        self.col_uv = self._column_uv()
        self.neuron_column = neuron_column.astype(np.int32)

        # injection matrix: (n_neurons, n_columns)
        rows = np.nonzero(input_neuron_mask & (neuron_column >= 0))[0]
        cols = neuron_column[rows]
        self.input_rows = rows.astype(np.int32)
        self.input_cols = cols.astype(np.int32)
        self.n_input_neurons = int(len(rows))

        driven = np.unique(cols) if len(cols) else np.array([], dtype=np.int32)
        self.columns_with_input = driven.astype(np.int32)
        #: columns that both contain input neurons *and* fall inside the image
        #: window -- these are the ones the stimulus can actually reach
        covered = np.nonzero(self.coverage_mask())[0]
        image_driven = np.intersect1d(self.columns_with_input, covered)
        self.driven_columns = image_driven.astype(np.int32)
        self.qc = EncoderQC(
            n_columns=field.n_cols,
            n_columns_with_input=int(len(self.columns_with_input)),
            n_driven_columns=int(len(image_driven)),
            n_input_neurons=self.n_input_neurons,
            map_mode=map_mode,
            input_types=input_types,
            input_eye=input_eye,
            driven_fraction=round(len(image_driven) / max(field.n_cols, 1), 4),
        )

    # ------------------------------------------------------------------ #
    def _column_uv(self) -> np.ndarray:
        """Image coordinates (u, v) in pixels for every column.

        Orientation is part of the design contract: the **top-left of the image
        must drive the top-left of the visual field**.  Image row 0 is the top
        (``imshow`` default) while the field's ``y`` grows upwards, so ``v`` is
        built by *subtracting* the field's y offset.  Getting this backwards
        merely mirrors the stimulus, which is harmless for counting, but it
        would contradict the documented mapping and mislead anyone reading the
        encoder figure.
        """
        xy = self.field.xy.astype(np.float64)
        S = self.image_size
        if self.map_mode == "stretch":
            x0, x1, y0, y1 = self.field.bounds
            u = (xy[:, 0] - x0) / max(x1 - x0, 1e-9) * (S - 1)
            v = (y1 - xy[:, 1]) / max(y1 - y0, 1e-9) * (S - 1)
            return np.stack([u, v], axis=1)
        if self.map_mode == "isotropic":
            cx, cy = self.field.centroid
            u = xy[:, 0] - cx + (S - 1) / 2.0
            v = -(xy[:, 1] - cy) + (S - 1) / 2.0
            return np.stack([u, v], axis=1)
        raise ValueError(f"unknown map_mode: {self.map_mode!r}")

    # ------------------------------------------------------------------ #
    def sample(self, image: np.ndarray) -> np.ndarray:
        """Bilinearly sample one image at every column -> ``(n_columns,)``.

        ``image`` is ``(32, 32)``; positions outside the image contribute zero.
        """
        image = np.asarray(image, dtype=np.float32)
        if image.shape != (self.image_size, self.image_size):
            raise ValueError(
                f"expected image {self.image_size}x{self.image_size}, got {image.shape}"
            )
        u = self.col_uv[:, 0]
        v = self.col_uv[:, 1]
        S = self.image_size
        u = np.clip(u, -1.0, S)
        v = np.clip(v, -1.0, S)
        u0 = np.floor(u).astype(np.int32)
        v0 = np.floor(v).astype(np.int32)
        du = (u - u0).astype(np.float32)
        dv = (v - v0).astype(np.float32)

        def at(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
            ok = (uu >= 0) & (uu < S) & (vv >= 0) & (vv < S)
            out = np.zeros_like(uu, dtype=np.float32)
            out[ok] = image[vv[ok], uu[ok]]
            return out

        return (
            at(u0, v0) * (1 - du) * (1 - dv)
            + at(u0 + 1, v0) * du * (1 - dv)
            + at(u0, v0 + 1) * (1 - du) * dv
            + at(u0 + 1, v0 + 1) * du * dv
        ).astype(np.float32)

    # ------------------------------------------------------------------ #
    def to_matrix(self, column_values: np.ndarray) -> np.ndarray:
        """Scatter column values onto the input neurons -> ``(n_neurons,)``."""
        out = np.zeros(len(self.neuron_column), dtype=np.float32)
        out[self.input_rows] = column_values[self.input_cols]
        return out

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Full image -> per-neuron input vector."""
        return self.to_matrix(self.sample(image))

    # ------------------------------------------------------------------ #
    def coverage_mask(self) -> np.ndarray:
        """Boolean over columns: which ones receive non-zero image support."""
        u = self.col_uv[:, 0]
        v = self.col_uv[:, 1]
        S = self.image_size
        return (u >= -0.5) & (u <= S - 0.5) & (v >= -0.5) & (v <= S - 0.5)


def select_input_neurons(
    cell_type: np.ndarray,
    soma_side: np.ndarray,
    column_of_neuron: np.ndarray,
    *,
    input_types: str = "lamina",
    input_eye: str = "both",
) -> np.ndarray:
    """Boolean mask of neurons that receive the image injection."""
    from ..data.annotations import INPUT_TYPE_SETS

    types = INPUT_TYPE_SETS[input_types]
    mask = np.isin(cell_type, list(types)) & (column_of_neuron >= 0)
    if input_eye != "both":
        mask &= soma_side == input_eye
    return mask
