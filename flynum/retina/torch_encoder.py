"""GPU-side implementation of the fixed retinal sampling map.

The numpy :class:`~flynum.retina.encoder.RetinaEncoder` defines *what* the map
is; this class precomputes the bilinear sampling indices once so that encoding a
batch of images on the GPU is a handful of gathers.  The mapping is identical --
this is purely a performance path -- and a test asserts the two agree.
"""

from __future__ import annotations

import numpy as np
import torch

from .encoder import RetinaEncoder


class TorchRetina:
    """Bilinear image -> column activations, on the GPU."""

    def __init__(self, encoder: RetinaEncoder, device: torch.device | str = "cpu"):
        self.encoder = encoder
        self.image_size = encoder.image_size
        self.n_columns = encoder.field.n_cols
        self.device = torch.device(device)

        uv = encoder.col_uv.astype(np.float32)
        S = self.image_size
        u = np.clip(uv[:, 0], -1.0, S)
        v = np.clip(uv[:, 1], -1.0, S)
        u0 = np.floor(u).astype(np.int64)
        v0 = np.floor(v).astype(np.int64)
        du = (u - u0).astype(np.float32)
        dv = (v - v0).astype(np.float32)

        def pack(idx: np.ndarray, off: int) -> tuple[np.ndarray, np.ndarray]:
            i = idx + off
            valid = (i >= 0) & (i < S)
            return np.clip(i, 0, S - 1), valid

        u0c, u0v = pack(u0, 0)
        u1c, u1v = pack(u0, 1)
        v0c, v0v = pack(v0, 0)
        v1c, v1v = pack(v0, 1)

        def t(arr, dtype=torch.long):
            return torch.as_tensor(arr, dtype=dtype, device=self.device)

        self.u0, self.u1 = t(u0c), t(u1c)
        self.v0, self.v1 = t(v0c), t(v1c)
        self.m00 = t(u0v & v0v, torch.float32)
        self.m10 = t(u1v & v0v, torch.float32)
        self.m01 = t(u0v & v1v, torch.float32)
        self.m11 = t(u1v & v1v, torch.float32)
        self.du = t(du, torch.float32)
        self.dv = t(dv, torch.float32)
        self.valid_mask = t(encoder.coverage_mask(), torch.float32)

    # ------------------------------------------------------------------ #
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, S, S)`` images -> ``(B, n_columns)`` column activations."""
        if images.shape[1:] != (self.image_size, self.image_size):
            raise ValueError(f"expected (B,{self.image_size},{self.image_size})")
        flat = images.reshape(images.shape[0], -1)
        S = self.image_size

        def gather(u: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            idx = (v * S + u).unsqueeze(0).expand(flat.shape[0], -1)
            return flat.gather(1, idx) * m.unsqueeze(0)

        w00 = self.m00 * (1 - self.du) * (1 - self.dv)
        w10 = self.m10 * self.du * (1 - self.dv)
        w01 = self.m01 * (1 - self.du) * self.dv
        w11 = self.m11 * self.du * self.dv
        out = (
            gather(self.u0, self.v0, w00)
            + gather(self.u1, self.v0, w10)
            + gather(self.u0, self.v1, w01)
            + gather(self.u1, self.v1, w11)
        )
        return out.to(images.dtype)
