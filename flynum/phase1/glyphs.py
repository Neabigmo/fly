"""Operator glyphs, drawn through exactly the same path as the dots.

The two-rule task (A2) and the two-step task (A3) need the network to see *which*
operation to apply, not to infer it from the operands.  A ``+`` or a ``-`` therefore has
to occupy the same window of the sequence that a blank delay occupies in the plain
addition task, so that the temporal structure is unchanged and only the middle window's
content differs.

The glyph is built as a chain of small discs and rendered by
:func:`flynum.stimuli.dots.render_dots` -- the same supersampled, area-averaged renderer
the dots use.  Two consequences matter.  Overlapping discs merge into a continuous
stroke, and the ink of a glyph is directly comparable to the ink of a dot group, so
"an operator is present" is not a much louder signal than "here are three dots".  The
stroke radius and arm length are set so that the glyph covers roughly the same retinal
extent as a dot cloud.

Whether the fixed retinotopic map preserves the difference between the two glyphs is not
assumed: ``scripts/25_phase1_preflight.py`` decodes ``+`` against ``-`` from the column
activations before any long run is allowed to start.
"""

from __future__ import annotations

import numpy as np

from ..stimuli.dots import render_dots

#: Symbols, in the fixed order used as the operand encoding in task items.
OPS: tuple[str, ...] = ("+", "-")

#: Half-length of the glyph arms and the disc radius used to draw them, in pixels of the
#: 32x32 canvas.  A dot cloud lives inside a ring of radius 9.5, so the glyph is
#: deliberately the same order of size; the stroke is about as wide as the smallest dot.
HALF_ARM = 5.5
STROKE_RADIUS = 1.6
SPACING = 0.5


def glyph_centres(symbol: str, *, half_arm: float = HALF_ARM,
                  spacing: float = SPACING, size: int = 32) -> np.ndarray:
    """Lattice points along the strokes of ``symbol``, centred on the canvas."""
    if symbol not in OPS:
        raise ValueError(f"unknown operator glyph {symbol!r}")
    mid = size / 2.0 - 0.5
    n = int(round(2 * half_arm / spacing)) + 1
    line = np.linspace(-half_arm, half_arm, n, dtype=np.float32)
    parts = [np.stack([mid + line, np.full(n, mid, np.float32)], axis=1)]   # horizontal
    if symbol == "+":
        parts.append(np.stack([np.full(n, mid, np.float32), mid + line], axis=1))
    return np.concatenate(parts, axis=0)


def render_glyph(symbol: str, sc, rng: np.random.Generator | None = None, *,
                 half_arm: float = HALF_ARM, size: int = 32) -> np.ndarray:
    """``(S, S)`` float32 image of an operator, rendered like a dot stimulus."""
    centres = glyph_centres(symbol, half_arm=half_arm, size=size)
    radii = np.full(len(centres), STROKE_RADIUS, dtype=np.float32)
    return render_dots(
        size, centres, radii,
        supersample=getattr(sc, "supersample", 4),
        noise_sigma=getattr(sc, "noise_sigma", 0.0),
        rng=rng,
    )


def render_blank(size: int) -> np.ndarray:
    """A blank window.  Deliberately exactly zero: no noise, so 'nothing here' is a
    well-defined condition and the comparison between a glyph window and a delay window
    is not confounded by the glyph window carrying extra baseline noise."""
    return np.zeros((size, size), dtype=np.float32)


def ink(symbol: str, sc, *, half_arm: float = HALF_ARM) -> float:
    """Total ink of a noiseless glyph, for comparison with a dot group's ink."""
    size = int(getattr(sc, "image_size", 32))
    centres = glyph_centres(symbol, half_arm=half_arm, size=size)
    radii = np.full(len(centres), STROKE_RADIUS, dtype=np.float32)
    img = render_dots(size, centres, radii,
                      supersample=getattr(sc, "supersample", 4), noise_sigma=0.0)
    return float(img.sum())
