"""Device selection.

A one-line helper, but it has to be a helper rather than an inline expression
because the obvious inline expression is wrong on this stack.  Measured on the
project's environment (torch 2.5.1+cu121, Windows):

>>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
>>> torch.cuda.is_available(), torch.cuda.device_count()
(True, 0)

``is_available()`` reports whether the *driver* sees CUDA, not whether a device
is usable by this process.  Selecting ``"cuda"`` on that basis then hands the code
a device that ``torch.load(..., map_location="cuda")`` refuses to deserialise onto
and whose ``get_device_properties`` raises -- a confusing failure a long way from
its cause.  Gate on ``device_count()`` instead.
"""

from __future__ import annotations


def pick_device() -> str:
    """``"cuda"`` when at least one CUDA device is usable by this process."""
    import torch

    try:
        return "cuda" if torch.cuda.device_count() > 0 else "cpu"
    except Exception:  # a broken CUDA install must not stop a CPU run
        return "cpu"
