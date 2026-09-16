"""Signed synapses from the MaleCNS neurotransmitter predictions.

The v1 network used ``W_ij = log1p(synapse_count) > 0`` everywhere -- an
all-excitatory recurrence.  With ReLU units that is only marginally stable:
measured activity in the ``core`` circuit grows ~10^4-fold over 32 steps, and
counting worked mainly because the readout tapped the state at t=8 before it blew
up.  Any task needing a longer horizon (the addition task's delay) exposes it.

This module supplies the missing ingredient: a sign per neuron, taken from
``body-neurotransmitters-male-cns-v1.0.feather``.  Coverage is essentially
complete for the circuits used here (33,480/33,480 for ``core``), and 83% of
those neurons carry a definite transmitter:

===============  ==========  ==========================================
transmitter      sign        rationale
===============  ==========  ==========================================
acetylcholine    +1          excitatory in the insect CNS
gaba             -1          inhibitory
glutamate        -1          inhibitory at most insect central synapses
                             (chloride-permeable GluCl receptors)
histamine        -1          photoreceptor transmitter, histamine-gated
                             chloride channel
unclear / other   0 -> fallback
                             dopamine, octopamine, serotonin and
                             ``unclear`` are neuromodulatory or unknown;
                             they take ``unknown_sign`` (default +1)
===============  ==========  ==========================================

Only the *sign* is taken.  Magnitudes stay ``log1p(synapse_count)``, so the
strength information from the connectome is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import paths
from .annotations import Annotations

#: transmitter -> sign
NT_SIGN: dict[str, int] = {
    "acetylcholine": +1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    # neuromodulators / unresolved: handled via ``unknown_sign``
    "dopamine": 0,
    "octopamine": 0,
    "serotonin": 0,
    "unclear": 0,
}

EXCITATORY = tuple(k for k, v in NT_SIGN.items() if v > 0)
INHIBITORY = tuple(k for k, v in NT_SIGN.items() if v < 0)


@dataclass
class SignReport:
    n_neurons: int
    n_with_prediction: int
    n_excitatory: int
    n_inhibitory: int
    n_unknown: int
    unknown_sign: int

    @property
    def fraction_signed(self) -> float:
        return (self.n_excitatory + self.n_inhibitory) / max(self.n_neurons, 1)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["fraction_signed"] = round(self.fraction_signed, 4)
        return d


def load_neuron_signs(
    ann: Annotations,
    *,
    unknown_sign: int = +1,
    use_celltype_fallback: bool = True,
    path=None,
) -> tuple[np.ndarray, SignReport]:
    """Per-neuron sign aligned with ``ann`` (+1 excitatory, -1 inhibitory).

    ``use_celltype_fallback`` resolves neurons whose own ``consensus_nt`` is
    unclear by falling back to the cell-type-level prediction, which is estimated
    from all neurons of that type and is therefore better determined.
    """
    path = path or paths.raw_path("neurotransmitters")
    df = pd.read_feather(path)
    key = df["body"].to_numpy().astype(np.int64)

    def col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.array(["unclear"] * len(df), dtype=object)
        out = np.empty(len(df), dtype=object)
        for i, v in enumerate(df[name].to_numpy(dtype=object)):
            out[i] = "unclear" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        return out

    primary = col("consensus_nt")
    fallback = col("celltype_predicted_nt") if use_celltype_fallback else primary

    sign_lut = np.full(int(key.max()) + 1, 0, dtype=np.int8)
    has = np.zeros(int(key.max()) + 1, dtype=bool)
    resolved = np.where(primary == "unclear", fallback, primary)
    for nt, s in NT_SIGN.items():
        if s == 0:
            continue
        sign_lut[key[resolved == nt]] = s
    has[key] = True

    idx = ann.body_ids
    in_range = (idx >= 0) & (idx < len(sign_lut))
    signs = np.full(ann.n, int(np.sign(unknown_sign)), dtype=np.int8)
    present = np.zeros(ann.n, dtype=bool)
    signs[in_range] = sign_lut[idx[in_range]]
    present[in_range] = has[idx[in_range]]
    # neurons with no prediction at all, or an unresolved transmitter, take the
    # configured default
    unresolved = (signs == 0) | ~present
    signs[unresolved] = int(np.sign(unknown_sign))

    rep = SignReport(
        n_neurons=int(ann.n),
        n_with_prediction=int(present.sum()),
        n_excitatory=int((signs > 0).sum()),
        n_inhibitory=int((signs < 0).sum()),
        n_unknown=int(unresolved.sum()),
        unknown_sign=int(np.sign(unknown_sign)),
    )
    return signs, rep


def edge_signs(pre: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Sign of each edge, taken from its **presynaptic** neuron."""
    return signs[pre].astype(np.float32)
