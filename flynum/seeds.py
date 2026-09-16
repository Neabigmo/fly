"""Seed handling.

Four *independent* seeds are used so that comparisons stay paired:

``data_seed``
    Controls stimulus generation.  Fixed globally: every model in the study
    therefore sees byte-identical images.
``split_seed``
    Controls the train/val/test split and the nested subsets used for the
    learning curve.  Shared between real and shuffled graphs so that a given
    (N_train, seed) pair uses the *same* training images.
``model_seed``
    Controls parameter initialisation and batch order.
``shuffle_seed``
    Controls the degree/weight-preserving edge swap that produces a control
    connectome.  Varying it gives independent control graph realisations.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class Seeds:
    data_seed: int = 1234
    split_seed: int = 1235
    model_seed: int = 0
    shuffle_seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def tag(self) -> str:
        return f"m{self.model_seed}_sh{self.shuffle_seed}"


def set_seeds(seed: int, deterministic: bool = False) -> None:
    """Seed python / numpy / torch (CPU+GPU)."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is always present in practice
        pass


def rng(seed: int) -> np.random.Generator:
    """Independent NumPy generator (use this rather than the global RNG)."""
    return np.random.default_rng(seed)
