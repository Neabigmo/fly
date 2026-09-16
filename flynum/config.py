"""Configuration tree.

A single nested dataclass describes a complete experiment.  ``config.json`` is
written verbatim into every run directory, and ``config_hash`` gives a short
stable digest so that runs sharing a configuration can be grouped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any


# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    #: which circuit to build: "core" (33.5k neurons) or "full" (106.9k neurons)
    circuit: str = "core"
    #: which connectome realisation: "real" or "shuffled"
    graph: str = "real"
    #: if graph == "shuffled", swap repetitions as a multiple of edge count
    shuffle_rounds_per_edge: float = 20.0
    #: drop edges whose weight (synapse count) is below this
    min_weight: int = 1


@dataclass
class RetinaConfig:
    #: "isotropic" (1 px = 1 lattice unit, centred) or "stretch" (full field)
    map_mode: str = "isotropic"
    #: cell types receiving a direct image injection
    input_types: str = "lamina"  # lamina | lamina_all | all_hex
    #: which optic lobe receives the image
    input_eye: str = "both"  # both | R | L
    image_size: int = 32


@dataclass
class StimulusConfig:
    #: canvas size; must match retina.image_size (asserted by the pipeline)
    image_size: int = 32
    n_min: int = 1
    n_max: int = 5
    #: Radius range for the natural condition.  Two constraints must hold at
    #: once, and together they pin the range down tightly:
    #:
    #: 1. **Area control is possible.**  Holding total ink constant across
    #:    N in [1, 5] requires r(1)/r(5) = sqrt(5) = 2.236, so the range must
    #:    span at least 2.236x.  Otherwise conditions B/C have to use dot sizes
    #:    the network never saw, and the "numerosity" test silently becomes a
    #:    size-generalisation test.
    #: 2. **Five dots must fit.**  Five discs of radius r fit inside the canvas
    #:    only if r + min_gap/2 <= 15 / (1 + csc(pi/5)) = 5.55, so r <= ~4.8;
    #:    with a safety margin the practical bound is ~3.5.
    #:
    #: U[1.3, 3.4] spans 2.62x, satisfying (1) with room to spare, and keeps
    #: five dots comfortably packable, satisfying (2).
    radius_min: float = 1.3
    radius_max: float = 3.4
    margin: float = 1.0
    #: Minimum surface-to-surface gap between dots, in pixels.  Must exceed the
    #: 1-pixel footprint of the box downsampling, otherwise the anti-aliasing
    #: bridges two nearby dots into a single blob and the "number of objects"
    #: stops matching the label.  1.5 px leaves a comfortable margin.
    min_gap: float = 1.5
    noise_sigma: float = 0.02
    supersample: int = 4
    #: Area-controlled ink target for conditions B/C/D.  Gives r(1) = 3.20 and
    #: r(5) = 1.43, both strictly inside [radius_min, radius_max].
    area_target: float = 32.2
    #: ring radius for the envelope-controlled condition C
    ring_radius: float = 9.5
    #: addition task
    a_min: int = 1
    a_max: int = 4
    holdout_pairs: tuple[tuple[int, int], ...] = ((2, 3), (3, 2))
    #: images generated per ordered pair for the addition task.  600 gives
    #: 14 * 600 = 8400 training items, comparable to the counting budget.
    add_reps_per_pair: int = 600


@dataclass
class ModelConfig:
    alpha: float = 0.2
    nonlinearity: str = "relu"  # relu | tanh
    #: global scale of the initial weight matrix (calibrated in Stage 0)
    w_scale: float = 1.0
    #: trainable per-edge gains are parameterised as g = 1 + delta
    learn_gains: bool = True
    #: weight of the penalty lambda * mean(delta^2)
    lambda_gain: float = 1e-3
    #: population the readout reads from
    readout: str = "vpn"  # vpn | all_except_input | all
    #: average the last k timesteps for the readout
    readout_window: int = 2
    #: Standardise each readout feature with statistics measured once from the
    #: (untrained) network on the training set.  The recurrent state is heavy
    #: tailed -- a few high-degree neurons dominate -- and without this the
    #: linear readout is ill-conditioned: the frozen probe measured 0.53 raw vs
    #: 0.60 standardised.  Statistics are frozen at initialisation so the
    #: comparison between M0 and M1 stays honest.
    readout_standardize: bool = True
    #: number of training images used to estimate the feature statistics
    readout_stats_samples: int = 2000
    #: How the measured synapse counts are turned into weights.
    #: "global" divides by the mean post-synaptic row sum (preserves the fact
    #: that high-degree neurons are more strongly driven; used for all reported
    #: results).  "row" divides each neuron by its own row sum, which is required
    #: by circuits whose degree distribution is very heterogeneous -- the `full`
    #: circuit has a max/mean row sum of 129 and explodes within one epoch under
    #: global normalisation, whatever the learning rate.
    weight_normalization: str = "global"  # global | row


@dataclass
class TimeConfig:
    #: counting task: number of recurrent steps
    steps: int = 8
    #: addition task: steps for epoch A, gap, epoch B
    steps_a: int = 5
    steps_gap: int = 4
    steps_b: int = 5


@dataclass
class TrainConfig:
    n_train: int = 20000
    n_val: int = 2500
    n_test: int = 5000
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 10
    lr_gain: float = 3e-3
    lr_readout: float = 3e-3
    wd_readout: float = 1e-4
    grad_clip: float = 1.0
    cosine: bool = True


@dataclass
class ExperimentConfig:
    stage: str = "1"
    task: str = "count"  # count | add
    model: str = "M1"  # M0 (frozen probe) | M1 (taught)
    run_name: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    retina: RetinaConfig = field(default_factory=RetinaConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seeds: dict[str, int] = field(
        default_factory=lambda: {
            "data_seed": 1234,
            "split_seed": 1235,
            "model_seed": 0,
            "shuffle_seed": 0,
        }
    )

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    def stimulus_fingerprint(self) -> str:
        """Digest of everything that determines *what the network sees*.

        Runs must never be pooled across stimulus versions.  The stimulus design
        changed materially during development (the trained radius range had to
        be widened so the area-controlled conditions stayed in-distribution),
        and a run trained on the old images is not comparable with one trained on
        the new ones -- pooling them silently mixes two experiments.
        """
        blob = json.dumps(
            {"stimulus": asdict(self.stimulus), "retina": asdict(self.retina)},
            sort_keys=True,
            default=str,
        ).encode()
        return hashlib.sha1(blob).hexdigest()[:8]

    def run_id(self) -> str:
        d, t, m = self.data, self.train, self.seeds
        base = (
            f"s{self.stage}_{self.task}_{self.model}_{d.circuit}_{d.graph}"
            f"_N{t.n_train}_m{m['model_seed']}_sh{m['shuffle_seed']}"
        )
        return f"{base}_{self.config_hash()}" if self.run_name == "" else self.run_name

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        """Build from a (possibly partial) nested dict; unknown keys error."""
        kwargs: dict[str, Any] = {}
        subs = {
            "data": DataConfig,
            "retina": RetinaConfig,
            "stimulus": StimulusConfig,
            "model_cfg": ModelConfig,
            "time": TimeConfig,
            "train": TrainConfig,
        }
        valid = {f.name for f in fields(cls)}
        for key, value in raw.items():
            if key not in valid:
                raise KeyError(f"unknown config key: {key!r}")
            if key in subs and isinstance(value, dict):
                sub_valid = {f.name for f in fields(subs[key])}
                bad = set(value) - sub_valid
                if bad:
                    raise KeyError(f"unknown key(s) in {key}: {sorted(bad)}")
                if "holdout_pairs" in value:
                    value = dict(value)
                    value["holdout_pairs"] = tuple(tuple(p) for p in value["holdout_pairs"])
                kwargs[key] = subs[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)
