"""Stage 1 / 3 -- run a single dot-counting experiment.

Examples
--------
Single pilot run (taught fly, real connectome, 20k images)::

    python scripts/06_run_count.py --model M1 --graph real --n-train 20000

Frozen-fly linear probe::

    python scripts/06_run_count.py --model M0 --graph real --n-train 20000

Shuffled control at the same operating point::

    python scripts/06_run_count.py --model M1 --graph shuffled --shuffle-seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.experiments.count import run_count_experiment  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402


def build_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.stage = args.stage
    cfg.task = "count"
    cfg.model = args.model
    cfg.data.circuit = args.circuit
    cfg.data.graph = args.graph
    cfg.train.n_train = args.n_train
    cfg.train.max_epochs = args.max_epochs
    cfg.train.patience = args.patience
    cfg.train.batch_size = args.batch_size
    cfg.train.lr_gain = args.lr_gain
    cfg.train.lr_readout = args.lr_readout
    cfg.train.cosine = not args.no_cosine
    cfg.retina.map_mode = args.map_mode
    cfg.retina.input_types = args.input_types
    cfg.retina.input_eye = args.input_eye
    cfg.model_cfg.readout = args.readout
    cfg.model_cfg.alpha = args.alpha
    cfg.model_cfg.nonlinearity = args.act
    cfg.model_cfg.learn_gains = not args.freeze_weights
    cfg.model_cfg.lambda_gain = args.lambda_gain
    cfg.model_cfg.readout_standardize = args.standardize
    cfg.model_cfg.weight_normalization = args.normalization
    cfg.time.steps = args.steps
    cfg.seeds["model_seed"] = args.model_seed
    cfg.seeds["shuffle_seed"] = args.shuffle_seed
    cfg.seeds["data_seed"] = args.data_seed
    cfg.seeds["split_seed"] = args.split_seed

    # w_scale is circuit specific; use the value calibrated in Stage 0 unless overridden
    if args.w_scale is not None:
        cfg.model_cfg.w_scale = args.w_scale
    else:
        from flynum.pipeline import load_w_scale

        cfg.model_cfg.w_scale = load_w_scale(cfg.data.circuit)
    if args.run_name:
        cfg.run_name = args.run_name
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="1")
    p.add_argument("--model", default="M1", choices=["M0", "M1"])
    p.add_argument("--graph", default="real", choices=["real", "shuffled"])
    p.add_argument("--circuit", default="core", choices=["core", "full", "c3"])
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr-gain", type=float, default=3e-3)
    p.add_argument("--lr-readout", type=float, default=3e-3)
    p.add_argument("--no-cosine", action="store_true",
                   help="keep the learning rate constant instead of annealing")
    p.add_argument("--standardize", action="store_true",
                   help="enable readout feature standardisation. Off by default, "
                        "matching scripts/07 and the reported results: with the "
                        "default learning rate it destabilises training and the "
                        "first epoch diverges.")
    p.add_argument("--normalization", default="global", choices=["global", "row"],
                   help="global: divide by the mean row sum (default, used for all "
                        "reported results). row: divide each neuron by its own row "
                        "sum, needed for circuits with extreme degree heterogeneity "
                        "such as `full`.")
    p.add_argument("--shuffle-rounds", type=float, default=20.0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--act", default="relu", choices=["relu", "tanh"])
    p.add_argument("--lambda-gain", type=float, default=1e-3, dest="lambda_gain")
    p.add_argument("--freeze-weights", action="store_true",
                   help="train only the readout even for M1 (ablation)")
    p.add_argument("--readout", default="vpn",
                   choices=["vpn", "all_except_input", "all", "lc11"])
    p.add_argument("--map-mode", default="isotropic", choices=["isotropic", "stretch"])
    p.add_argument("--input-types", default="lamina",
                   choices=["lamina", "lamina_all", "all_hex"])
    p.add_argument("--input-eye", default="both", choices=["both", "R", "L"])
    p.add_argument("--w-scale", type=float, default=None)
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--split-seed", type=int, default=1235)
    p.add_argument("--run-name", default="")
    args = p.parse_args()

    cfg = build_config(args)
    log = get_logger("run_count")
    log.info("config: %s", json.dumps(cfg.to_dict(), default=str))
    summary = run_count_experiment(cfg, logger=log)
    log.info("test A=%.4f B=%.4f C=%.4f D=%.4f",
             summary["test_acc_a"], summary["test_acc_b"],
             summary["test_acc_c"], summary["test_acc_d"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
