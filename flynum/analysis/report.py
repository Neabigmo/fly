"""Aggregate every run into the study's headline figures and a markdown report.

Everything is read back from disk (``runs/*/full_summary.json`` plus the
per-run ``predictions.npz``), so the report can be regenerated at any time
without retraining and always reflects what actually ran.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..train.metrics import bootstrap_ci, paired_gap
from .figures import (
    plot_addition_matrix,
    plot_condition_bars,
    plot_confusion,
    plot_curriculum,
    plot_epoch_curves,
    plot_fixed_updates,
    plot_gain_distribution,
    plot_gap_over_epochs,
    plot_learning_curve,
    plot_lesion,
    plot_propagation,
    plot_retina_map,
    plot_shuffle_mixing,
    plot_signed_dynamics,
    plot_system_diagram,
)


# --------------------------------------------------------------------------- #
def _holdout_key(s: dict) -> str:
    """Which ordered pairs a curriculum run held out, as a stable label."""
    hp = s.get("holdout_pairs") or []
    return "-".join(f"{a}{b}" for a, b in hp) if hp else "?"


def load_summaries() -> list[dict]:
    """Every completed run, annotated with its stimulus fingerprint.

    The fingerprint comes from each run's own ``config.json`` so that even runs
    produced before the field existed are classified correctly.
    """
    out = []
    for path in sorted(paths.RUNS.glob("*/full_summary.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cfg_path = path.parent / "config.json"
        if cfg_path.exists():
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                from ..config import ExperimentConfig

                # strict=False: a run's config.json carries annotation keys alongside
                # the configuration (``curriculum``, ``sign_report``, ...).  With the
                # strict default this raised, the fingerprint became "unknown", and
                # filter_current_stimuli then dropped every curriculum run -- so all
                # twelve of them were missing from the report while the block had
                # reported success.
                cfg_obj = ExperimentConfig.from_dict(
                    {k: v for k, v in raw.items() if not k.startswith("_")},
                    strict=False,
                )
                rec["stimulus_fingerprint"] = cfg_obj.stimulus_fingerprint()
                # record how the weights were built, so circuits that need a
                # different normalisation are never pooled by accident
                rec["weight_normalization"] = cfg_obj.model_cfg.weight_normalization
                # ``graph`` was missing from older summaries' info block, so backfill
                # it (and the other identity fields) from the config rather than
                # letting a null quietly drop the run out of every graph-keyed table.
                for field, value in (
                    ("graph", cfg_obj.data.graph),
                    ("circuit", cfg_obj.data.circuit),
                    ("task", cfg_obj.task),
                    ("model", cfg_obj.model),
                    ("n_train", int(cfg_obj.train.n_train)),
                    ("model_seed", cfg_obj.seeds["model_seed"]),
                    ("shuffle_seed", cfg_obj.seeds["shuffle_seed"]),
                ):
                    if rec.get(field) is None:
                        rec[field] = value
                # and how it was trained: the fixed-update control differs from the
                # epoch-budget curve only in these fields, so they must travel with
                # every summary or the two blocks would be silently pooled
                rec["cfg_max_steps"] = int(cfg_obj.train.max_steps)
                rec["cfg_eval_every"] = int(cfg_obj.train.eval_every)
                rec["cfg_steps"] = int(cfg_obj.time.steps)
                rec["cfg_alpha"] = float(cfg_obj.model_cfg.alpha)
                rec["cfg_w_scale"] = float(cfg_obj.model_cfg.w_scale)
                rec["cfg_signed"] = bool(cfg_obj.model_cfg.signed_synapses)
                rec["cfg_standardize"] = bool(cfg_obj.model_cfg.readout_standardize)
                rec["cfg_batch_size"] = int(cfg_obj.train.batch_size)
            except Exception:
                rec["stimulus_fingerprint"] = "unknown"
                rec["weight_normalization"] = "unknown"
        else:
            rec["stimulus_fingerprint"] = "unknown"
            rec["weight_normalization"] = "unknown"
        out.append(rec)
    return out


def filter_current_stimuli(
    summaries: list[dict], *, keep: str | None = None, logger=None
) -> tuple[list[dict], str]:
    """Keep only runs that used the same stimulus definition.

    Returns the filtered list and the fingerprint kept.  By default the most
    common fingerprint wins, which is the current design once the grid has run.
    """
    from ..config import ExperimentConfig

    target = keep or ExperimentConfig().stimulus_fingerprint()
    kept = [s for s in summaries if s.get("stimulus_fingerprint") == target]
    if len(kept) < len(summaries) and logger:
        dropped = len(summaries) - len(kept)
        logger.warning(
            "dropped %d run(s) trained on a different stimulus definition "
            "(fingerprint %s); keeping %d runs at %s",
            dropped, target, len(kept), target,
        )
    return kept, target


def summaries_to_frame(summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        rows.append(
            {
                "run_id": s.get("run_id"),
                "stage": s.get("stage"),
                "task": s.get("task"),
                "model": s.get("model"),
                "circuit": s.get("circuit"),
                "graph": s.get("graph"),
                "n_train": s.get("n_train"),
                "model_seed": s.get("model_seed"),
                "shuffle_seed": s.get("shuffle_seed"),
                "n_params": s.get("n_params"),
                "best_val_acc": s.get("best_val_acc"),
                "test_acc_a": s.get("test_acc_a"),
                "test_acc_b": s.get("test_acc_b"),
                "test_acc_c": s.get("test_acc_c"),
                "test_acc_d": s.get("test_acc_d"),
                "addition_train_accuracy": s.get("addition_train_accuracy"),
                "addition_test_accuracy": s.get("addition_test_accuracy"),
                "unseen_pair_accuracy": s.get("unseen_pair_accuracy"),
                "wall_seconds_total": s.get("wall_seconds_total"),
                "max_steps": s.get("cfg_max_steps"),
                "steps": s.get("cfg_steps"),
                "alpha": s.get("cfg_alpha"),
                "w_scale": s.get("cfg_w_scale"),
                "signed": s.get("cfg_signed"),
                "standardize": s.get("cfg_standardize"),
                "epochs_run": (s.get("train_info") or {}).get("epochs_run"),
                "optimizer_steps": (s.get("train_info") or {}).get("optimizer_steps"),
                "batch_size": s.get("cfg_batch_size"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
@dataclass
class Curve:
    graph: str
    by_n: dict[str, dict]

    def as_dict(self) -> dict:
        return {self.graph: self.by_n}


def build_learning_curves(
    df: pd.DataFrame, metric: str = "test_acc_a", model: str = "M1",
    circuit: str = "core",
) -> dict[str, Curve]:
    """Group runs into one curve per graph, for a given model/metric/circuit."""
    sel = df[(df["task"] == "count") & (df["model"] == model) & (df["circuit"] == circuit)]
    sel = sel.dropna(subset=[metric])
    out: dict[str, Curve] = {}
    for graph, sub in sel.groupby("graph"):
        by_n: dict[str, dict] = {}
        for n, g in sub.groupby("n_train"):
            vals = g[metric].astype(float).tolist()
            by_n[str(int(n))] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": vals,
                "seeds": g["model_seed"].astype(int).tolist(),
                "run_ids": g["run_id"].tolist(),
            }
        out[graph] = Curve(graph, by_n)
    return out


def paired_gaps(df: pd.DataFrame, metric_correct: str = "A") -> list[dict]:
    """Item-paired real-minus-shuffled gaps at every matched operating point."""
    out = []
    count = df[(df["task"] == "count")].copy()
    key = ["model", "circuit", "n_train", "model_seed"]
    real = count[count["graph"] == "real"].set_index(key)
    shuf = count[count["graph"] == "shuffled"].set_index(key)
    for k in real.index.intersection(shuf.index):
        rr, ss = real.loc[k], shuf.loc[k]
        rr_id = rr["run_id"] if isinstance(rr["run_id"], str) else rr["run_id"].iloc[0]
        ss_id = ss["run_id"] if isinstance(ss["run_id"], str) else ss["run_id"].iloc[0]
        pr = paths.run_dir(rr_id) / "predictions.npz"
        ps = paths.run_dir(ss_id) / "predictions.npz"
        if not (pr.exists() and ps.exists()):
            continue
        with np.load(pr) as zr, np.load(ps) as zs:
            y = zr[f"{metric_correct}_label"]
            cr = (zr[f"{metric_correct}_pred"] == y).astype(np.float64)
            cs = (zs[f"{metric_correct}_pred"] == y).astype(np.float64)
        gap = paired_gap(cr, cs)
        gap.update(
            {
                "model": k[0], "circuit": k[1], "n_train": int(k[2]), "model_seed": int(k[3]),
                "real_run": rr_id, "shuffled_run": ss_id, "condition": metric_correct,
            }
        )
        out.append(gap)
    return out


def load_epoch_curve(run_id: str) -> list[float]:
    """Per-epoch validation accuracy from a run's metrics.jsonl."""
    path = paths.run_dir(run_id) / "metrics.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" in rec and "val_acc" in rec:
            out.append(float(rec["val_acc"]))
    return out


def epoch_series(
    df: pd.DataFrame, model: str, circuit: str, n_train: int
) -> dict[str, dict]:
    """Mean +/- std validation curve per graph, for one operating point."""
    sel = df[
        (df["task"] == "count") & (df["model"] == model)
        & (df["circuit"] == circuit) & (df["n_train"] == n_train)
    ]
    out: dict[str, dict] = {}
    for graph, sub in sel.groupby("graph"):
        curves = [load_epoch_curve(r) for r in sub["run_id"]]
        curves = [c for c in curves if c]
        if not curves:
            continue
        L = min(len(c) for c in curves)
        arr = np.array([c[:L] for c in curves], dtype=float)
        out[graph] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": (arr.std(axis=0, ddof=1).tolist() if len(arr) > 1
                    else [0.0] * L),
            "runs": len(arr),
        }
    return out


def gap_series(
    df: pd.DataFrame, model: str, circuit: str
) -> dict[str, list[float]]:
    """Real-minus-shuffled validation gap over epochs, per training-set size."""
    out: dict[str, list[float]] = {}
    sel = df[
        (df["task"] == "count") & (df["model"] == model) & (df["circuit"] == circuit)
    ]
    for n in sorted(sel["n_train"].dropna().unique()):
        series = epoch_series(df, model, circuit, int(n))
        if "real" not in series or "shuffled" not in series:
            continue
        r = np.array(series["real"]["mean"])
        s = np.array(series["shuffled"]["mean"])
        L = min(len(r), len(s))
        out[f"N={int(n)}"] = (r[:L] - s[:L]).tolist()
    return out


# --------------------------------------------------------------------------- #
def make_all_figures(
    summaries: list[dict], figdir: Path, *, retina_bundle: Path | None = None
) -> dict[str, Path]:
    """Produce every figure the report references."""
    figdir.mkdir(parents=True, exist_ok=True)
    made: dict[str, Path] = {}
    df = summaries_to_frame(summaries)

    # ---- Figure 1: system diagram ------------------------------------- #
    circuit_info: dict = {}
    cr_path = paths.DATA_PROCESSED / "circuit_report.json"
    if cr_path.exists():
        cr = json.loads(cr_path.read_text(encoding="utf-8"))
        circuit_info = cr.get("core") or (next(iter(cr.values())) if cr else {})
    baselines_path = paths.DATA_PROCESSED / "baselines.json"
    baselines = (
        json.loads(baselines_path.read_text(encoding="utf-8"))
        if baselines_path.exists()
        else None
    )
    made["system"] = plot_system_diagram(
        figdir / "fig1_system.png", circuit_info, baselines
    )

    # ---- Figure S1: retina map ---------------------------------------- #
    if retina_bundle is not None and retina_bundle.exists():
        with np.load(retina_bundle, allow_pickle=True) as z:
            from ..retina.encoder import RetinaEncoder  # noqa: F401

            made["retina_map"] = plot_retina_map(
                _Rehydrated(z["field_xy"], z["field_side"]),
                _RehydratedEncoder(z),
                figdir / "figS1_retina_map.png",
            )

    # ---- Figure 2: confusion matrices --------------------------------- #
    for model in ("M0", "M1"):
        conf = {}
        for cond in ("A", "B", "C"):
            runs = [
                s for s in summaries
                if s.get("task") == "count" and s.get("model") == model
                and s.get("n_train") == 20000 and s.get("graph") == "real"
            ]
            if runs:
                conf[cond] = np.array(runs[0]["per_mode"][cond]["confusion"])
        if conf:
            made[f"confusion_{model}"] = plot_confusion(
                conf, 1, figdir / f"fig2_confusion_{model}.png",
                title=f"Numerosity confusion, {model} (real connectome, N=20k)",
            )

    # ---- Figure 3: learning curves ------------------------------------ #
    for model in ("M0", "M1"):
        curves = build_learning_curves(df, "test_acc_a", model=model)
        if curves:
            made[f"curve_A_{model}"] = plot_learning_curve(
                {g: c.by_n for g, c in curves.items()},
                figdir / f"fig3_learning_curve_{model}_A.png",
                ylabel="count accuracy (condition A)",
                title=f"Sample efficiency {model}: real vs shuffled (natural stimuli)",
            )
        curves_b = build_learning_curves(df, "test_acc_b", model=model)
        if curves_b:
            made[f"curve_B_{model}"] = plot_learning_curve(
                {g: c.by_n for g, c in curves_b.items()},
                figdir / f"fig3_learning_curve_{model}_B.png",
                ylabel="count accuracy (condition B)",
                title=f"Sample efficiency {model}: area-controlled stimuli",
            )

    # ---- Figure 3b: learning speed (accuracy vs epoch) ---------------- #
    for model in ("M0", "M1"):
        sizes = sorted(
            df[
                (df["task"] == "count") & (df["model"] == model)
                & (df.get("circuit") == "core")
            ]["n_train"].dropna().unique()
        )
        if not sizes:
            continue
        best_n = int(max(sizes))
        series = epoch_series(df, model, "core", best_n)
        if len(series) >= 2:
            made[f"speed_{model}"] = plot_epoch_curves(
                series, figdir / f"fig3b_learning_speed_{model}.png",
                title=f"Learning speed {model} (N={best_n}): real vs shuffled",
                chance=0.2,
            )
    gaps = gap_series(df, "M1", "core")
    if gaps:
        made["gap_epochs"] = plot_gap_over_epochs(
            gaps, figdir / "fig3c_gap_over_epochs.png"
        )

    # ---- Figure 4: addition matrix ------------------------------------ #
    for graph in ("real", "shuffled"):
        runs = [
            s for s in summaries
            if s.get("task") == "add" and s.get("graph") == graph and s.get("model") == "M1"
        ]
        if runs:
            made[f"addition_{graph}"] = plot_addition_matrix(
                runs[0]["matrix_test"], [tuple(p) for p in runs[0]["holdout_pairs"]],
                figdir / f"fig4_addition_{graph}.png", 1, 4,
                title=f"Addition ({graph} connectome)",
            )

    # ---- Figure S2: shuffle mixing ------------------------------------ #
    shuf = [s for s in summaries if isinstance(s.get("shuffle_stats"), dict)
            and s["shuffle_stats"].get("overlap_curve")]
    if shuf:
        made["shuffle_mixing"] = plot_shuffle_mixing(
            shuf[0]["shuffle_stats"], figdir / "figS2_shuffle_mixing.png"
        )

    # ---- Figure S3: calibration / propagation ------------------------- #
    cal_path = paths.DATA_PROCESSED / "calibration.json"
    if cal_path.exists():
        made["propagation"] = plot_propagation(
            json.loads(cal_path.read_text(encoding="utf-8")),
            figdir / "figS3_calibration.png",
        )

    # ---- Figure 6: the equal-update control ---------------------------- #
    # Left panel is the original curve (max_steps == 0, 60 epochs each); right
    # panel is the same conditions spending an identical 18,780 updates.
    # Numbering starts at 6 because fig5 is the temporal-decoding figure.
    fxu = df[
        (df["task"] == "count") & (df["model"] == "M1") & (df["circuit"] == "core")
        & (df["standardize"] == False)  # noqa: E712 - pandas needs ==
        & (df["alpha"] == 0.2) & (df["steps"] == 8) & (df["w_scale"] == 1.0)
        & (df["signed"] == False)  # noqa: E712
    ]
    base_curves = build_learning_curves(fxu[fxu["max_steps"] == 0], "test_acc_a")
    fxu_curves = build_learning_curves(fxu[fxu["max_steps"] > 0], "test_acc_a")
    if base_curves and fxu_curves:
        made["fixed_updates"] = plot_fixed_updates(
            {g: c.by_n for g, c in base_curves.items()},
            {g: c.by_n for g, c in fxu_curves.items()},
            figdir / "fig6_fixed_updates.png",
            budget=int(fxu[fxu["max_steps"] > 0]["max_steps"].max()),
        )

    # ---- Figure 7: signed synapses (Fly-v2) ---------------------------- #
    dyn_path = paths.DATA_PROCESSED / "dynamics_signed_core.json"
    if dyn_path.exists():
        recs = json.loads(dyn_path.read_text(encoding="utf-8"))["records"]
        keep = {k: v for k, v in recs.items()
                if k == "unsigned w=1" or k.startswith("signed w=0.5")}
        if keep:
            made["signed_dynamics"] = plot_signed_dynamics(
                keep, figdir / "fig7_signed_dynamics.png"
            )
    for label, flag in (("unsigned", False), ("signed", True)):
        sel = df[
            (df["task"] == "count") & (df["model"] == "M1") & (df["circuit"] == "core")
            & (df["n_train"] == 20000) & (df["signed"] == flag)
            & (df["max_steps"] == 0) & (df["standardize"] == False)  # noqa: E712
        ]
        agg: dict[str, dict] = {}
        for graph, sub in sel.groupby("graph"):
            agg[graph] = {
                c: {"mean": float(sub[f"test_acc_{c.lower()}"].mean()),
                    "values": sub[f"test_acc_{c.lower()}"].dropna().tolist()}
                for c in "ABCD"
            }
        if agg:
            made[f"count_{label}"] = plot_condition_bars(
                agg, figdir / f"fig7_count_{label}.png",
                title=f"{label} synapses, core + 20k (real vs shuffled)",
            )

    # What signing costs, at matched weight scale: the real arm under unsigned@1.0
    # (the headline), unsigned@0.5 and signed@0.5.  Without the middle bar a drop
    # from the headline cannot be attributed to the sign rather than the scale.
    cost: dict[str, dict] = {}
    for key, signed, ws in (("unsigned, w=1.0", False, 1.0),
                            ("unsigned, w=0.5", False, 0.5),
                            ("signed, w=0.5", True, 0.5)):
        sel = df[
            (df["task"] == "count") & (df["model"] == "M1") & (df["circuit"] == "core")
            & (df["graph"] == "real") & (df["n_train"] == 20000)
            & (df["max_steps"] == 0) & (df["signed"] == signed)
            & (df["standardize"] == False) & (df["w_scale"] == ws)  # noqa: E712
        ]
        if len(sel):
            cost[key] = {
                c: {"mean": float(sel[f"test_acc_{c.lower()}"].mean()),
                    "values": sel[f"test_acc_{c.lower()}"].dropna().tolist()}
                for c in "ABCD"
            }
    if len(cost) > 1:
        made["signed_cost"] = plot_condition_bars(
            cost, figdir / "fig7_signed_cost.png",
            title="Cost of signing synapses (real connectome, matched scale)",
        )

    # ---- Figure 8: the addition curriculum ---------------------------- #
    # Only runs sharing the primary held-out pair are drawn: the leave-pair-out
    # block repeats the same conditions at other holdouts, and averaging them into
    # one bar would report a mixture of two different tests.
    curr = [s for s in summaries
            if s.get("task") == "add_curriculum" and s.get("sum_train") is not None]
    if curr:
        counts: dict[str, int] = {}
        for s in curr:
            counts[_holdout_key(s)] = counts.get(_holdout_key(s), 0) + 1
        primary = max(counts, key=lambda k: counts[k])
        cells: dict[str, list[dict]] = {}
        for s in curr:
            if _holdout_key(s) != primary:
                continue
            key = f"{s.get('graph')}/{'scratch' if s.get('scratch') else 'pretrained'}"
            cells.setdefault(key, []).append(
                {"sum_train": s.get("sum_train"), "sum_test": s.get("sum_test")}
            )
        if cells:
            made["curriculum"] = plot_curriculum(cells, figdir / "fig8_curriculum.png")

    # ---- Figure 9: the lesion panel ----------------------------------- #
    # One panel per source model; average them so the figure reports the same thing as
    # the report table.  The random knockouts keep every model's values, which is what
    # makes the spread visible rather than averaged away.
    lesion_files = sorted(paths.DATA_PROCESSED.glob("lesion_*.json"))
    panels = []
    for f in lesion_files:
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if s.get("deltas"):
            panels.append(s)
    if panels:
        names = sorted({n for p in panels for n in p["deltas"]})
        merged = {
            "source_run": ", ".join(p.get("source_run", "?") for p in panels),
            "n_neuron_types": panels[0].get("n_neuron_types", {}),
            "deltas": {},
        }
        for name in names:
            per = [p["deltas"][name] for p in panels if name in p["deltas"]]
            merged["deltas"][name] = {
                c: float(np.mean([d[c] for d in per])) for c in "ABCD"
            }
        if len(panels) > 1:
            # keep each model's random knockouts distinct so the figure shows a spread
            for i, p in enumerate(panels):
                for name, d in p["deltas"].items():
                    if name.startswith("random_"):
                        merged["deltas"][f"{name}_m{i}"] = d
        made["lesion"] = plot_lesion(merged, figdir / "fig9_lesion.png")

    return made


# --------------------------------------------------------------------------- #
class _Rehydrated:
    """Minimal stand-in for :class:`~flynum.retina.hexmap.HexField` for plotting."""

    def __init__(self, xy, side):
        self.xy = np.asarray(xy, dtype=np.float32)
        self.side = np.asarray(side, dtype=object)
        self.columns = self.xy

    @property
    def n_cols(self) -> int:
        return len(self.xy)

    @property
    def bounds(self):
        return (
            float(self.xy[:, 0].min()), float(self.xy[:, 0].max()),
            float(self.xy[:, 1].min()), float(self.xy[:, 1].max()),
        )

    @property
    def centroid(self):
        return self.xy.mean(0)


class _RehydratedEncoder:
    """Minimal stand-in for :class:`~flynum.retina.encoder.RetinaEncoder`."""

    def __init__(self, z):
        self.field = _Rehydrated(z["field_xy"], z["field_side"])

        n_total = int(z["n_columns"]) if "n_columns" in z.files else int(z["n_columns_with_input"])

        class _QC:
            n_driven_columns = int(z["n_driven_columns"])
            n_columns_with_input = int(z["n_columns_with_input"])
            n_input_neurons = int(z["n_input_neurons"])

        _QC.n_columns = n_total
        self.qc = _QC()
        self.map_mode = str(z["map_mode"]) if "map_mode" in z.files else "isotropic"
        self.image_size = int(z["image_size"]) if "image_size" in z.files else 32
        self.driven_columns = z["driven_columns"]
        self.col_uv = z["col_uv"]
        self._cover = z["coverage"]

    def coverage_mask(self):
        return self._cover.astype(bool)

    def sample(self, image):
        img = np.asarray(image, dtype=np.float32)
        S = self.image_size
        u = np.clip(self.col_uv[:, 0], -1.0, S)
        v = np.clip(self.col_uv[:, 1], -1.0, S)
        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        du = (u - u0).astype(np.float32)
        dv = (v - v0).astype(np.float32)

        def at(uu, vv):
            ok = (uu >= 0) & (uu < S) & (vv >= 0) & (vv < S)
            out = np.zeros(len(uu), dtype=np.float32)
            out[ok] = img[vv[ok], uu[ok]]
            return out

        return (
            at(u0, v0) * (1 - du) * (1 - dv)
            + at(u0 + 1, v0) * du * (1 - dv)
            + at(u0, v0 + 1) * (1 - du) * dv
            + at(u0 + 1, v0 + 1) * du * dv
        )


def save_retina_bundle(encoder, out: Path) -> Path:
    """Persist the retinal map so figures can be drawn without rebuilding it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        field_xy=encoder.field.xy,
        field_side=np.asarray(encoder.field.side, dtype=object),
        col_uv=encoder.col_uv,
        coverage=encoder.coverage_mask(),
        driven_columns=encoder.driven_columns,
        n_driven_columns=encoder.qc.n_driven_columns,
        n_columns_with_input=encoder.qc.n_columns_with_input,
        n_columns=encoder.qc.n_columns,
        n_input_neurons=encoder.n_input_neurons,
        map_mode=encoder.map_mode,
        image_size=encoder.image_size,
    )
    return out
