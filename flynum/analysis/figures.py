"""Publication-style figures.

Colour choices are colour-blind safe (Okabe-Ito).  Every figure writes both a
300-dpi PNG (for the report) and a vector PDF (for a manuscript).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colour-blind safe palette
C_REAL = "#0072B2"
C_SHUFFLED = "#D55E00"
C_ACCENT = "#009E73"
C_NEUTRAL = "#666666"
C_WARN = "#CC79A7"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def plot_retina_map(field, encoder, out: Path, sample_image: np.ndarray | None = None):
    """The fixed retinotopic map: hex field, image window, and a sample drive."""
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))

    xy = field.xy.astype(float)
    uniq, idx = np.unique(xy, axis=0, return_index=True)
    ax = axes[0]
    ax.scatter(uniq[:, 0], uniq[:, 1], s=6, c=C_NEUTRAL, marker="h", linewidths=0)
    ax.set_title(f"Hexagonal visual field\n{len(uniq)} unique columns (both eyes)")
    ax.set_xlabel("lattice x")
    ax.set_ylabel("lattice y")
    ax.set_aspect("equal")

    ax = axes[1]
    covered = encoder.coverage_mask()
    ax.scatter(uniq[:, 0], uniq[:, 1], s=6, c="#DDDDDD", marker="h", linewidths=0)
    driven_cols = np.zeros(field.n_cols, dtype=bool)
    driven_cols[encoder.driven_columns] = True
    m = driven_cols[idx]
    ax.scatter(uniq[m, 0], uniq[m, 1], s=8, c=C_ACCENT, marker="h", linewidths=0)
    x0, x1, y0, y1 = field.bounds
    cx, cy = field.centroid
    S = encoder.image_size
    ax.plot(
        [cx - S / 2, cx + S / 2, cx + S / 2, cx - S / 2, cx - S / 2],
        [cy - S / 2, cy - S / 2, cy + S / 2, cy + S / 2, cy - S / 2],
        color=C_WARN, lw=1.2, ls="--",
    )
    ax.set_title(
        f"Image window ({encoder.map_mode})\n{encoder.qc.n_driven_columns} columns driven"
    )
    ax.set_aspect("equal")

    ax = axes[2]
    if sample_image is None:
        rng = np.random.default_rng(0)
        sample_image = np.zeros((S, S), dtype=np.float32)
        for _ in range(4):
            cx_, cy_ = rng.uniform(6, S - 6, 2)
            r = rng.uniform(1.8, 3.0)
            yy, xx = np.mgrid[0:S, 0:S]
            sample_image = np.maximum(
                sample_image, ((xx + 0.5 - cx_) ** 2 + (yy + 0.5 - cy_) ** 2 <= r * r)
            )
    ax.imshow(sample_image, cmap="gray_r", origin="upper")
    ax.set_title("Stimulus (32x32)")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[3]
    vals = encoder.sample(sample_image)
    ax.scatter(uniq[:, 0], uniq[:, 1], s=10, c=vals[idx], cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Column activation")
    ax.set_aspect("equal")

    fig.suptitle("Fixed retinotopic encoder (never trained)", y=1.04, fontsize=11)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
def plot_system_diagram(out: Path, info: dict | None = None, baselines: dict | None = None):
    """Figure 1 -- the experimental pipeline and where the connectome sits in it.

    Everything the network is *allowed* to learn is drawn with a dashed outline;
    everything fixed is solid.  The distinction is the whole point of the design:
    the image-to-neuron map and the wiring are frozen, only the synaptic gains and
    the readout move.
    """
    _style()
    info = info or {}
    fig = plt.figure(figsize=(11.0, 4.6))
    ax = fig.add_axes([0.02, 0.06, 0.62, 0.86])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, text, *, fc, ec, ls="-", fs=8.5):
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                          lw=1.4, linestyle=ls, zorder=2)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, linespacing=1.35)

    def arrow(x0, y0, x1, y1, label="", dy=0.16):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#333333"))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center",
                    fontsize=7.4, color="#333333")

    box(0.05, 4.55, 1.75, 1.0, "dot image\n32 x 32\n(1-5 dots)",
        fc="#FFFFFF", ec=C_NEUTRAL)
    box(2.35, 4.55, 2.15, 1.0,
        f"retina map\n{info.get('n_columns', '~1771')} hexagonal columns\n"
        f"{info.get('n_driven_columns', '~1356')} driven",
        fc="#EAF3FA", ec=C_REAL, ls="--")
    box(5.05, 4.55, 2.25, 1.0,
        f"MaleCNS subgraph\n{info.get('n_neurons', 33480):,} neurons\n"
        f"{info.get('n_edges', 1170024):,} edges",
        fc="#EAF3FA", ec=C_REAL)
    box(7.85, 4.55, 2.1, 1.0, "recurrent state\nh(t), 8 steps\n"
        r"$h_{t+1}=(1-\alpha)h_t+\alpha\phi(Wh_t+I_t)$".replace(",", ","),
        fc="#EAF3FA", ec=C_REAL)
    box(7.85, 2.85, 2.1, 1.1, "visual projection\nneurons (VPN)\n9,200 readout pool",
        fc="#EAF3FA", ec=C_REAL)
    box(5.05, 1.15, 2.25, 1.1, "linear readout\n(VPN -> classes)",
        fc="#FDF0E6", ec=C_SHUFFLED, ls="--")
    box(1.35, 1.15, 2.4, 1.1, "prediction\ncount 1-5  /  a + b",
        fc="#FFFFFF", ec=C_NEUTRAL)

    arrow(1.80, 5.05, 2.35, 5.05, "fixed")
    arrow(4.50, 5.05, 5.05, 5.05, "fixed\ndrive")
    arrow(7.30, 5.05, 7.85, 5.05)
    arrow(8.90, 4.55, 8.90, 3.95)
    arrow(8.90, 2.85, 8.90, 2.30)
    arrow(7.85, 1.70, 3.75, 1.70)
    ax.annotate("", xy=(5.05, 2.10), xytext=(7.30, 2.10),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#333333",
                                connectionstyle="arc3,rad=0.35"))

    ax.text(0.05, 5.85, "Fixed (never trained)", fontsize=8.6, color=C_REAL,
            fontweight="bold")
    ax.text(0.05, 0.72,
            "only these are trained:  per-edge gains  g = 1 + delta   (delta init 0, "
            "lambda*mean(delta^2) penalty)  +  the linear readout",
            fontsize=7.6, color=C_SHUFFLED)
    ax.text(0.05, 0.40,
            "topology is frozen:  W_ij = 0  stays 0 -- no connection absent from the "
            "connectome can ever appear",
            fontsize=7.6, color="#333333")
    ax.set_title("Figure 1 | Connectome-constrained network", fontsize=10.5, loc="left")

    # right panel: baselines, to show what the task demands
    ax2 = fig.add_axes([0.70, 0.14, 0.28, 0.72])
    if baselines:
        names = list(baselines)
        conds = ["A", "B", "C", "D"]
        x = np.arange(len(conds))
        width = 0.8 / max(len(names), 1)
        for i, name in enumerate(names):
            vals = [baselines[name][c]["accuracy"] for c in conds]
            ax2.bar(x + i * width - 0.4 + width / 2, vals, width,
                    label=name, color=PALETTE[i % len(PALETTE)])
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"cond {c}" for c in conds])
        ax2.set_ylabel("accuracy")
        ax2.axhline(0.2, color=C_NEUTRAL, ls=":", lw=1)
        ax2.set_title("Non-connectome baselines", fontsize=9)
        ax2.legend(fontsize=7.5)
        ax2.set_ylim(0, 1.05)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "baselines\nnot computed yet", ha="center", va="center",
                 fontsize=8, color=C_NEUTRAL)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
def plot_epoch_curves(
    series: dict[str, dict[str, list[float]]],
    out: Path,
    *,
    xlabel: str = "epoch",
    ylabel: str = "validation accuracy",
    title: str = "Learning speed: real connectome vs shuffled control",
    chance: float | None = None,
):
    """Validation accuracy against *training epoch*, not training-set size.

    Two graphs with the same final accuracy can still differ enormously in how
    fast they get there, and that difference is the inductive bias.  This figure
    is the one that makes it visible: at a matched number of gradient steps, is
    one wiring already better?

    ``series`` maps graph name -> {"mean": [...], "std": [...], "runs": n}.
    """
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    colors = {"real": C_REAL, "shuffled": C_SHUFFLED}
    for i, (graph, curves) in enumerate(series.items()):
        mean = np.asarray(curves["mean"], dtype=float)
        std = np.asarray(curves.get("std", np.zeros_like(mean)), dtype=float)
        x = np.arange(len(mean))
        col = colors.get(graph, PALETTE[i % len(PALETTE)])
        ax.plot(x, mean, "-", color=col, lw=1.8,
                label=f"{graph} (n={curves.get('runs', 1)})")
        ax.fill_between(x, mean - std, mean + std, color=col, alpha=0.18, linewidth=0)
    if chance is not None:
        ax.axhline(chance, color=C_NEUTRAL, ls=":", lw=1, label=f"chance = {chance:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.15)
    ax.set_ylim(0, 1.0)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
def plot_gap_over_epochs(
    gaps: dict[str, list[float]], out: Path, *, chance: float | None = None
):
    """Real-minus-shuffled validation gap against epoch, per training-set size."""
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, (label, curve) in enumerate(gaps.items()):
        ax.plot(np.arange(len(curve)), curve, "-",
                color=PALETTE[i % len(PALETTE)], lw=1.6, label=label)
    ax.axhline(0, color=C_NEUTRAL, ls="--", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("real − shuffled (validation accuracy)")
    ax.set_title("Advantage of the real connectome over training", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    return _save(fig, out)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
def plot_confusion(
    confusions: dict[str, np.ndarray],
    label_offset: int,
    out: Path,
    title: str = "Numerosity confusion",
    vmax: float | None = None,
):
    """Row-normalised confusion matrices side by side."""
    _style()
    keys = list(confusions)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.2), squeeze=False)
    for ax, key in zip(axes[0], keys):
        cm = np.asarray(confusions[key], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1 if vmax is None else vmax)
        n = cm.shape[0]
        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(i + label_offset) for i in ticks])
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(i + label_offset) for i in ticks])
        ax.set_xlabel("predicted")
        ax.set_title(key)
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if norm[i, j] > 0.55 else "black",
                )
    axes[0][0].set_ylabel("true count")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="fraction of true class")
    fig.suptitle(title, y=1.03, fontsize=11)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
def plot_learning_curve(
    curves: dict[str, dict[str, dict]],
    out: Path,
    metric: str = "test_acc_a",
    ylabel: str = "count accuracy",
    title: str = "Sample efficiency: real connectome vs shuffled control",
    chance: float | None = None,
):
    """Accuracy vs training-set size for each graph, with seed spread.

    ``curves`` maps graph name -> {n_train_label -> {"mean":..,"std":..,"values":[..]}}
    """
    _style()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    colors = {"real": C_REAL, "shuffled": C_SHUFFLED}
    for i, (graph, by_n) in enumerate(curves.items()):
        xs = sorted(by_n, key=lambda k: int(k))
        mean = np.array([by_n[k]["mean"] for k in xs])
        std = np.array([by_n[k]["std"] for k in xs])
        x = np.array([int(k) for k in xs])
        col = colors.get(graph, PALETTE[i % len(PALETTE)])
        ax.plot(x, mean, "-o", color=col, label=f"{graph} (n={len(by_n[xs[0]]['values'])})", ms=4)
        ax.fill_between(x, mean - std, mean + std, color=col, alpha=0.18, linewidth=0)
        for k, m_, xv in zip(xs, mean, x):
            ax.scatter([xv] * len(by_n[k]["values"]), by_n[k]["values"],
                       s=10, color=col, alpha=0.55, zorder=3)
    if chance is not None:
        ax.axhline(chance, color=C_NEUTRAL, ls=":", lw=1, label=f"chance = {chance:.2f}")
    ax.set_xscale("log")
    ax.set_xticks([100, 500, 1000, 5000, 20000])
    ax.set_xticklabels(["100", "500", "1k", "5k", "20k"])
    ax.set_xlabel("training samples")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.15)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
def plot_addition_matrix(
    matrix: dict,
    holdout_pairs: list[tuple[int, int]],
    out: Path,
    a_min: int,
    a_max: int,
    title: str = "Addition: a dots then b dots -> a+b",
):
    """Accuracy per (a, b); held-out pairs are marked so generalisation is visible."""
    _style()
    n = a_max - a_min + 1
    acc = np.full((n, n), np.nan)
    for i, a in enumerate(range(a_min, a_max + 1)):
        for j, b in enumerate(range(a_min, a_max + 1)):
            acc[i, j] = matrix.get(f"{a}+{b}", {}).get("accuracy", np.nan)

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(acc, cmap="Blues", vmin=0, vmax=1)
    hold = {(a, b) for a, b in holdout_pairs}
    for i, a in enumerate(range(a_min, a_max + 1)):
        for j, b in enumerate(range(a_min, a_max + 1)):
            v = acc[i, j]
            txt = "n/a" if np.isnan(v) else f"{v:.2f}"
            if (a, b) in hold:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor=C_SHUFFLED, lw=2.2)
                )
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if (not np.isnan(v) and v > 0.55) else "black")
    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in range(a_min, a_max + 1)])
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(v) for v in range(a_min, a_max + 1)])
    ax.set_xlabel("b (second epoch)")
    ax.set_ylabel("a (first epoch)")
    n_hold = len(hold)
    ax.set_title(f"{title}\norange boxes = never seen in training ({n_hold} pairs)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="accuracy")
    return _save(fig, out)


# --------------------------------------------------------------------------- #
def plot_shuffle_mixing(stats: dict, out: Path):
    """Edge-overlap decay during the swap procedure (evidence of good mixing)."""
    _style()
    curve = stats.get("overlap_curve") or []
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    if curve:
        x = np.array([c[0] for c in curve], dtype=float)
        y = np.array([c[1] for c in curve], dtype=float)
        ax.plot(x / max(stats.get("edges", 1), 1), y, "-o", ms=3, color=C_SHUFFLED)
    ax.set_xlabel("swap attempts per edge")
    ax.set_ylabel("edge overlap with real graph")
    ax.set_title(
        f"Shuffled control mixing\n{stats.get('swaps', 0):,} swaps, "
        f"final overlap {100 * stats.get('final_edge_overlap', float('nan')):.2f}%",
        fontsize=9,
    )
    ax.grid(alpha=0.15)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
def plot_gain_distribution(
    delta: np.ndarray,
    out: Path,
    presyn_type: np.ndarray | None = None,
    *,
    top_k: int = 20000,
):
    """How far training moved the synaptic gains away from the measured wiring.

    ``presyn_type`` gives the presynaptic cell type of each edge; the right panel
    then shows which cell types account for the largest gain changes.  Types are
    ranked with a *stable* digest rather than Python's ``hash``, which is salted
    per process and would make the figure differ between runs.
    """
    import hashlib

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax = axes[0]
    ax.hist(delta, bins=120, color=C_REAL, alpha=0.85)
    ax.axvline(0, color=C_NEUTRAL, ls="--", lw=1)
    ax.set_xlabel(r"$\delta$   ($g = 1 + \delta$)")
    ax.set_ylabel("edges")
    ax.set_title(
        f"Trained gain deviation\nmean|d|={np.abs(delta).mean():.4f}, "
        f"max|d|={np.abs(delta).max():.3f}",
        fontsize=9,
    )
    ax = axes[1]
    if presyn_type is not None and len(presyn_type) == len(delta):
        k = min(top_k, len(delta))
        order = np.argsort(-np.abs(delta))[:k]
        types = np.asarray(presyn_type)[order].astype(str)
        vocab, codes = np.unique(types, return_inverse=True)
        counts = np.bincount(codes, minlength=len(vocab)).astype(float)
        counts /= max(counts.sum(), 1)
        keep = np.argsort(-counts)[:15]
        ax.barh(
            np.arange(len(keep))[::-1], counts[keep], color=C_ACCENT
        )
        ax.set_yticks(np.arange(len(keep))[::-1])
        ax.set_yticklabels([str(vocab[i]) for i in keep], fontsize=7)
        ax.set_xlabel(f"share of the {k:,} most-changed edges")
        ax.set_title("Presynaptic cell types that move most", fontsize=9)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "cell types unavailable", ha="center", va="center",
                fontsize=8, color=C_NEUTRAL)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
def plot_temporal_decoding(dec: dict, meta: dict, out: Path, title: str = ""):
    """Linear decodability of a, b and a+b at every recurrent step.

    The phase shading makes the diagnosis readable at a glance: if ``a`` stays
    decodable through the blank delay the network holds the first count, and if
    ``a+b`` never rises above chance the failure is in the combination, not in
    the memory.
    """
    _style()
    T = int(dec.get("T", 0))
    if T == 0:
        raise ValueError("temporal decoding record has no timesteps")
    x = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    bands = [
        (meta.get("epoch_a", []), "#EAF3FA", "epoch A"),
        (meta.get("gap", []), "#F2F2F2", "delay"),
        (meta.get("epoch_b", []), "#FDF0E6", "epoch B"),
    ]
    for steps, colour, label in bands:
        if not steps:
            continue
        ax.axvspan(min(steps) - 0.5, max(steps) + 0.5, color=colour, zorder=0)
        ax.text(np.mean(steps), 1.005, label, ha="center", va="bottom",
                fontsize=7.5, color=C_NEUTRAL, transform=ax.get_xaxis_transform())

    styles = {"a": (C_REAL, "-o", "first count a"),
              "b": (C_SHUFFLED, "-s", "second count b"),
              "sum": (C_ACCENT, "-^", "sum a+b")}
    for key, (colour, fmt, label) in styles.items():
        if key not in dec:
            continue
        ax.plot(x, dec[key]["curve"], fmt, ms=3.5, color=colour, lw=1.5, label=label)
        chance = dec[key].get("chance")
        if chance is not None:
            ax.axhline(chance, color=C_NEUTRAL, ls=":", lw=1)
    ax.set_xlabel("recurrent step")
    ax.set_ylabel("linear-probe accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0.5, T + 0.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(title or "Temporal decoding of the addition task", fontsize=10)
    ax.grid(alpha=0.15)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
def plot_propagation(calibration: dict, out: Path):
    """Activity build-up per superclass -- does the image reach the readout?"""
    _style()
    prop = calibration.get("propagation", {})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    ax = axes[0]
    for i, (name, curve) in enumerate(prop.items()):
        ax.plot(range(1, len(curve) + 1), curve, "-o", ms=3,
                color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xlabel("recurrent step")
    ax.set_ylabel("mean activity")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_title("Signal propagation", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)

    ax = axes[1]
    sweep = calibration.get("sweep", [])
    ws = [s["w_scale"] for s in sweep]
    ax.plot(ws, [s["activity_final"] for s in sweep], "-o", ms=3, color=C_REAL,
            label="mean activity")
    ax.plot(ws, [s["dead_fraction"] for s in sweep], "-s", ms=3, color=C_SHUFFLED,
            label="dead fraction")
    chosen = calibration.get("chosen_w_scale")
    if chosen is not None:
        ax.axvline(chosen, color=C_ACCENT, ls="--", lw=1.2, label=f"chosen {chosen:g}")
    ax.set_xlabel("w_scale")
    ax.set_title("Weight-scale calibration", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)
    return _save(fig, out)
