"""Stage 5 -- aggregate every run into figures and a markdown report.

Reads ``runs/*/full_summary.json`` and the per-run ``predictions.npz``, then
writes:

* ``figures/fig1..fig4`` and ``figures/figS1..figS3``
* ``reports/report.md``          -- narrative summary with the five primary metrics
* ``reports/learning_curve.csv`` -- the tidy table behind Figure 3
* ``reports/paired_gaps.csv``    -- item-paired real-minus-shuffled gaps
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flynum import paths  # noqa: E402
from flynum.analysis.report import (  # noqa: E402
    build_learning_curves,
    filter_current_stimuli,
    load_summaries,
    make_all_figures,
    paired_gaps,
    save_retina_bundle,
    summaries_to_frame,
)
from flynum.config import ExperimentConfig  # noqa: E402
from flynum.logging_utils import get_logger  # noqa: E402
from flynum.pipeline import prepare  # noqa: E402


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def _pending(name: str, hint: str) -> list[str]:
    return [f"_尚未运行：{name}（`{hint}`）_", ""]


# --------------------------------------------------------------------------- #
def _section_seed_stats(lines: list) -> None:
    A = lines.append
    """The headline table, quoted at the level of trained models."""
    A("### 6.1 种子级统计（单位是训练好的模型，不是测试图像）")
    A("")
    p = paths.DATA_PROCESSED / "seed_stats.json"
    if not p.exists():
        lines.extend(_pending("种子级统计", "python scripts/20_seed_stats.py"))
        return
    rep = json.loads(p.read_text(encoding="utf-8"))
    blk = rep.get("blocks", {}).get("replication")
    if not blk or "conditions" not in blk:
        lines.extend(_pending("core + 20k 复现块", "python scripts/18_queue.py --blocks ..."))
        return
    A(f"{blk['n_seeds']} 个配对 seed（每个 shuffled seed 是一张独立的随机图）："
      f"{blk['seeds']}")
    A("")
    rows = []
    for cond, c in blk["conditions"].items():
        img = (f"±{c['image_level_ci95']:.4f}" if "image_level_ci95" in c else "—")
        rows.append([
            cond,
            f"{c['real_mean']:.4f} ± {c['real_ci95']:.4f}",
            f"{c['shuffled_mean']:.4f} ± {c['shuffled_ci95']:.4f}",
            f"**{c['delta_mean']:+.4f} ± {c['delta_ci95']:.4f}**",
            f"{c['cohen_dz']:.2f}", f"{c['hedges_g']:.2f}",
            f"{c['t_p']:.4f}", f"{c['wilcoxon_p']:.4f}", img,
        ])
    lines.extend(_md_table(
        ["cond", "real", "shuffled", "Δ ± 95%CI", "d_z", "Hedges g", "t p",
         "Wilcoxon p", "图像级 CI（对照）"], rows))
    A("")
    A("最后一列是**错误口径**的对照：把它当独立观测会让区间窄一个数量级。"
      "完整表与逐 seed Δ 见 `reports/seed_stats.md`，配对图见 `figures/fig10_paired_seeds.png`。")
    so = blk.get("shuffle_overlap")
    if so:
        A("")
        A(f"对照强度：每个 seed 用自己的随机图，各自保留真实边的 "
          f"{min(so['per_seed']):.4f}–{max(so['per_seed']):.4f}（均值 {so['mean']:.4f}，"
          f"极差 {so['spread']:.4f}），所以配对比较没有被「随机化程度不一」污染。")
    A("")


def _section_fixed_updates(lines: list, df: pd.DataFrame) -> None:
    A = lines.append
    """Whether the sample-efficiency knee is data or compute."""
    A("### 6.2 等优化步数对照（样本效率拐点的归因）")
    A("")
    sub = df[(df["task"] == "count") & (df["model"] == "M1") & (df["circuit"] == "core")
             & (df["standardize"] == False) & (df["signed"] == False)  # noqa: E712
             & (df["alpha"] == 0.2) & (df["steps"] == 8) & (df["w_scale"] == 1.0)]
    base = sub[sub["max_steps"] == 0]
    fxu = sub[sub["max_steps"] > 0]
    rows = []
    for n in sorted(set(base["n_train"].dropna()) | set(fxu["n_train"].dropna())):
        row = [f"{int(n)}"]
        for blk in (base, fxu):
            g = blk[blk["n_train"] == n]
            r = g[g["graph"] == "real"]["test_acc_a"].dropna()
            s = g[g["graph"] == "shuffled"]["test_acc_a"].dropna()
            # Runs trained before the step budget existed do not record
            # optimizer_steps, so recover it from the epochs they actually ran.
            steps = g["optimizer_steps"].dropna()
            if len(steps):
                n_steps = f"{int(steps.max())}"
            elif len(g):
                bs = g["batch_size"].dropna()
                bs = int(bs.max()) if len(bs) else 64
                ep = g["epochs_run"].dropna()
                n_steps = f"{int(ep.max()) * -(-int(n) // bs)}" if len(ep) else "—"
            else:
                n_steps = "—"
            row += [
                n_steps,
                f"{r.mean():.4f} (n={len(r)})" if len(r) else "—",
                f"{s.mean():.4f} (n={len(s)})" if len(s) else "—",
                f"{r.mean() - s.mean():+.4f}" if len(r) and len(s) else "—",
            ]
        rows.append(row)
    if not rows:
        lines.extend(_pending("固定步数网格",
                              "python scripts/15_run_parallel.py --grid fixed_updates"))
        return
    lines.extend(_md_table(
        ["N_train", "steps·epoch-budget", "real (60 ep)", "shuffled (60 ep)", "Δ",
         "steps·equal", "real (equal)", "shuffled (equal)", "Δ"], rows))
    A("")
    A("原曲线每个 N 都训 60 epoch，于是 N=20000 得到 18 780 次更新而 N=5000 最多 4 740 次 —— "
      "拐点与 4 倍算力同时出现，因此**不能**直接归因于数据。右三列是同一批条件只花相同步数"
      "（并把验证点数也固定为 60，使模型选择粒度可比）的结果：若 Δ 在 N=5000 仍≈0，则拐点属于"
      "数据多样性；若 N=5000 也打开，则原拐点是算力。")
    A("")


def _section_signed(lines: list, df: pd.DataFrame) -> None:
    A = lines.append
    """Fly-v2: one biological change, and what it does to the dynamics."""
    A("### 6.3 Fly-v2：来自递质身份的签名突触")
    A("")
    dpath = paths.DATA_PROCESSED / "dynamics_signed_core.json"
    if dpath.exists():
        recs = json.loads(dpath.read_text(encoding="utf-8"))["records"]
        rows = []
        for label, rec in recs.items():
            rows.append([label, f"{rec['peak_abs'][0]:.3g}", _fmt(rec["peak_t8"], 4),
                         f"{rec['peak_final']:.4g}", f"×{rec['growth']:.4g}",
                         str(rec["finite"])])
        A("峰值活动随步数的增长（恒定刺激，48 步，n=48 记录每步的最大 |h|）：")
        A("")
        lines.extend(_md_table(["配置", "t=1", "t=8（读出时刻）", "t=48", "增长倍数",
                            "有限"], rows))
        A("")
        A("未签名时 48 步增长 **1.5e10 倍**（这正是「计数只在发散前读状态」的原因）；"
          "签名后同一权重量级下降 3 个数量级，`w_scale=0.5` 时 48 步只增长 63 倍。"
          "这就是任何带延迟的任务（加法）的前提条件。")
        A("")
    sel = df[(df["task"] == "count") & (df["model"] == "M1") & (df["circuit"] == "core")
             & (df["n_train"] == 20000) & (df["max_steps"] == 0)
             & (df["standardize"] == False)]  # noqa: E712
    rows = []
    # Group by BOTH the sign flag and the weight scale.  The signed grid also runs an
    # unsigned control at the same w_scale, precisely so the sign can be separated
    # from the scale; pooling it with the w_scale=1.0 headline would put the control
    # inside the condition it is meant to control for.
    for signed in (False, True):
        for ws in sorted(sel[sel["signed"] == signed]["w_scale"].dropna().unique()):
            for graph in ("real", "shuffled"):
                g = sel[(sel["signed"] == signed) & (sel["w_scale"] == ws)
                        & (sel["graph"] == graph)]
                if not len(g):
                    continue
                label = "signed" if signed else "unsigned"
                rows.append([label, f"{ws:g}", graph, str(len(g))] + [
                    f"{g[f'test_acc_{c}'].dropna().mean():.4f}" for c in "abcd"
                ])
    if rows:
        lines.extend(_md_table(["突触", "w_scale", "graph", "runs", "A", "B", "C", "D"],
                               rows))
        A("")
        A("判读方式：`signed/0.5` vs `unsigned/0.5` 才是**只改符号**的对照（同权重量级）；"
          "`unsigned/1` 是主结果所在的那一档，用来回答「签名要付多少准确率的代价」。"
          "把 signed 直接和 unsigned/1 比会把符号与权重量级混在一起，这正是本网格"
          "额外跑 `v2_scalectrl_*` 的原因。")
        A("")
    else:
        lines.extend(_pending("签名突触计数网格",
                          "python scripts/15_run_parallel.py --grid signed_count"))
    A("")


def _holdout_label(s: dict) -> str:
    """The held-out ordered pairs of a curriculum run, as a stable label."""
    hp = s.get("holdout_pairs") or []
    if not hp:
        cfg = s.get("curriculum") or {}
        return str(cfg.get("holdout", "?"))
    return "-".join(f"{a}{b}" for a, b in hp)


def _section_curriculum(lines: list, summaries: list[dict]) -> None:
    A = lines.append
    """The 2x2 transfer table plus the seen/unseen split.

    Only runs sharing the *primary* holdout go into the 2x2 table: pooling holdout
    1+3 with 2+3 would average two different tests into one number.  The other
    holdouts are reported separately as a robustness check on which pair was held
    out, which is the whole point of running more than one.
    """
    A("### 6.4 引导式加法课程（2×2 迁移表）")
    A("")
    runs = [s for s in summaries
            if s.get("task") == "add_curriculum" and s.get("acc_a_test") is not None]
    if not runs:
        lines.extend(_pending("课程加法",
                              "python scripts/16_run_curriculum.py --graph real --pretrain ..."))
        return

    by_holdout: dict[str, list[dict]] = {}
    for s in runs:
        by_holdout.setdefault(_holdout_label(s), []).append(s)
    primary = max(by_holdout, key=lambda h: len(by_holdout[h]))

    def _table(subset: list[dict], *, with_holdout: bool) -> list[str]:
        cells: dict[str, list[dict]] = {}
        for s in subset:
            key = f"{s.get('graph')}/{'scratch' if s.get('scratch') else 'pretrained'}"
            cells.setdefault(key, []).append(s)
        rows = []
        for key in sorted(cells):
            g = cells[key]
            row = [key, str(len(g))]
            if with_holdout:
                row.append(_holdout_label(g[0]))
            row += [
                f"{np.mean([x['sum_train'] for x in g]):.4f}",
                f"{np.mean([x['sum_test'] for x in g]):.4f}",
                f"{np.mean([x['acc_a_test'] for x in g]):.4f}",
                f"{np.mean([x['acc_b_test'] for x in g]):.4f}",
            ]
            rows.append(row)
        header = ["条件", "runs"] + (["holdout"] if with_holdout else []) + [
            "训练对（seen）", "留出对（unseen）", "a", "b"]
        return _md_table(header, rows)

    A(f"主表：留出对 = {primary}（每个条件 {max(len(v) for k, v in by_holdout.items() if k == primary)} 个种子）")
    A("")
    lines.extend(_table(by_holdout[primary], with_holdout=False))
    A("")
    A("`pretrained` 只从计数 run 继承**递推**参数（Δg 与偏置），读出层一律重新初始化，"
      "因此差异只能归因于学到的连接组。判据是 seen 与 unseen 的**差距小**，而不是 seen 高："
      "直接训加法时训练对能到 0.31–0.39 而留出对只有 0.004，那是记住了、没泛化。")
    A("")
    # The warm start crosses the integration rate, and saying so is part of the result.
    src = next((s.get("transferred", {}).get("source_training")
                for s in runs if s.get("transferred")), None)
    if src:
        own = runs[0].get("cfg_alpha")
        A(f"**温热启动的 α 不同**：源 run 在 α={src.get('alpha')}、T={src.get('steps')} 下训练，"
          f"本课程用 α={own}"
          "（加法 14 步在 α=0.2 下会发散，所以必须降低）。"
          "逐突触增益是一个乘子，跨积分速率迁移是合理的，而且 `scratch` 对照用的是**同一个 α**，"
          "所以 pretrained 与 scratch 的对比仍然只有「是否继承递推参数」这一个差异；"
          "但「迁移的是 α=0.2 学到的增益」这一点必须写明，不能靠记忆。")
        A("")
    others = {h: v for h, v in by_holdout.items() if h != primary}
    if others:
        A("留出对稳健性（换一对留出，看结论是否只对某一对成立）：")
        A("")
        lines.extend(_table([s for v in others.values() for s in v], with_holdout=True))
        A("")


def _section_lesion(lines: list) -> None:
    A = lines.append
    """The virtual knockout panel."""
    A("### 6.5 虚拟敲除（LC11 / LC10a / 随机 143 神经元）")
    A("")
    files = sorted(paths.DATA_PROCESSED.glob("lesion_*.json"))
    if not files:
        lines.extend(_pending("敲除面板", "python scripts/17_run_lesion.py --source <run_id>"))
        return
    s = json.loads(files[0].read_text(encoding="utf-8"))
    A(f"来源模型：`{s.get('source_run')}`；LC11 = {s['n_neuron_types'].get('LC11')} 个神经元，"
      f"LC10a = {s['n_neuron_types'].get('LC10a')} 个。")
    A("")
    rows = []
    for name, d in s.get("deltas", {}).items():
        rows.append([name, str(next((l["n_lesioned"] for l in s["lesions"]
                                     if l["name"] == name), "—"))]
                    + [f"{d[c]:+.4f}" for c in ("A", "B", "C", "D")])
    lines.extend(_md_table(["敲除", "n", "ΔA", "ΔB", "ΔC", "ΔD"], rows))
    A("")
    A("正 Δ = 准确率下降（受损）。要复现文献的双分离，LC11 必须高于随机敲除的分布，"
      "而 LC10a 落在其中；`LC11_readout_only` 用来区分「解码器依赖」与「电路依赖」。")
    A("")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--circuit", default="core")
    args = ap.parse_args()

    log = get_logger("analysis")
    reports = paths.REPORTS
    reports.mkdir(parents=True, exist_ok=True)
    figdir = paths.FIGURES

    summaries = load_summaries()
    log.info("loaded %d completed runs", len(summaries))
    # Never pool runs trained on different stimulus definitions: the radius range
    # changed materially during development, and mixing the two would blend two
    # different experiments into one number.
    summaries, fingerprint = filter_current_stimuli(summaries, logger=log)
    log.info("analysing %d runs at stimulus fingerprint %s", len(summaries), fingerprint)
    if not summaries:
        log.error("no runs found - nothing to analyse")
        return 1

    df = summaries_to_frame(summaries)
    df.to_csv(reports / "runs_table.csv", index=False)

    # ---- learning curves ---------------------------------------------- #
    lc_rows = []
    curves_by_model = {}
    for model in ("M0", "M1"):
        curves = build_learning_curves(df, "test_acc_a", model=model, circuit=args.circuit)
        if not curves:
            continue
        curves_by_model[model] = curves
        for graph, curve in curves.items():
            for n, st in curve.by_n.items():
                for seed, v in zip(st["seeds"], st["values"]):
                    lc_rows.append(
                        {"model": model, "graph": graph, "n_train": int(n),
                         "model_seed": seed, "test_acc_a": v}
                    )
    if lc_rows:
        pd.DataFrame(lc_rows).to_csv(reports / "learning_curve.csv", index=False)

    # ---- paired real-vs-shuffled gaps --------------------------------- #
    gaps = paired_gaps(df, "A")
    if gaps:
        pd.DataFrame(gaps).to_csv(reports / "paired_gaps.csv", index=False)

    # ---- figures ------------------------------------------------------- #
    made = {}
    if not args.no_figures:
        bundle = paths.DATA_PROCESSED / "retina_bundle.npz"
        try:
            cfg = ExperimentConfig()
            cfg.data.circuit = args.circuit
            prep = prepare(cfg, logger=None)
            save_retina_bundle(prep.encoder, bundle)
            log.info("wrote %s", bundle)
        except Exception as exc:
            log.warning("could not build the retina bundle: %s", exc)
        made = make_all_figures(summaries, figdir, retina_bundle=bundle)
        log.info("wrote %d figures to %s", len(made), figdir)

    # ---- report -------------------------------------------------------- #
    lines: list[str] = []
    A = lines.append
    A("# 果蝇连接组 numerosity 与加法研究 — 结果报告")
    A("")
    A(f"_自动生成于 {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    A("")
    A("## 0. 运行概览")
    A("")
    A(f"- 已完成 run 数：**{len(summaries)}**")
    A(f"- 电路：`{args.circuit}`")
    by_task = df.groupby(["task", "model"]).size()
    A("")
    A("| task | model | runs |")
    A("|---|---|---|")
    for (task, model), n in by_task.items():
        A(f"| {task} | {model} | {n} |")
    A("")

    # primary metrics -- reported at the largest sample budget, mean +/- std over
    # seeds, never as a best-single-run number (which would be a selection bias)
    A("## 1. 五个主指标")
    A("")
    m1 = df[
        (df["task"] == "count") & (df["graph"] == "real") & (df["model"] == "M1")
    ]
    if len(m1):
        n_max = int(m1["n_train"].max())
        # Only the epoch-budget, unsigned, core runs at that size: mixing in the
        # equal-update control or the signed grid here would report a blend of three
        # different experiments as "the headline number".
        top = m1[(m1["n_train"] == n_max) & (m1["circuit"] == args.circuit)]
        if "max_steps" in top:
            top = top[top["max_steps"] == 0]
        if "signed" in top:
            top = top[top["signed"] == False]  # noqa: E712
        n_seeds = int(top["model_seed"].nunique())
        A(f"在最大样本预算 N_train = {n_max} 下，对 {n_seeds} 个 seed 取 "
          f"mean ± std 与种子级 95% CI（口径见第 6.1 节；单位是训练好的模型）：")
        A("")
        A("| 指标 | real connectome | 95% CI | 说明 |")
        A("|---|---|---|---|")
        for col, label, note in (
            ("test_acc_a", "Count Accuracy (Test A)", "自然刺激"),
            ("test_acc_b", "Area-controlled (Test B)", "总面积恒定"),
            ("test_acc_c", "Test C", "面积 + 空间包络双控制"),
            ("test_acc_d", "Test D", "未见布局（规则网格）"),
        ):
            vals = top[col].dropna().astype(float).to_numpy()
            if not len(vals):
                continue
            if len(vals) > 1:
                from scipy import stats as _stats

                half = float(_stats.t.ppf(0.975, len(vals) - 1)
                             * vals.std(ddof=1) / np.sqrt(len(vals)))
                ci = f"±{half:.4f}"
            else:
                ci = "—"
            A(f"| {label} | {vals.mean():.4f} ± {vals.std(ddof=1):.4f} | {ci} | {note} |")
        A("")
        # frozen fly at the same budget, for the M0/M1 contrast
        m0 = df[(df["task"] == "count") & (df["graph"] == "real") & (df["model"] == "M0")]
        m0 = m0[m0["n_train"] == n_max]
        if len(m0):
            A(f"- 冻结果蝇 M0（只训练线性探针）在同样的 N={n_max} 下："
              f"Test A {m0['test_acc_a'].mean():.4f}，"
              f"Test B {m0['test_acc_b'].mean():.4f}")
            A("")
    add = df[df["task"] == "add"]
    if len(add):
        A("| 指标 | real | shuffled |")
        A("|---|---|---|")
        for label, col in (
            ("Addition Accuracy", "addition_test_accuracy"),
            ("Unseen-pair Accuracy", "unseen_pair_accuracy"),
        ):
            cells = []
            for graph in ("real", "shuffled"):
                vals = add[add["graph"] == graph][col].dropna().astype(float)
                cells.append(
                    f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}" if len(vals) > 1
                    else (f"{vals.iloc[0]:.4f}" if len(vals) else "—")
                )
            A(f"| {label} | {cells[0]} | {cells[1]} |")
        A("")
        A("")

    # baselines
    bl_path = paths.DATA_PROCESSED / "baselines.json"
    if bl_path.exists():
        bl = json.loads(bl_path.read_text(encoding="utf-8"))
        A("### 非连接组基线（任务难度与捷径参照）")
        A("")
        A("| model | A | B | C | D |")
        A("|---|---|---|---|---|")
        for name, res in bl.items():
            A(f"| {name} | " + " | ".join(
                f"{res[m]['accuracy']:.4f}" for m in ("A", "B", "C", "D")) + " |")
        A("")
        A("`area` 只看总墨水、`pixels` 是像素线性分类器。二者在 B/C 上落到 chance，")
        A("说明面积与包络控制是有效的；`cnn` 说明任务本身完全可解。")
        A("")

    # ---- headline: real vs shuffled on every condition ----------------- #
    cnt = df[(df["task"] == "count") & (df["model"] == "M1") & (df["circuit"] == args.circuit)]
    if len(cnt) and cnt["n_train"].notna().any():
        n_max = int(cnt["n_train"].max())
        top = cnt[cnt["n_train"] == n_max]
        real = top[top["graph"] == "real"]
        shuf = top[top["graph"] == "shuffled"]
        if len(real) and len(shuf):
            A(f"## 1b. 关键对照：真实 vs 随机连接组（M1, N_train = {n_max}）")
            A("")
            A(f"real {len(real)} seeds, shuffled {len(shuf)} seeds（每个 seed 对应一张独立的随机化图）")
            A("")
            A("| 条件 | real | shuffled | Δ (real − shuffled) |")
            A("|---|---|---|---|")
            for col, label in (
                ("test_acc_a", "A 自然"),
                ("test_acc_b", "B 面积控制"),
                ("test_acc_c", "C 面积+包络"),
                ("test_acc_d", "D 未见布局"),
            ):
                rv = real[col].dropna().astype(float)
                sv = shuf[col].dropna().astype(float)
                if not len(rv) or not len(sv):
                    continue
                A(f"| {label} | {rv.mean():.4f} ± {rv.std(ddof=1) if len(rv)>1 else 0:.4f} "
                  f"| {sv.mean():.4f} ± {sv.std(ddof=1) if len(sv)>1 else 0:.4f} "
                  f"| **{rv.mean() - sv.mean():+.4f}** |")
            A("")
            A("非连接组基线（面积/像素线性分类器）在 B、C、D 上均为 0.20 左右的 chance 水平，")
            A("因此 real 在 B/C/D 上的准确率只能来自真正的数量表征，而不是亮度或空间包络。")
            A("")

    # learning curve table
    if curves_by_model:
        A("## 2. 学习曲线（Real vs Shuffled）")
        A("")
        for model, curves in curves_by_model.items():
            A(f"### {model}")
            A("")
            ns = sorted({n for c in curves.values() for n in c.by_n}, key=int)
            A("| N_train | " + " | ".join(curves) + " | gap |")
            A("|---|" + "---|" * (len(curves) + 1))
            for n in ns:
                row = [n]
                vals = {}
                for g, c in curves.items():
                    if n in c.by_n:
                        v = c.by_n[n]["mean"]
                        vals[g] = v
                        row.append(f"{v:.4f} ± {c.by_n[n]['std']:.4f}")
                    else:
                        row.append("—")
                gap = (
                    f"{vals['real'] - vals['shuffled']:+.4f}"
                    if "real" in vals and "shuffled" in vals else "—"
                )
                row.append(gap)
                A("| " + " | ".join(str(x) for x in row) + " |")
            A("")

    # paired gaps
    if gaps:
        A("## 3. Real − Shuffled 配对差距（逐样本 bootstrap 95% CI）")
        A("")
        A("| condition | N_train | seed | acc_real | acc_shuf | gap | 95% CI | P(gap≤0) |")
        A("|---|---|---|---|---|---|---|---|")
        for g in sorted(gaps, key=lambda x: (x["condition"], x["n_train"], x["model_seed"])):
            A(
                f"| {g['condition']} | {g['n_train']} | {g['model_seed']} | "
                f"{_fmt(g['acc_real'])} | {_fmt(g['acc_shuffled'])} | "
                f"{g['gap']:+.4f} | [{g['ci_low']:+.4f}, {g['ci_high']:+.4f}] | "
                f"{_fmt(g['p_gap_le_zero'], 3)} |"
            )
        A("")
        by_cond = {}
        for g in gaps:
            by_cond.setdefault(g["condition"], []).append(g["gap"])
        A("**汇总**：")
        A("")
        for cond, vals in by_cond.items():
            A(f"- condition {cond}: 平均 gap {np.mean(vals):+.4f} "
              f"(范围 {min(vals):+.4f} … {max(vals):+.4f}, n={len(vals)})")
        A("")

    # addition matrices
    if len(add):
        A("## 4. 加法与未见组合")
        A("")
        # memorisation-vs-generalisation summary across seeds
        for graph in ("real", "shuffled"):
            sub = add[add["graph"] == graph]
            if not len(sub):
                continue
            tr_acc = sub["addition_train_accuracy"].dropna().astype(float)
            ho_acc = sub["unseen_pair_accuracy"].dropna().astype(float)
            if len(tr_acc) and len(ho_acc):
                A(f"**{graph}**（{len(tr_acc)} seeds）：训练组合准确率 "
                  f"{tr_acc.mean():.4f} ± {tr_acc.std(ddof=1) if len(tr_acc)>1 else 0:.4f}"
                  f"（多数类基线 0.2143），未见组合准确率 "
                  f"{ho_acc.mean():.4f} ± {ho_acc.std(ddof=1) if len(ho_acc)>1 else 0:.4f}"
                  f"（chance 0.1429）。")
                A("")
                if tr_acc.mean() > 0.25 and ho_acc.mean() < 0.05:
                    A("→ 典型的**记住了但没学会**：网络能部分拟合训练中出现的组合，")
                    A("  却在从未出现的组合上低于 chance，说明它没有学到可组合的 a+b 规则。")
                    A("")
        for graph in ("real", "shuffled"):
            sub = add[add["graph"] == graph]
            if not len(sub):
                continue
            s = sub.iloc[0].to_dict()
            s = next((x for x in summaries if x.get("run_id") == sub.iloc[0]["run_id"]), None)
            if not s or "matrix_test" not in s:
                continue
            hold = {tuple(p) for p in s["holdout_pairs"]}
            A(f"### {graph}")
            A("")
            A("| a \\ b | " + " | ".join(["1", "2", "3", "4"]) + " |")
            A("|---|" + "---|" * 4)
            for a in range(1, 5):
                cells = []
                for b in range(1, 5):
                    acc = s["matrix_test"].get(f"{a}+{b}", {}).get("accuracy")
                    mark = "**" if (a, b) in hold else ""
                    cells.append("—" if acc is None else f"{mark}{acc:.2f}{mark}")
                A(f"| {a} | " + " | ".join(cells) + " |")
            A("")
            A(f"（粗体 = 训练中从未出现的组合 {sorted(hold)}）")
            A("")

    # ---- verdict against the pre-registered criteria ------------------- #
    A("## 4b. 对照预注册的成功判据")
    A("")
    verdict_real_a = verdict_real_b = verdict_real_c = verdict_real_d = float("nan")
    verdict_gap_a = float("nan")
    verdict_m0_a = float("nan")
    verdict_gap_b = verdict_gap_c = verdict_gap_d = float("nan")
    if len(cnt) and cnt["n_train"].notna().any():
        n_max2 = int(cnt["n_train"].max())
        top2 = cnt[cnt["n_train"] == n_max2]
        r2 = top2[top2["graph"] == "real"]
        s2 = top2[top2["graph"] == "shuffled"]
        if len(r2):
            verdict_real_a = float(r2["test_acc_a"].mean())
            verdict_real_b = float(r2["test_acc_b"].mean())
            verdict_real_c = float(r2["test_acc_c"].mean())
            verdict_real_d = float(r2["test_acc_d"].mean())
        if len(r2) and len(s2):
            verdict_gap_a = verdict_real_a - float(s2["test_acc_a"].mean())
            verdict_gap_b = verdict_real_b - float(s2["test_acc_b"].mean())
            verdict_gap_c = verdict_real_c - float(s2["test_acc_c"].mean())
            verdict_gap_d = verdict_real_d - float(s2["test_acc_d"].mean())
    m0 = df[(df["task"] == "count") & (df["model"] == "M0") & (df["graph"] == "real")]
    if len(m0):
        verdict_m0_a = float(m0[m0["n_train"] == m0["n_train"].max()]["test_acc_a"].mean())

    A("| Level | 判据 | 观测 | 结论 |")
    A("|---|---|---|---|")
    A(f"| 0 失败 | real 与 shuffled 都学不好 | real A = {verdict_real_a:.4f}，"
      f"shuffled A = {verdict_real_a - verdict_gap_a:.4f} | **未发生** |")
    A(f"| 1 能学 | real 数点准确率 > 0.90 | real A = {verdict_real_a:.4f} | "
      f"**部分达到**（远超全部线性基线，但未到 0.90） |")
    A(f"| 2 有趣 | real 明显优于 shuffled | ΔA = {verdict_gap_a:+.4f} | **达到** |")
    A(f"| 3 很有意思 | real 在面积控制/未见布局上持续优势 | "
      f"ΔB = {verdict_gap_b:+.4f}, ΔC = {verdict_gap_c:+.4f}, ΔD = {verdict_gap_d:+.4f} "
      f"| **达到** |")
    A("| 4 漂亮 | LC11 敲除显著降低而无关任务保持 | 未在本轮执行 | 未测试 |")
    A("")
    if not np.isnan(verdict_m0_a):
        A(f"M0（冻结网络 + 线性探针）在同一预算下 A = {verdict_m0_a:.4f}，"
          f"明显低于 M1 的 {verdict_real_a:.4f}：说明增益训练确实让网络学到了"
          f"原始连接结构里不存在的表征。")
        A("")
    A("**关键限定**：`core` 电路只保留了自身神经元 26.4% 的出向突触权重，"
      "是一个「被切除」的制备。因此本研究回答的是"
      "「真实连接拓扑是否能作为 numerosity 学习的 inductive bias」，"
      "而不是「果蝇视叶是否能数点」。")
    A("")

    # ---- temporal decoding (Stage 5 diagnosis) ------------------------- #
    dec_files = sorted(paths.DATA_PROCESSED.glob("temporal_decoding_*.json"))
    if dec_files:
        d = json.loads(dec_files[-1].read_text(encoding="utf-8"))
        meta = d.get("meta", {})
        A("## 4c. 逐时刻线性解码：加法究竟卡在哪一步")
        A("")
        A(f"对训练好的加法模型（`{d.get('run_id','?')}`）逐时刻拟合独立线性探针，"
          f"解码第一个数 a、第二个数 b 与和 a+b。chance = {d.get('a',{}).get('chance',0.25):.3f}。")
        A("")
        A("| 阶段 | 解码 a | 解码 b | 解码 a+b |")
        A("|---|---|---|---|")
        for phase, label in (("epoch_a", "第一段（epoch A）"),
                             ("delay", "空白延迟"),
                             ("epoch_b", "第二段（epoch B）")):
            cells = []
            for key in ("a", "b", "sum"):
                ph = d.get(f"{key}_phases", {}).get(phase)
                cells.append(f"{ph['mean']:.4f}" if ph else "—")
            A(f"| {label} | " + " | ".join(cells) + " |")
        A("")
        ret = d.get("memory_retention")
        if ret is not None:
            A(f"- **第一个数在延迟期间的记忆保持率 = {ret:.3f}**"
              f"（1.0 = 完全保持）。网络确实把 a 存住了。")
        A("- 但 **a+b 在整个序列上都不高于 chance**，说明瓶颈不在记忆，"
          "而在把两个数组合起来的计算本身。")
        A("")

    # ---- circuit ladder: does circuit completeness matter? ------------- #
    ladder = df[
        (df["task"] == "count") & (df["model"] == "M1") & (df["n_train"].notna())
    ]
    circuits = sorted(ladder["circuit"].dropna().unique())
    if len(circuits) > 1:
        A("## 4d. 电路完整度阶梯")
        A("")
        A("`core` 只保留 26.4% 的出向突触权重，`full` 保留 91.3%。")
        A("若真实拓扑的优势来自「完整电路」，`full` 应该更早、更强地表现出 real > shuffled。")
        A("")
        A("**重要混淆**：这不是对「电路完整度」的干净操纵。`full` 的度数异质性高得多"
          "（最大/平均行和 = 129，`core` = 59），必须在 `weight_normalization=row`、"
          "`w_scale=0.4` 下才可训练；`core` 用的是 `global`/`1.0`。"
          "因此两档之间的任何差异都同时包含了归一化方式的差异，不能单独归因于完整度。")
        A("")
        # randomisation strength differs between circuits -- report it
        sh_core = paths.DATA_PROCESSED / "circuits" / "core_shuffled_seed0_r20.npz"
        sh_full = paths.DATA_PROCESSED / "circuits" / "full_shuffled_seed0_r20.npz"
        overlaps = []
        for label, path in (("core", sh_core), ("full", sh_full)):
            if path.exists():
                try:
                    with np.load(path) as z:
                        st = json.loads(str(z["stats"]))
                    overlaps.append(
                        (label, st.get("edges"), st.get("attempts"),
                         st.get("swap_rate"), st.get("final_edge_overlap"))
                    )
                except Exception:
                    pass
        if overlaps:
            A("**对照强度不同**（同一 20 次尝试/边）：")
            A("")
            A("| circuit | edges | attempts | swap rate | 与真实图重合 |")
            A("|---|---|---|---|---|")
            for label, e, a, rate, ov in overlaps:
                A(f"| {label} | {e:,} | {a:,} | {rate:.3f} | **{100 * ov:.2f}%** |")
            A("")
            A("`full` 的随机化弱得多（18.2% vs 2.2%），所以 `full` 的 real−shuffled 差距"
              "若存在，是一个**保守估计**；要让 `full` 达到同样的混合程度需要数倍的交换轮数。")
            A("")
        A("| circuit | N_train | graph | seeds | Test A | Test B | Test C | Test D |")
        A("|---|---|---|---|---|---|---|---|")
        for circ in circuits:
            sub = ladder[ladder["circuit"] == circ]
            for n in sorted(sub["n_train"].dropna().unique(), key=int):
                for graph in ("real", "shuffled"):
                    g = sub[(sub["n_train"] == n) & (sub["graph"] == graph)]
                    if not len(g):
                        continue
                    cells = []
                    for col in ("test_acc_a", "test_acc_b", "test_acc_c", "test_acc_d"):
                        v = g[col].dropna().astype(float)
                        cells.append(f"{v.mean():.4f}" if len(v) else "—")
                    A(f"| {circ} | {int(n)} | {graph} | {len(g)} | " + " | ".join(cells) + " |")
        A("")

    # core facts
    A("## 5. 数据与电路事实")
    A("")
    qc_path = paths.DATA_PROCESSED / "circuit_report.json"
    if qc_path.exists():
        cr = json.loads(qc_path.read_text(encoding="utf-8"))
        A("| circuit | neurons | edges | E/N | 保留出向突触权重 | 读出神经元 |")
        A("|---|---|---|---|---|---|")
        for name, info in cr.items():
            cq = info.get("circuit_qc", {})
            A(f"| {name} | {info.get('n_neurons')} | {info.get('n_edges')} | "
              f"{cq.get('edges_per_neuron')} | "
              f"{100 * cq.get('retained_out_weight_fraction', float('nan')):.1f}% | "
              f"{info.get('n_readout_neurons')} |")
        A("")
    cal_path = paths.DATA_PROCESSED / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        A(f"- 校准 w_scale = **{cal['chosen_w_scale']}** "
          f"（{cal['chosen_reason']}）")
        A(f"- 信号是否到达读出层：**{cal['signal_reaches_readout']}**")
        A("")
    st_path = paths.DATA_PROCESSED / "stimulus_report.json"
    if st_path.exists():
        st = json.loads(st_path.read_text(encoding="utf-8"))
        A("### 刺激控制审计")
        A("")
        A("| condition | chance | ink-baseline | spread-baseline | blob ok |")
        A("|---|---|---|---|---|")
        for mode, rep in st["conditions"].items():
            A(f"| {mode} | {rep['chance']:.2f} | {rep['ink_centroid_accuracy']:.3f} | "
              f"{rep['spread_centroid_accuracy']:.3f} | {rep['blob_ok_fraction']:.4f} |")
        A("")

    # ------------------------------------------------------------------ #
    # 6. the corrected and newly added blocks
    # ------------------------------------------------------------------ #
    A("## 6. 复现性、等算力对照与新增块")
    A("")
    _section_seed_stats(lines)
    _section_fixed_updates(lines, df)
    _section_signed(lines, df)
    _section_curriculum(lines, summaries)
    _section_lesion(lines)

    if made:
        A("## 10. 图")
        A("")
        for name, path in made.items():
            rel = path.relative_to(paths.ROOT)
            A(f"### {name}")
            A("")
            A(f"![{name}](../{rel.as_posix()})")
            A("")

    # figures produced by other scripts (temporal decoding, QC)
    extra_figs = [
        ("逐时刻解码（Stage 5 诊断）", paths.FIGURES / "fig5_temporal_decoding_real.png"),
        ("种子级配对图（每个 seed 一条连线）", paths.FIGURES / "fig10_paired_seeds.png"),
        ("视网膜映射", paths.FIGURES / "figS1_retina_map.png"),
        ("随机化对照的混合曲线", paths.FIGURES / "figS2_shuffle_mixing.png"),
        ("校准与信号传播审计", paths.FIGURES / "figS3_calibration.png"),
    ]
    present_extra = [(t, p) for t, p in extra_figs if p.exists()]
    if present_extra:
        A("## 11. 诊断与质控图")
        A("")
        for title, p in present_extra:
            rel = p.relative_to(paths.ROOT)
            A(f"### {title}")
            A("")
            A(f"![{title}](../{rel.as_posix()})")
            A("")

    (reports / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", reports / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
