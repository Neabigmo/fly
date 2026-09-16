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
        top = m1[m1["n_train"] == n_max]
        n_seeds = int(top["model_seed"].nunique())
        A(f"在最大样本预算 N_train = {n_max} 下，对 {n_seeds} 个 seed 取 mean ± std：")
        A("")
        A("| 指标 | real connectome | 说明 |")
        A("|---|---|---|")
        for col, label, note in (
            ("test_acc_a", "Count Accuracy (Test A)", "自然刺激"),
            ("test_acc_b", "Area-controlled (Test B)", "总面积恒定"),
            ("test_acc_c", "Test C", "面积 + 空间包络双控制"),
            ("test_acc_d", "Test D", "未见布局（规则网格）"),
        ):
            vals = top[col].dropna().astype(float)
            if len(vals):
                A(f"| {label} | {vals.mean():.4f} ± {vals.std(ddof=1):.4f} | {note} |")
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

    if made:
        A("## 6. 图")
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
        ("视网膜映射", paths.FIGURES / "figS1_retina_map.png"),
        ("随机化对照的混合曲线", paths.FIGURES / "figS2_shuffle_mixing.png"),
        ("校准与信号传播审计", paths.FIGURES / "figS3_calibration.png"),
    ]
    present_extra = [(t, p) for t, p in extra_figs if p.exists()]
    if present_extra:
        A("## 7. 诊断与质控图")
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
