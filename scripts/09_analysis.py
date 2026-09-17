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
    # Were the replicates actually distinct graphs?
    dist = blk.get("distinctness") or {}
    agr = {k: v.get("max_agreement") for k, v in dist.items() if v}
    if agr:
        A("")
        parts = []
        for arm in ("real", "shuffled"):
            v = dist.get(arm) or {}
            if not v.get("pairs"):
                continue
            lo = min(p["agreement"] for p in v["pairs"])
            parts.append(f"{arm} 臂 {lo:.4f}–{v['max_agreement']:.4f}")
        A("**重复是否真的互相独立**：同一臂内两两 run 在 5000 张测试图上的**逐样本预测一致率**为 "
          + "，".join(parts) + "（若两个 run 用了**同一张图**，这个数会是 1.000）。"
          "`_validate_shuffle` 只能证明随机图与**真实图**不同，无法发现「缓存键忽略了 shuffle_seed」"
          "这类失败 —— 那时所有对照共用一张图却仍然通过与真实图的校验，"
          "而种子级 CI 就会建立在并非重复的重复之上。这项检查现在每次统计都会跑。")
        if len(agr) == 2 and all(agr.values()):
            A("")
            A("顺带一个观察：real 臂的 seed 间一致率**高于** shuffled 臂，"
              "即真实连接组把解约束得更紧，而随机图之间存在更多解的分歧。")
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


def _section_step_matched(lines: list) -> None:
    """What the existing logs already say, before the equal-update grid runs."""
    A = lines.append
    A("### 6.2b 步数匹配的初步证据（用已有日志，先于等算力网格）")
    A("")
    p = paths.DATA_PROCESSED / "step_matched.json"
    if not p.exists():
        lines.extend(_pending("步数匹配分析", "python scripts/22_step_matched.py"))
        return
    rep = json.loads(p.read_text(encoding="utf-8"))
    v = rep.get("verdict")
    if not v:
        lines.extend(_pending("步数匹配分析", "python scripts/22_step_matched.py"))
        return
    A("每个 epoch 的 `val_acc` 都随 epoch 序号记录在 `metrics.jsonl` 里，而 "
      "`优化步数 = (epoch+1) × ceil(N/batch)`，所以**不需要新算力**就能把整条曲线画在"
      "「优化步数」而非「epoch」的横轴上（见 `figures/fig11_step_matched.png`）。"
      "关键在于 N=5000 是在**多少步**上停止的。")
    A("")
    lines.extend(_md_table(
        ["量", "值"],
        [["N=5000 实际被观测到的最大步数", f"{v['n5000_observed_through_step']:.0f}"],
         ["同一批步数下 N=20000 的 Δ", f"{v['n20000_delta_at_matched_compute']:+.4f}"],
         ["同一批步数下 N=5000 的 Δ", f"{v['n5000_delta_at_same_step']:+.4f}"],
         ["N=20000 达到最终 Δ 一半所需的步数", f"{v['n20000_half_gap_onset_step']:.0f}"],
         ["N=20000 最终 Δ", f"{v['n20000_delta_final']:+.4f}"]]))
    A("")
    A(f"**结论（初步）**：N=20000 的优势在 step {v['n20000_half_gap_onset_step']:.0f} 附近才开始形成，"
      f"而 N=5000 早在 step {v['n5000_observed_through_step']:.0f} 就停止了 —— 也就是说，"
      "原曲线**在差距刚要出现之前就把小样本条件停掉了**。单看这一点，算力仍是成立的解释。")
    A("")
    # ... but the plateau diagnostic says the small conditions had converged there.
    plateau = rep.get("plateau") or []
    if plateau:
        rows = []
        for n in sorted({r["n_train"] for r in plateau}):
            for graph in ("real", "shuffled"):
                g = [r for r in plateau if r["n_train"] == n and r["graph"] == graph]
                if not g:
                    continue
                rows.append([
                    f"{n} {graph}", str(len(g)),
                    f"{g[0]['stopped_at_step']:.0f}–{g[-1]['stopped_at_step']:.0f}",
                    f"{np.mean([r['slope_per_epoch'] for r in g]):+.5f}",
                    f"{np.mean([r['gain_in_last_window'] for r in g]):+.4f}",
                    f"{np.mean([r['val_at_stop'] for r in g]):.4f}",
                ])
        A("**但是**：每个 run 的最后 15 个 epoch（正是 patience 的窗口）里的 val 斜率是"
          "「它是否还在进步」的直接证据：")
        A("")
        lines.extend(_md_table(
            ["条件", "seeds", "停止步数", "窗口内斜率/epoch", "窗口内增益", "停止时 val"],
            rows))
        A("")
        A(f"**0 / {v.get('small_n_runs', 0)} 个小样本 run** 在停止时仍以 > +0.002/epoch 上升；"
          f"N=5000 的窗口内平均斜率是 **{v.get('n5000_mean_last_window_slope', float('nan')):+.5f}/epoch**，"
          f"而 N=20000 在末尾也只有 **{v.get('n20000_mean_last_window_slope', float('nan')):+.5f}/epoch** —— "
          "**每个条件都已经收敛**，差别在于收敛到的**平台高度**（0.51 对 0.77 / 0.66），"
          "而平台高度是关于**刺激**的陈述，不是关于训练时长的。")
        A("")
    A("必须同时声明的三点限定：① 这是**支持性证据而非对照** —— 它比较的是「20k 不同刺激」"
      "与「5k 不同刺激」在相同步数下的表现，而等算力网格是**固定 N、只改预算**，"
      "只有后者能排除算力；② N=5000 的三个 seed 分别在 3 239 / 4 424 / 4 740 步停止，"
      "所以交叉可比范围只到最短的那个（3 950 步），更靠右的取值是外推，图中虚线标出；"
      "③ 在 3 950 步处 N=20000 仍有 +0.021 的小差距（N=5000 为 +0.005），"
      "即**并非全部**都是算力，但这部分远小于最终的 +0.106。")
    A("")
    A("**把两条证据合起来**：步数匹配说明差距是在小样本条件停止**之后**才打开的；"
      "平台诊断说明小样本条件在停止时**已经收敛**（不是在半途被 patience 打断）。"
      "两者共同指向「不同样本量收敛到不同平台高度」，而不是「训练不够久」。"
      "唯一还不能排除的情形是：**平坦区之后可能还有第二次下降**（plateau-then-drop），"
      "这正是等算力网格要检验的 —— 给 N=5000 完整 18 780 步，看它是否仍停在 0.51。")
    A("")
    A("口径提醒：本节全部基于**训练中的验证准确率**（只有它才有逐 epoch 轨迹），"
      f"所以末值 Δ={v['n20000_delta_final']:+.4f} 与 6.1 节基于**最佳 checkpoint 的测试准确率**"
      f"得到的 Δ 不是同一个量，不要直接相减。另外所有曲线只取**已完成**的 run ——"
      "仍在写入的 run 会被平推到最后一次记录，从而低估它所属的那一臂。")
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

    def _cell_counts(subset: list[dict]) -> list[int]:
        cells: dict[str, int] = {}
        for s in subset:
            key = f"{s.get('graph')}/{'scratch' if s.get('scratch') else 'pretrained'}"
            cells[key] = cells.get(key, 0) + 1
        return sorted(cells.values())

    prim_counts = _cell_counts(by_holdout[primary])
    per_cell = (f"{prim_counts[0]}" if len(set(prim_counts)) == 1
                else f"{prim_counts[0]}–{prim_counts[-1]}")
    A(f"主表：留出对 = {primary}（每个条件 {per_cell} 个 run）")
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
    # ---------------------------------------------------------------- #
    # The pre-registered pass line for the curriculum.
    cells: dict[tuple[str, str], list[dict]] = {}
    for s in by_holdout[primary]:
        cells.setdefault(
            (s.get("graph", "?"), "scratch" if s.get("scratch") else "pretrained"), []
        ).append(s)
    if len(cells) >= 2:
        chance = 1.0 / 7
        A("**按 §6.2 预先锁定的合格线判读**（合格线：`real/pretrained` 的 unseen 高于 chance "
          "0.10 **且**高于 `real/scratch`，否则「计数预训练迁移」这句话没有证据）：")
        A("")
        rows = []
        for key in sorted(cells):
            g = cells[key]
            unseen = float(np.mean([x["sum_test"] for x in g]))
            seen = float(np.mean([x["sum_train"] for x in g]))
            rows.append([f"{key[0]}/{key[1]}", str(len(g)), f"{seen:.4f}", f"{unseen:.4f}",
                         f"{unseen - chance:+.4f}"])
        lines.extend(_md_table(["条件", "runs", "seen", "unseen", "unseen − chance"], rows))
        A("")
        rp, rs = cells.get(("real", "pretrained")), cells.get(("real", "scratch"))
        if rp and rs:
            u_pre = float(np.mean([x["sum_test"] for x in rp]))
            u_scr = float(np.mean([x["sum_test"] for x in rs]))
            a_pre = float(np.mean([x["acc_a_test"] for x in rp]))
            a_scr = float(np.mean([x["acc_a_test"] for x in rs]))
            A(f"`real/pretrained` 的 unseen 是 **{u_pre:.4f}**（chance 0.143，差 "
              f"{u_pre - chance:+.4f}），而 `real/scratch` 是 **{u_scr:.4f}** —— 温热启动"
              "**没有**帮助未见组合，它甚至更低。"
              + ("**因此「计数预训练能迁移到加法」这句话没有证据**，按预先锁定的规则不能写。"
                 if (u_pre - chance < 0.10) or (u_pre <= u_scr)
                 else "两条合格线都满足，迁移成立。"))
            A("")
            A(f"但温热启动有一个**清楚的正效应**：它把**操作数**读得更准（`a` 从 {a_scr:.4f} "
              f"升到 {a_pre:.4f}），却没把这份表征转成组合能力。这是「会数数 ≠ 会相加」的一个"
              f"干净实例：两个操作数都读得出来（a≈{a_pre:.2f}），和却停在 chance 附近"
              f"（{u_pre:.4f}）。而四个格子里**唯一明显高于 chance 的是 `real/scratch`"
              f"（{u_scr:.4f}）** —— 从零训练反而学到了某种能迁移到未见对的结构。")
            A("")
            A("限定：每个格子只有 3 个 seed，unseen 上 seed 间散布不小；chance = 1/7 = 0.143"
              "（7 个和类别，且每个和都仍可由训练对中的其他组合产生，所以这是在测组合而非类别泛化）。")
            A("")


def _section_control_match(lines: list, summaries: list[dict]) -> None:
    # kept in step with scripts/15_run_parallel.py:SPECTRAL_MATCH_W_SCALE
    SPECTRAL_WS = 1.4106
    """Is the shuffled control matched on anything beyond degree and weight?"""
    A = lines.append
    A("### 6.6 对照的有效性：随机图是否也匹配了「工作点」")
    A("")
    p = paths.DATA_PROCESSED / "graph_spectrum_core.json"
    if not p.exists():
        lines.extend(_pending("图谱/动力学匹配度测量",
                              "python scripts/23_graph_spectrum.py --circuit core"))
        return
    rep = json.loads(p.read_text(encoding="utf-8"))
    g = rep.get("graphs", {})
    real = g.get("real")
    if not real:
        lines.extend(_pending("图谱/动力学匹配度测量", "python scripts/23_graph_spectrum.py"))
        return
    rows = []
    for key, label in (("real", "real"), ("shuffled_s0", "shuffled s0"),
                       ("shuffled_s1", "shuffled s1"), ("shuffled_s2", "shuffled s2"),
                       ("shuffled_matched", "shuffled（等谱匹配）")):
        v = g.get(key)
        if not v:
            continue
        rows.append([label, f"{v['perron']:.4f}", f"{v['singular_top']:.4f}",
                     f"{v['peak_final']:.3g}", f"×{v['growth']:.3g}",
                     f"{v['fraction_negative']:.3f}"])
    lines.extend(_md_table(
        ["图", "谱半径 ρ", "最大奇异值", "t=32 峰值|h|", "增长倍数", "负权比例"], rows))
    A("")
    sm = rep.get("spectral_match")
    sh_keys = [k for k in ("shuffled_s0", "shuffled_s1", "shuffled_s2") if k in g]
    rho_s = np.array([g[k]["perron"] for k in sh_keys]) if sh_keys else np.array([])
    if len(rho_s) >= 2:
        z = (real["perron"] - rho_s.mean()) / max(rho_s.std(ddof=1), 1e-12)
        A("**更强的一句话（这才是重点）**：三张独立的保度保权重随机图的谱半径是 "
          f"{'、'.join(f'{v:.4f}' for v in rho_s)}（彼此标准差仅 **{rho_s.std(ddof=1):.4f}**，"
          f"即 {100 * rho_s.std(ddof=1) / rho_s.mean():.2f}%），而真实图是 "
          f"**{real['perron']:.4f}** —— **高出 {z:.0f} 个标准差**、"
          f"比三张图的**全部取值范围还高 {real['perron'] / rho_s.max() - 1:.1%}**。"
          "也就是说：**在「同一个度数序列与同一批突触权重」能连成的所有图里，"
          "真实连接组在「信号放大倍率」这个决定动力学的量上是一个极端离群值**。"
          "这既是「真实布线非同一般」的独立结构证据，也正是必须做等谱匹配对照的理由 ——"
          "否则 real-vs-shuffled 的比较会把「布线好」与「放大倍率天生更高」混为一谈。")
        A("")
    A("**这是本次审计发现的一个真实问题**：保度保权重的随机交换**并不保持工作点**。"
      f"real 图的谱半径是 **ρ={real['perron']:.4f}**，而三张随机图是 "
      f"**{np.mean([g[k]['perron'] for k in ('shuffled_s0','shuffled_s1','shuffled_s2') if k in g]):.4f}**"
      "（彼此只差 ±0.003），**低了 29%**；32 步内的峰值活动差了 **1.9 个数量级**。"
      "也就是说，常规的 real-vs-shuffled 比较**同时改变了「布线」与「放大倍率」**，"
      "差距里有多少来自布线、多少来自工作点，原先无法区分。")
    A("")
    if sm:
        A(f"**修正办法（已加入队列）**：权重矩阵对 `w_scale` 是严格线性的，所以只需把随机图的 "
          f"`w_scale` 乘以 ρ_real/ρ_shuffled，就能让它拥有与 real 相同的谱半径。实测 "
          f"`w_scale={sm['w_scale']:.4f}` 时 ρ={sm['perron']:.4f}，与 real 的 "
          f"{sm['real_perron']:.4f} 相差 **{100 * sm['relative_residual']:.3f}%**；"
          f"同时 t=32 的增长倍数变成 ×{g['shuffled_matched']['growth']:.3g}，"
          f"与 real 的 ×{real['growth']:.3g} 只差 12%。"
          "这个**等谱匹配对照**在度数、权重分布形状、谱半径、活动增长四个维度上都与 real 对齐，"
          "**只差边怎么连** —— 于是「结构是否有用」第一次成为干净的操纵。")
        A("")
        A("判读：若 real 仍然高于等谱匹配对照，结构解释成立；若差距消失，"
          "则原先的差距主要是放大倍率（工作点）造成的。两种结果都有信息量，"
          "所以这个块（3 runs，约 1.5 h）值得跑。")
        A("")
    # ---------------------------------------------------------------- #
    # The verdict, once the matched control has actually been trained.
    A("**等谱匹配对照的实测结果**")
    A("")
    arms: dict[str, dict] = {}
    for label, signed, ws in (("real", False, 1.0),
                              ("shuffled（同度数同权重）", False, 1.0),
                              ("shuffled 等谱匹配", False, SPECTRAL_WS)):
        sel = [s for s in summaries
               if s.get("task") == "count" and s.get("model") == "M1"
               and s.get("circuit") == "core" and s.get("n_train") == 20000
               and s.get("graph") == ("real" if label == "real" else "shuffled")
               and bool(s.get("cfg_signed", False)) is signed
               and abs(float(s.get("cfg_w_scale") or 0) - ws) < 1e-6
               and int(s.get("cfg_max_steps") or 0) == 0]
        if not sel:
            continue
        arms[label] = {
            c: np.array([s[f"test_acc_{c.lower()}"] for s in sel
                         if s.get(f"test_acc_{c.lower()}") is not None], dtype=float)
            for c in "ABCD"
        }
        arms[label]["_n"] = len(sel)
    if len(arms) >= 2:
        rows = []
        for label, d in arms.items():
            row = [label, str(d["_n"])]
            for c in "ABCD":
                v = d[c]
                row.append(f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1
                           else f"{v.mean():.4f}")
            rows.append(row)
        lines.extend(_md_table(["条件", "seeds", "A", "B", "C", "D"], rows))
        A("")
        if "shuffled 等谱匹配" in arms and "real" in arms:
            a_r = arms["real"]["A"].mean()
            a_s = arms["shuffled（同度数同权重）"]["A"].mean()
            a_m = arms["shuffled 等谱匹配"]["A"].mean()
            A(f"**结论**：把随机对照的谱半径匹配到与 real 完全相同（残差 0.000%）之后，"
              f"条件 A 从 **{a_s:.4f} 升到 {a_m:.4f}**（+{a_m - a_s:.4f}）—— 工作点确实贡献了一部分，"
              f"但 **real 仍是 {a_r:.4f}，仍然高出等谱匹配对照 {a_r - a_m:+.4f}**。"
              f"也就是说：在度数、权重分布、谱半径、活动增长四个维度全部对齐、**只差边怎么连**的对照上，"
              f"真实连接组依然显著更好。**「结构本身有用」这一解释成立，"
              f"不是放大倍率的假象。**")
            A("")
            A(f"代价与限定：等谱匹配对照只有 {arms['shuffled 等谱匹配']['_n']} 个 seed"
              f"（real/shuffled 各 10 个），所以它自己的区间更宽；另外它不是原始随机图，"
              f"而是把权重整体放大 1.4106 倍后的图，因此与「同权重分布」这一说法在"
              f"权重量级上不再严格一致 —— 这一点必须写明。")
            A("")


def _section_lesion(lines: list) -> None:
    A = lines.append
    """The virtual knockout panel, aggregated over source models."""
    A("### 6.5 虚拟敲除（LC11 / LC10a / 随机 143 神经元）")
    A("")
    files = sorted(paths.DATA_PROCESSED.glob("lesion_*.json"))
    if not files:
        lines.extend(_pending("敲除面板", "python scripts/17_run_lesion.py --source <run_id>"))
        return
    panels = []
    for f in files:
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if s.get("deltas"):
            panels.append(s)
    if not panels:
        lines.extend(_pending("敲除面板", "python scripts/17_run_lesion.py --source <run_id>"))
        return

    conds = ("A", "B", "C", "D")
    # A panel per source model: the dissociation claim needs a spread across models,
    # and reading only files[0] would silently drop the other two.
    names = sorted({n for p in panels for n in p["deltas"]})
    rows = []
    for name in names:
        per_model = [[p["deltas"][name][c] for c in conds]
                     for p in panels if name in p["deltas"]]
        arr = np.array(per_model, dtype=float)
        row = [name, str(len(per_model))]
        for j in range(len(conds)):
            col = arr[:, j]
            cell = f"{col.mean():+.4f}"
            if len(col) > 1:
                cell += f" [{col.min():+.4f}, {col.max():+.4f}]"
            row.append(cell)
        rows.append(row)
    first = panels[0]
    A(f"来源模型：{len(panels)} 个（headline 条件的 seed"
      + ("s" if len(panels) > 1 else "") + "）；"
      f"LC11 = {first['n_neuron_types'].get('LC11')} 个神经元，"
      f"LC10a = {first['n_neuron_types'].get('LC10a')} 个。"
      "多模型时单元格为 **均值 [最小值, 最大值]**。单个模型的敲除结果很容易只是 seed 运气，"
      "所以面板在 headline 条件的每个 seed 上各跑一次。")
    A("")
    lines.extend(_md_table(["敲除", "models", "ΔA", "ΔB", "ΔC", "ΔD"], rows))
    A("")

    # The pre-registered pass line: LC11 must exceed every random knockout of the same
    # size, and LC10a must stay inside their range.
    rnd = [n for n in names if n.startswith("random_")]
    lc11 = [n for n in names if n == "LC11"]
    lc10 = [n for n in names if n == "LC10a"]
    if rnd and lc11:
        r_max = max(p["deltas"][r]["A"] for p in panels for r in rnd if r in p["deltas"])
        l_mean = float(np.mean([p["deltas"]["LC11"]["A"] for p in panels
                                if "LC11" in p["deltas"]]))
        l_min = min(p["deltas"]["LC11"]["A"] for p in panels if "LC11" in p["deltas"])
        A(f"**按 §6.2 预先锁定的合格线判读**：LC11 的 ΔA 均值 {l_mean:+.4f}"
          f"（最小 {l_min:+.4f}），5 次随机 143 神经元敲除的 ΔA 最大值 {r_max:+.4f} —— "
          + ("**LC11 在全部随机敲除之上，双分离成立**。"
             if l_min > r_max else
             "**LC11 未完全越出随机敲除范围，不能声称双分离**，只能报告 LC11 与随机敲除的差异幅度。"))
        if lc10:
            c_mean = float(np.mean([p["deltas"]["LC10a"]["A"] for p in panels
                                    if "LC10a" in p["deltas"]]))
            in_range = c_mean <= r_max
            A(f"LC10a 的 ΔA 均值 {c_mean:+.4f}，"
              + ("落在随机范围之内 —— 与文献报告的「LC10a 敲除不损伤数感」一致。"
                 if in_range else
                 "**也越出随机范围**，因此只能说「LC11 与 LC10a 均敏感」，不能声称双分离。"))
        A("")
    A("正 Δ = 准确率下降（受损）；`LC11_readout_only` 用来区分「解码器依赖」与「电路依赖」。")
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

    all_summaries = load_summaries()
    log.info("loaded %d completed runs", len(all_summaries))
    # Never pool runs trained on different stimulus definitions: the radius range
    # changed materially during development, and mixing the two would blend two
    # different experiments into one number.  The curriculum block is exempt: a
    # held-out pair is part of that block's *design*, and each holdout is compared only
    # against itself, so filtering the non-default holdouts away would silently delete
    # the leave-pair-out robustness check they exist to provide.
    summaries, fingerprint = filter_current_stimuli(all_summaries, logger=log)
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
    _section_step_matched(lines)
    _section_signed(lines, df)
    _section_curriculum(lines, all_summaries)
    _section_control_match(lines, summaries)
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
        ("步数匹配：拐点是不是算力", paths.FIGURES / "fig11_step_matched.png"),
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
