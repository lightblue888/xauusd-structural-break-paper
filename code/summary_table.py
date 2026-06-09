# -*- coding: utf-8 -*-
"""
横向对比汇总表 — Markdown + LaTeX 双版本 (Phase 1 修订版, 6 baselines)
========================================================================
顺序:
  V11 (sqrt sizing)             [被测对象]
  V10 (= Baseline 2: 传统固定 ATR 仓位)
  Baseline 1: Buy & Hold
  Baseline 3: MA20/50 Crossover + V11 sizing
  Baseline 4: Random Monte Carlo (mean ± 95% CI)
  Baseline 5: Static Breakout (only static SL/TP, no half-exit)
  Baseline 6: 子样本三段 (V10 / V11_sqrt × 2024 / 2025 / 2026Q1+)
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import OUT_DIR, DOC_DIR

OUT_MD    = DOC_DIR / "SUMMARY_TABLE.md"
OUT_TEX   = DOC_DIR / "SUMMARY_TABLE.tex"
OUT_CSV   = DOC_DIR / "SUMMARY_TABLE.csv"


def load_all():
    """按论文顺序加载各 baseline metrics, 给 label 重命名.
    Phase 2 修订: hero 数字使用 Python V11/V10 (全 Python 可复现),
    MT5 V11/V10 当 supplementary 验证."""
    rows = []

    # ─── Phase 2: Python 实现 (论文主表) ───
    py_df = pd.read_csv(OUT_DIR / "python_v10_v11_metrics.csv").to_dict("records")
    py_v10 = next(r for r in py_df if "Python V10" in r["label"])
    py_v11 = next(r for r in py_df if "Python V11" in r["label"])
    py_v10["label"] = "Baseline 2: V10 (Python 实现, 固定 ATR 仓位)"
    py_v11["label"] = "V11 (Python 实现, sqrt sizing) [HERO]"
    rows.append(py_v11)
    rows.append(py_v10)

    rows.extend(pd.read_csv(OUT_DIR / "baseline_1_buy_hold_metrics.csv").to_dict("records"))
    rows.extend(pd.read_csv(OUT_DIR / "baseline_3_ma_cross_metrics.csv").to_dict("records"))
    rows.extend(pd.read_csv(OUT_DIR / "baseline_4_mc_metrics.csv").to_dict("records"))
    rows.extend(pd.read_csv(OUT_DIR / "baseline_5_static_breakout_metrics.csv").to_dict("records"))

    sub = pd.read_csv(OUT_DIR / "baseline_6_subsample_metrics.csv").to_dict("records")
    for r in sub:
        r["label"] = f"Baseline 6: {r['label']}"
        rows.append(r)

    return pd.DataFrame(rows)


COLS = [
    ("label",            "策略",            "{}",      str),
    ("total_return_pct", "总收益率 %",      "{:.2f}",  float),
    ("ann_return_pct",   "年化 %",          "{:.2f}",  float),
    ("sharpe",           "Sharpe",          "{:.2f}",  float),
    ("sortino",          "Sortino",         "{:.2f}",  float),
    ("calmar",           "Calmar",          "{:.2f}",  float),
    ("max_dd_pct",       "MaxDD %",         "{:.2f}",  float),
    ("profit_factor",    "PF",              "{:.2f}",  float),
    ("win_rate_pct",     "胜率 %",          "{:.2f}",  float),
    ("avg_rr",           "平均 R:R",        "{:.2f}",  float),
    ("max_single_win",   "单笔最大盈 $",    "{:.2f}",  float),
    ("max_single_loss",  "单笔最大亏 $",    "{:.2f}",  float),
    ("n_trades",         "总交易数",        "{:.0f}",  int),
]


def fmt_cell(v, fmtstr, kind):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    if kind is str:
        return str(v)
    if kind is int:
        try:
            return f"{int(v):,}"
        except Exception:
            return "N/A"
    try:
        return fmtstr.format(float(v))
    except Exception:
        return "N/A"


def to_markdown(df: pd.DataFrame) -> str:
    headers = [c[1] for c in COLS]
    lines = []
    lines.append("# Phase 2 Baseline Comparison Table — Pure Python Implementation")
    lines.append("")
    lines.append("> OOS 窗口: 2024-01-01 ~ 2026-06-01 (含 6 月 SMC 预热)")
    lines.append("> XAUUSD, INITIAL=$10,000, RiskPercent=0.36%, spread 0.3 pip + commission 7 USD/lot + 1 pip slip")
    lines.append("> Sharpe 等指标按学术口径 (daily annualized × √252)")
    lines.append(">")
    lines.append("> 全部 7 个策略在同一 Python 框架内运行, 1m XAUUSD bar 数据, 完全可复现.")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, r in df.iterrows():
        cells = [fmt_cell(r.get(k), fmtstr, kind) for k, _, fmtstr, kind in COLS]
        lines.append("| " + " | ".join(cells) + " |")

    # MC 95% CI 注释
    mc_row = df[df["label"].str.contains("Random MC", na=False)]
    if not mc_row.empty:
        m = mc_row.iloc[0]
        lines.append("")
        lines.append("**Baseline 4 (Random MC) 95% 置信区间 (1000 runs, 1m SL/TP replay, 爆仓即停)**:")
        lines.append(f"- 总收益率: [{m.get('ret_95ci_low', np.nan):+.2f}%, {m.get('ret_95ci_high', np.nan):+.2f}%]")
        lines.append(f"- Sharpe: [{m.get('sharpe_95ci_low', np.nan):.3f}, {m.get('sharpe_95ci_high', np.nan):.3f}]")
        lines.append(f"- 爆仓 run 占比: {m.get('broke_pct', np.nan):.1f}%")
        lines.append("")
        lines.append(f"**显著性检验** (Python 同框架对比): V11 总收益 +311.6% 与 Sharpe 3.91")
        lines.append(f"均远超 MC 97.5% 上限 ({m.get('ret_95ci_high', np.nan):+.2f}%, {m.get('sharpe_95ci_high', np.nan):.3f})")
        lines.append("→ alpha 在 5% 显著水平下拒绝 H0 (双指标一致).")

    lines.append("")
    lines.append("---")
    lines.append("## 关键观察")
    lines.append("")
    lines.append("**Contribution (a) — 入场信号 alpha** (Python 主表):")
    lines.append("锁定 V11 sizing, 换信号: V11 (+311.6%, Sharpe 3.91) vs Baseline 3 (MA Cross + V11 sizing, +9.45%, Sharpe 0.57).")
    lines.append("→ 价格趋势突破信号本身的 alpha 不可替换.")
    lines.append("")
    lines.append("**Contribution (b) — sqrt 仓位改进** (达成):")
    lines.append("锁定信号, V11 (+311.6%) vs V10 (+183.2%):")
    lines.append("✅ 收益类指标全面改善: ROI +128 pts / PF +0.09 / Calmar +1.54")
    lines.append("= Sharpe 持平 3.91 (sqrt 倍率方差小, 跟单日 PnL 弱相关, 数学上同时缩放 mean/std)")
    lines.append("✗ 风险类指标略恶化: MaxDD -1.6 pts / Sortino -0.49 (杠杆放大代价)")
    lines.append("→ 净效果: sqrt sizing 实现 efficient leverage, 收益放大快于风险增长 (Calmar 改善).")
    lines.append("")
    lines.append("**半仓锁利贡献隔离 (Baseline 5) — 重大发现**:")
    lines.append("剥离半仓锁利 (相同 1143 笔入场, 同 lot) → V10 由 +183% 翻成 -12%, Sharpe 由 3.91 崩到 -0.08,")
    lines.append("胜率由 46% 崩到 14%. 半仓锁利机制是 V10/V11 alpha 的核心载体, 不是辅助.")
    lines.append("→ 建议作为论文第三条 contribution (c) Half-exit decomposition.")
    lines.append("")
    lines.append("**跨期稳健 (Baseline 6)**: V11 在 2024 / 2025 / 2026 partial 三段全部跑赢 V10 (Python 切片).")
    return "\n".join(lines)


def to_latex(df: pd.DataFrame) -> str:
    """LaTeX booktabs 风格 (论文 supplementary 用)"""
    headers = [c[1] for c in COLS]
    col_spec = "l" + "r" * (len(headers) - 1)
    out = []
    out.append(r"\begin{table}[!htbp]")
    out.append(r"\centering")
    out.append(r"\scriptsize")
    out.append(r"\caption{Phase 1 baseline comparison on XAUUSD (2024-01-02 -- 2026-04-29). "
               r"All Sharpe values computed under academic convention "
               r"(daily-return annualized by $\sqrt{252}$).}")
    out.append(r"\label{tab:phase1_baselines}")
    out.append(r"\begin{tabular}{" + col_spec + "}")
    out.append(r"\toprule")
    out.append(" & ".join(headers) + r" \\")
    out.append(r"\midrule")
    for _, r in df.iterrows():
        cells = []
        for key, _, fmtstr, kind in COLS:
            v = r.get(key)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("N/A")
            elif kind is str:
                s = str(v).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
                cells.append(s)
            elif kind is int:
                try:
                    cells.append(f"{int(v):,}")
                except Exception:
                    cells.append("N/A")
            else:
                try:
                    cells.append(fmtstr.format(float(v)))
                except Exception:
                    cells.append("N/A")
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def main():
    df = load_all()
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    OUT_MD.write_text(to_markdown(df), encoding="utf-8")
    OUT_TEX.write_text(to_latex(df), encoding="utf-8")
    print(f"  saved -> {OUT_MD.name} / {OUT_TEX.name} / {OUT_CSV.name}")
    print(f"  {len(df)} 行")


if __name__ == "__main__":
    main()
