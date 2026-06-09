# -*- coding: utf-8 -*-
"""
汇总可视化 (Phase 1 修订版, 6 baselines):
  Fig 1: 累计收益曲线 (V11 / V10=Baseline2 / B1 / B3 / B5 叠加)
  Fig 2: Drawdown underwater plot
  Fig 3: Baseline 4 (Random MC) 直方图 + V11 / V10 红线
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import OUT_DIR, FIG_DIR, STYLE, INITIAL_BALANCE

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

FIG_EQUITY     = FIG_DIR / "fig_1_cumulative_returns.png"
FIG_UNDERWATER = FIG_DIR / "fig_2_drawdown_underwater.png"
FIG_MC         = FIG_DIR / "fig_3_mc_distribution.png"


def style_ax(ax, title=""):
    ax.set_facecolor(STYLE["ax"])
    ax.tick_params(colors=STYLE["text"], labelsize=9)
    for s in ax.spines.values():
        s.set_color(STYLE["grid"])
    ax.grid(True, color=STYLE["grid"], alpha=0.4, linewidth=0.5)
    if title:
        ax.set_title(title, color=STYLE["text"], fontsize=12, fontweight='bold', pad=10)
    ax.xaxis.label.set_color(STYLE["text"])
    ax.yaxis.label.set_color(STYLE["text"])


STRATS = [
    ("V11 Python (sqrt sizing) [HERO]",   "python_v11_equity.csv",                STYLE["yellow"], 2.6),
    ("Baseline 2: V10 Python",             "python_v10_equity.csv",                STYLE["blue"],   1.8),
    ("Baseline 1: Buy & Hold",             "baseline_1_buy_hold_equity.csv",       STYLE["orange"], 1.6),
    ("Baseline 3: MA Cross + V11 sizing",  "baseline_3_ma_cross_equity.csv",       STYLE["loss"],   1.4),
    ("Baseline 5: Static Breakout",        "baseline_5_static_breakout_equity.csv", STYLE["purple"], 1.6),
]


def load_equity_series():
    series = []
    for label, fname, color, lw in STRATS:
        path = OUT_DIR / fname
        if not path.exists():
            print(f"  skip (no file): {fname}")
            continue
        df = pd.read_csv(path, encoding="utf-8")
        df["Date"] = pd.to_datetime(df["Date"])
        s = df.set_index("Date")["equity"].sort_index()
        ts0 = s.index.min() - pd.Timedelta(days=1)
        s = pd.concat([pd.Series([INITIAL_BALANCE], index=[ts0]), s])
        series.append((label, s, color, lw))
    return series


def plot_cumulative(series):
    fig, ax = plt.subplots(figsize=(15, 7), facecolor=STYLE["bg"])
    style_ax(ax, "Phase 2 — 累计资金曲线对比 (initial $10,000, 28-mo OOS)")
    for label, s, color, lw in series:
        ax.plot(s.index, s.values, color=color, linewidth=lw, label=label, alpha=0.95)
    ax.axhline(INITIAL_BALANCE, color=STYLE["text"], linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_ylabel("账户余额 USD", color=STYLE["text"])
    ax.set_xlabel("日期")
    ax.legend(facecolor=STYLE["ax"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"],
              fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_EQUITY, dpi=130, facecolor=STYLE["bg"], bbox_inches="tight")
    print(f"  saved -> {FIG_EQUITY.name}")
    plt.close(fig)


def plot_underwater(series):
    fig, ax = plt.subplots(figsize=(15, 6), facecolor=STYLE["bg"])
    style_ax(ax, "Phase 2 — Drawdown Underwater Plot (低 = 回撤越深)")
    for label, s, color, lw in series:
        peak = s.cummax()
        dd_pct = (s - peak) / peak * 100
        ax.fill_between(dd_pct.index, dd_pct.values, 0, color=color, alpha=0.15)
        ax.plot(dd_pct.index, dd_pct.values, color=color, linewidth=lw, label=label)
    ax.axhline(0, color=STYLE["text"], linestyle="-", alpha=0.5, linewidth=0.6)
    ax.set_ylabel("回撤 %", color=STYLE["text"])
    ax.set_xlabel("日期")
    ax.legend(facecolor=STYLE["ax"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"],
              fontsize=10, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_UNDERWATER, dpi=130, facecolor=STYLE["bg"], bbox_inches="tight")
    print(f"  saved -> {FIG_UNDERWATER.name}")
    plt.close(fig)


def plot_mc_distribution():
    runs = pd.read_csv(OUT_DIR / "baseline_4_mc_runs.csv", encoding="utf-8")
    # 用 Python V11/V10 标记到 MC 分布上 (跟 baseline 同框架公平对比)
    v = pd.read_csv(OUT_DIR / "python_v10_v11_metrics.csv", encoding="utf-8")
    v11 = v[v["label"].str.contains("Python V11")].iloc[0]
    v10 = v[v["label"].str.contains("Python V10")].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=STYLE["bg"])

    style_ax(ax1, "Baseline 4: 1000-run Random MC — 总收益分布 (1m SL/TP replay, 爆仓即停)")
    clip_max = max(runs["total_return_pct"].quantile(0.99), v11["total_return_pct"]) + 20
    ax1.hist(runs["total_return_pct"].clip(-110, clip_max), bins=60,
             color=STYLE["purple"], alpha=0.7, edgecolor=STYLE["grid"])
    ax1.axvline(v11["total_return_pct"], color=STYLE["yellow"], linewidth=2.5,
                label=f"V11 sqrt: +{v11['total_return_pct']:.1f}%")
    ax1.axvline(v10["total_return_pct"], color=STYLE["blue"], linewidth=2,
                label=f"V10: +{v10['total_return_pct']:.1f}%")
    ax1.axvline(runs["total_return_pct"].quantile(0.975), color=STYLE["loss"],
                linestyle="--", linewidth=1.4, label="MC 97.5%")
    ax1.axvline(runs["total_return_pct"].quantile(0.025), color=STYLE["loss"],
                linestyle="--", linewidth=1.4, label="MC 2.5%")
    ax1.set_xlabel("总收益率 %")
    ax1.set_ylabel("频数")
    ax1.legend(facecolor=STYLE["ax"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"], fontsize=9)

    style_ax(ax2, "Baseline 4: 1000-run Random MC — Sharpe 分布")
    ax2.hist(runs["sharpe"], bins=60, color=STYLE["purple"], alpha=0.7,
             edgecolor=STYLE["grid"])
    ax2.axvline(v11["sharpe"], color=STYLE["yellow"], linewidth=2.5,
                label=f"V11 sqrt: {v11['sharpe']:.2f}")
    ax2.axvline(v10["sharpe"], color=STYLE["blue"], linewidth=2,
                label=f"V10: {v10['sharpe']:.2f}")
    ax2.axvline(runs["sharpe"].quantile(0.975), color=STYLE["loss"],
                linestyle="--", linewidth=1.4, label="MC 97.5%")
    ax2.set_xlabel("Sharpe (daily annualized × √252)")
    ax2.set_ylabel("频数")
    ax2.legend(facecolor=STYLE["ax"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"], fontsize=9)

    fig.suptitle("Phase 2 — Random MC vs V10 / V11 (alpha 显著性检验, p<0.025 双指标)",
                 color=STYLE["text"], fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_MC, dpi=130, facecolor=STYLE["bg"], bbox_inches="tight")
    print(f"  saved -> {FIG_MC.name}")
    plt.close(fig)


def main():
    series = load_equity_series()
    plot_cumulative(series)
    plot_underwater(series)
    plot_mc_distribution()


if __name__ == "__main__":
    main()
