# -*- coding: utf-8 -*-
"""
统一 metrics 计算 — 所有 baseline 必须用这个模块出结果。
- 输入: daily_equity (必填) + trades_df (可选)
- 输出: dict 含 10 个指标
- 设计为 Buy & Hold (1 trade) 也能跑：trade 级指标缺失时填 NaN
"""
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(
    daily_equity: pd.Series,
    trades_pnl: pd.Series = None,
    trades_rr: pd.Series = None,
    label: str = "strategy",
    initial_balance: float = None,
) -> dict:
    """
    Args:
        daily_equity:    index=date, value=账户余额（必填，至少 2 个点）
        trades_pnl:      每笔交易 PnL（USD），可选
        trades_rr:       每笔交易 R:R 比，可选
        label:           策略名称（透传到 dict）
        initial_balance: 真实起点资金 (USD)。若提供, 在最早交易日前 1 天插入
                         一个 INITIAL 数据点, 让 Sharpe / max_dd / 总收益率
                         基于真正起点计算 (避免首日 PnL 把"初始"拉偏)。

    Returns:
        dict: 10 指标 + label + initial_balance + final_balance
    """
    eq = daily_equity.copy().sort_index()
    eq = eq[~eq.index.duplicated(keep="last")]
    if initial_balance is not None:
        ts0 = pd.Timestamp(eq.index.min()) - pd.Timedelta(days=1)
        eq = pd.concat([pd.Series([float(initial_balance)], index=[ts0]), eq]).sort_index()
    if len(eq) < 2:
        raise ValueError(f"[{label}] daily_equity 必须 >= 2 个点")

    initial = float(eq.iloc[0])
    final   = float(eq.iloc[-1])
    total_return_pct = (final / initial - 1) * 100

    # ---- 年化 ----
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25
    # MC 时偶尔出现 final<=0 -> 复数, 这里截断到 -99.99%
    if years > 0 and final > 0 and initial > 0:
        ann_return_pct = ((final / initial) ** (1 / years) - 1) * 100
    else:
        ann_return_pct = -99.99

    # ---- 日收益序列 ----
    daily_ret = (eq / eq.shift(1) - 1).dropna()
    mean_d = daily_ret.mean()
    std_d  = daily_ret.std(ddof=1)
    sharpe = (mean_d / std_d) * np.sqrt(TRADING_DAYS_PER_YEAR) if std_d > 0 else np.nan

    neg_ret = daily_ret[daily_ret < 0]
    downside_std = neg_ret.std(ddof=1) if len(neg_ret) > 1 else np.nan
    sortino = (mean_d / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR) if downside_std and downside_std > 0 else np.nan

    # ---- 最大回撤 ----
    peak = eq.cummax()
    dd_pct = (eq - peak) / peak * 100
    max_dd_pct = float(dd_pct.min())   # 负数

    calmar = ann_return_pct / abs(max_dd_pct) if max_dd_pct < 0 else np.nan

    # ---- 交易级指标 ----
    n_trades = int(len(trades_pnl)) if trades_pnl is not None else 0
    if n_trades >= 1:
        pnl = pd.to_numeric(trades_pnl, errors="coerce").dropna()
        wins   = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = len(wins) / len(pnl) * 100 if len(pnl) > 0 else np.nan
        profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.nan
        # 单笔最大盈/亏 各自只在对应方向有数据时才算 (修复 B&H 1 笔交易时 max==min 的 bug)
        max_single_win  = float(wins.max())   if len(wins)   > 0 else np.nan
        max_single_loss = float(losses.min()) if len(losses) > 0 else np.nan
    else:
        win_rate = profit_factor = max_single_win = max_single_loss = np.nan

    if trades_rr is not None and len(trades_rr) > 0:
        rr_clean = (pd.to_numeric(trades_rr, errors="coerce")
                      .replace([np.inf, -np.inf], np.nan).dropna())
        avg_rr = float(rr_clean.mean()) if len(rr_clean) else np.nan
    elif n_trades >= 1:
        # 用 |avg_win / avg_loss| 当作 R:R 近似（payoff ratio）
        if len(wins) > 0 and len(losses) > 0 and losses.mean() != 0:
            avg_rr = float(abs(wins.mean() / losses.mean()))
        else:
            avg_rr = np.nan
    else:
        avg_rr = np.nan

    return {
        "label":            label,
        "initial_balance":  initial,
        "final_balance":    final,
        "total_return_pct": round(total_return_pct, 3),
        "ann_return_pct":   round(ann_return_pct, 3),
        "profit_factor":    round(profit_factor, 3) if not np.isnan(profit_factor) else np.nan,
        "sharpe":           round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "sortino":          round(sortino, 3) if not np.isnan(sortino) else np.nan,
        "calmar":           round(calmar, 3) if not np.isnan(calmar) else np.nan,
        "max_dd_pct":       round(max_dd_pct, 3),
        "win_rate_pct":     round(win_rate, 3) if not np.isnan(win_rate) else np.nan,
        "avg_rr":           round(avg_rr, 3) if not np.isnan(avg_rr) else np.nan,
        "max_single_win":   round(max_single_win, 2) if not np.isnan(max_single_win) else np.nan,
        "max_single_loss":  round(max_single_loss, 2) if not np.isnan(max_single_loss) else np.nan,
        "n_trades":         n_trades,
    }


METRIC_LABELS = [
    ("label",            "策略"),
    ("total_return_pct", "总收益率 %"),
    ("ann_return_pct",   "年化收益率 %"),
    ("profit_factor",    "盈利因子 PF"),
    ("sharpe",           "Sharpe"),
    ("sortino",          "Sortino"),
    ("calmar",           "Calmar"),
    ("max_dd_pct",       "最大回撤 %"),
    ("win_rate_pct",     "胜率 %"),
    ("avg_rr",           "平均 R:R"),
    ("max_single_win",   "最大单笔盈"),
    ("max_single_loss", "最大单笔亏"),
    ("n_trades",         "总交易数"),
]


def print_metrics(m: dict):
    print(f"\n=== {m['label']} ===")
    print(f"  初始 -> 最终:    ${m['initial_balance']:,.2f} -> ${m['final_balance']:,.2f}")
    for k, name in METRIC_LABELS[1:]:
        v = m[k]
        if isinstance(v, float) and np.isnan(v):
            print(f"  {name:14}  N/A")
        else:
            print(f"  {name:14}  {v}")
