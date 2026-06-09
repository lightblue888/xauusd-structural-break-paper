# -*- coding: utf-8 -*-
"""
Baseline 4 — 等权重随机交易蒙特卡洛 (安慰剂检验, H0)
========================================================================
零假设 H0: 策略业绩与随机入场不可区分。
做法: 在交易日上均匀采样入场时刻 + 随机方向, 用与 V10/V11 完全
      相同的 SL/TP/成本模型, 用 1m 数据精确回放 SL/TP 命中,
      跑 N_MC 次得到指标分布; V11 落在分布右尾外即拒绝 H0。

数据精度局限: 1m 数据仅至 2025-12-31, 窗口缩到 24 个月 (2024-01-02
              ~ 2025-12-31), 对应频率匹配比例缩到 ~1190 笔/run.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (INITIAL_BALANCE, CONTRACT_SIZE, OUT_DIR,
                     MC_RUNS, MC_SEED, round_trip_cost_usd,
                     DATE_START, DATE_END)
from _metrics import compute_metrics
from _bar_replay import load_1m_arrays, replay_sl_tp

# ---- 策略参数 (与 V10/V11 一致) ----
SL_USD = 3.0
TP_USD = 20.0
WINDOW_START = DATE_START   # 2024-01-01
WINDOW_END   = DATE_END     # 2026-06-01

PY_V10_TRADES = OUT_DIR / "python_v10_trades.csv"
OUT_RUNS      = OUT_DIR / "baseline_4_mc_runs.csv"
OUT_METRICS   = OUT_DIR / "baseline_4_mc_metrics.csv"


def n_trades_from_v10():
    """匹配 Python V10 同窗口的交易数 + 平均 lot (跟当前主表一致)"""
    v10 = pd.read_csv(PY_V10_TRADES, encoding="utf-8")
    fixed_lot = float(v10["lot"].mean())
    return len(v10), fixed_lot


def single_mc_run(dt, o, h, l, c, n_trades: int, fixed_lot: float,
                  rng: np.random.Generator) -> dict:
    """
    跑一次 MC: 随机选 n_trades 个 1m bar 入场, 随机方向, 用 1m 回放 SL/TP,
    爆仓即停 (equity <= 0 后停止开新仓), 返回 metrics.
    """
    # 候选入场 bar 索引: 留出 max_minutes 缓冲, 避免越界
    valid_max = len(dt) - 1440
    if valid_max < n_trades:
        raise ValueError(f"1m bar 不够: {valid_max} < {n_trades}")
    pick = np.sort(rng.choice(valid_max, size=n_trades, replace=False))
    sides = rng.choice([1, -1], size=n_trades)

    cost = round_trip_cost_usd(fixed_lot)
    equity = INITIAL_BALANCE
    pnl_list = []
    close_dates = []

    for i, side in zip(pick, sides):
        if equity <= 0:
            break  # 爆仓即停
        entry_dt = dt[i]
        entry_px = o[i]   # 用 1m open 作为入场价 (next-bar 假设也可, 此处简化)
        if side == 1:
            sl = entry_px - SL_USD
            tp = entry_px + TP_USD
        else:
            sl = entry_px + SL_USD
            tp = entry_px - TP_USD
        r = replay_sl_tp(dt, h, l, c, entry_dt, side, entry_px, sl, tp, max_minutes=1440)
        if r is None:
            continue
        pnl_gross = side * (r["close_price"] - entry_px) * fixed_lot * CONTRACT_SIZE
        pnl_net = pnl_gross - cost
        equity += pnl_net
        pnl_list.append(pnl_net)
        close_dates.append(r["close_dt"])

    if len(pnl_list) < 2:
        return None  # 几乎全爆仓, 跳过

    trades = pd.DataFrame({"date": pd.to_datetime(close_dates), "pnl": pnl_list})
    trades = trades.sort_values("date").reset_index(drop=True)
    daily_pnl = trades.groupby(trades["date"].dt.normalize())["pnl"].sum()
    daily_equity = INITIAL_BALANCE + daily_pnl.cumsum()

    m = compute_metrics(daily_equity, trades_pnl=trades["pnl"],
                        initial_balance=INITIAL_BALANCE, label="MC")
    m["actual_n_trades"] = len(pnl_list)
    m["broke"]           = bool(daily_equity.iloc[-1] <= 0)
    return m


def main():
    dt, o, h, l, c = load_1m_arrays()
    n_trades, fixed_lot = n_trades_from_v10()
    print(f"[MC] 窗口 {WINDOW_START} ~ {WINDOW_END}  ({len(dt):,} 根 1m bar)")
    print(f"     V10 同窗口实际交易数 = {n_trades} (用作 MC 频率匹配)")
    print(f"     固定 lot = {fixed_lot:.4f}  SL=${SL_USD}  TP=${TP_USD}  跑 {MC_RUNS} 次")

    rng = np.random.default_rng(MC_SEED)
    results = []
    for k in range(MC_RUNS):
        m = single_mc_run(dt, o, h, l, c, n_trades, fixed_lot, rng)
        if m is None:
            continue
        results.append({
            "run": k + 1,
            "total_return_pct": m["total_return_pct"],
            "sharpe":           m["sharpe"],
            "max_dd_pct":       m["max_dd_pct"],
            "win_rate_pct":     m["win_rate_pct"],
            "final_balance":    m["final_balance"],
            "actual_n_trades":  m["actual_n_trades"],
            "broke":            m["broke"],
        })
        if (k + 1) % 50 == 0:
            print(f"   ... run {k+1}/{MC_RUNS}", flush=True)

    runs = pd.DataFrame(results)
    runs.to_csv(OUT_RUNS, index=False, encoding="utf-8")

    print("\n=== MC 分布 (跑通的 run) ===")
    n_broke = int(runs["broke"].sum())
    print(f"  跑通: {len(runs)}/{MC_RUNS}  爆仓 run: {n_broke}")
    for col in ["total_return_pct", "sharpe", "max_dd_pct", "win_rate_pct"]:
        s = runs[col]
        print(f"  {col:18}  mean={s.mean():8.3f}  std={s.std():7.3f}  "
              f"2.5%={s.quantile(0.025):8.3f}  50%={s.quantile(0.50):8.3f}  97.5%={s.quantile(0.975):8.3f}")

    # 单独算 non-broke 子集
    ok = runs[~runs["broke"]]
    if len(ok):
        print(f"\n=== non-broke 子集 ({len(ok)} runs) MaxDD 分布 ===")
        s = ok["max_dd_pct"]
        print(f"  mean={s.mean():.2f}  2.5%={s.quantile(0.025):.2f}  97.5%={s.quantile(0.975):.2f}")

    mean_metrics = {
        "label":            "Baseline 4: Random MC (mean)",
        "initial_balance":  INITIAL_BALANCE,
        "final_balance":    float(runs["final_balance"].mean()),
        "total_return_pct": float(runs["total_return_pct"].mean()),
        "ann_return_pct":   np.nan,
        "profit_factor":    np.nan,
        "sharpe":           float(runs["sharpe"].mean()),
        "sortino":          np.nan,
        "calmar":           np.nan,
        "max_dd_pct":       float(runs["max_dd_pct"].mean()),
        "win_rate_pct":     float(runs["win_rate_pct"].mean()),
        "avg_rr":           np.nan,
        "max_single_win":   np.nan,
        "max_single_loss":  np.nan,
        "n_trades":         int(n_trades),
        "ret_95ci_low":     float(runs["total_return_pct"].quantile(0.025)),
        "ret_95ci_high":    float(runs["total_return_pct"].quantile(0.975)),
        "sharpe_95ci_low":  float(runs["sharpe"].quantile(0.025)),
        "sharpe_95ci_high": float(runs["sharpe"].quantile(0.975)),
        "broke_pct":        float(runs["broke"].mean() * 100),
    }
    pd.DataFrame([mean_metrics]).to_csv(OUT_METRICS, index=False, encoding="utf-8")
    print(f"\n  saved -> {OUT_RUNS.name} / {OUT_METRICS.name}")


if __name__ == "__main__":
    main()
