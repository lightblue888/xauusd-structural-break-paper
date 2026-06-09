# -*- coding: utf-8 -*-
"""
Baseline 5 — 静态价格趋势突破策略 (剥离半仓锁利)
========================================================================
目的: 隔离半仓锁利仓位管理机制的贡献。
      若本 baseline 显著弱于 V10, 说明 V10/V11 alpha 部分来自半仓锁利;
      若接近, 说明 alpha 主要来自 entry edge + sizing 本身。

策略:
- 入场: 复用 V10 csv 1507 笔的入场时刻 + 方向 + 入场价 (价格趋势突破信号)
- 仓位: 固定 lot = V10 平均 lot ≈ 0.1804 (跟 Baseline 4 一致)
- 风控: 仅静态 SL = 3 USD / TP = 20 USD (与 V10 / V11 同距离)
- 关键剥离: 无半仓锁利 (V10/V11 含此机制, 本 baseline 不含)
- 出场: 用 1m bar 数据精确回放 SL/TP 命中

数据精度局限: 1m 数据仅至 2025-12-31, 2026 Q1+ 入场点用 V10 实际收盘价
              作为 fallback close 处理 (TIMEOUT 列同等对待)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (INITIAL_BALANCE, CONTRACT_SIZE, OUT_DIR,
                     round_trip_cost_usd)
from _metrics import compute_metrics, print_metrics
from _bar_replay import load_1m_arrays, replay_sl_tp

SL_USD = 3.0
TP_USD = 20.0
PY_V10_TRADES = OUT_DIR / "python_v10_trades.csv"

OUT_TRADES  = OUT_DIR / "baseline_5_static_breakout_trades.csv"
OUT_EQUITY  = OUT_DIR / "baseline_5_static_breakout_equity.csv"
OUT_METRICS = OUT_DIR / "baseline_5_static_breakout_metrics.csv"


def main():
    print("[Baseline 5] 加载 1m bar 数据...")
    dt, o, h, l, c = load_1m_arrays()
    bar_end = pd.Timestamp(dt[-1])
    print(f"             1m 数据末端: {bar_end}")

    # 使用 Python V10 入场点 (跟当前主表 V10 一致, 而非 MT5 csv)
    v10 = pd.read_csv(PY_V10_TRADES, encoding="utf-8")
    v10["open_time"] = pd.to_datetime(v10["open_time"])
    fixed_lot = float(v10["lot"].mean())
    cost = round_trip_cost_usd(fixed_lot)
    print(f"             Python V10 入场点 {len(v10)} 笔, 固定 lot = {fixed_lot:.4f}, "
          f"往返成本 = ${cost:.4f}")

    trades = []
    out_of_range = 0
    for _, row in v10.iterrows():
        entry_dt_ts = row["open_time"]
        side_str = row["side"]
        side = 1 if side_str == "BUY" else -1
        entry_px = float(row["open_price"])

        # 超出 1m 数据末端 → 用 V10 csv 的 final_close 作为 fallback (论文里注明)
        if entry_dt_ts > bar_end:
            out_of_range += 1
            close_dt = pd.to_datetime(row["final_close_time"])
            close_px = float(row["final_close_price"])
            reason = row["final_reason"] if row["final_reason"] in ("SL", "TP") else "FALLBACK_V10"
            hold_min = float(row["hold_min"])
        else:
            entry_dt = np.datetime64(entry_dt_ts.to_pydatetime().replace(microsecond=0), "s")
            sl = entry_px - SL_USD if side == 1 else entry_px + SL_USD
            tp = entry_px + TP_USD if side == 1 else entry_px - TP_USD
            r = replay_sl_tp(dt, h, l, c, entry_dt, side, entry_px, sl, tp, max_minutes=1440)
            if r is None:
                # 入场 dt 在 1m 数据外 (例如周末) → fallback 用 V10 csv
                out_of_range += 1
                close_dt = pd.to_datetime(row["final_close_time"])
                close_px = float(row["final_close_price"])
                reason = "FALLBACK_V10"
                hold_min = float(row["hold_min"])
            else:
                close_dt = pd.Timestamp(r["close_dt"])
                close_px = r["close_price"]
                reason = r["reason"]
                hold_min = r["hold_min"]

        pnl_gross = side * (close_px - entry_px) * fixed_lot * CONTRACT_SIZE
        pnl_net = pnl_gross - cost
        # R:R 永远 6.67 (TP_USD / SL_USD)
        trades.append({
            "pos_id":     row["pos_id"],
            "side":       side_str,
            "open_time":  entry_dt_ts,
            "open_price": entry_px,
            "close_time": close_dt,
            "close_price": close_px,
            "lot":        fixed_lot,
            "reason":     reason,
            "hold_min":   hold_min,
            "pnl":        pnl_net,
            "rr":         TP_USD / SL_USD,
        })

    trades_df = pd.DataFrame(trades)
    print(f"\n[Baseline 5] 回放完成 {len(trades_df)} 笔  (out-of-range fallback: {out_of_range})")
    print(f"             出场分布:")
    print(trades_df["reason"].value_counts().to_string())

    # daily equity
    trades_df = trades_df.sort_values("close_time").reset_index(drop=True)
    trades_df["close_time"] = pd.to_datetime(trades_df["close_time"])
    daily_pnl = trades_df.groupby(trades_df["close_time"].dt.normalize())["pnl"].sum()
    daily_equity = INITIAL_BALANCE + daily_pnl.cumsum()

    m = compute_metrics(daily_equity, trades_pnl=trades_df["pnl"], trades_rr=trades_df["rr"],
                        initial_balance=INITIAL_BALANCE,
                        label="Baseline 5: Static Breakout (only SL/TP, no half-exit)")
    print_metrics(m)

    trades_df.to_csv(OUT_TRADES, index=False, encoding="utf-8")
    pd.DataFrame({"Date": daily_equity.index, "equity": daily_equity.values}).to_csv(
        OUT_EQUITY, index=False, encoding="utf-8")
    pd.DataFrame([m]).to_csv(OUT_METRICS, index=False, encoding="utf-8")
    print(f"\n  saved -> {OUT_TRADES.name} / {OUT_EQUITY.name} / {OUT_METRICS.name}")


if __name__ == "__main__":
    main()
