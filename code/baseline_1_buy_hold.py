# -*- coding: utf-8 -*-
"""
Baseline 1 — Buy & Hold XAUUSD
========================================================================
策略: 2024-01-02 首日 close 全仓买入, 持到 2026-04-29 末日 close 卖出。
- unlevered, 仓位 = INITIAL_BALANCE 等效手数
- 等效手数 lot_eq = INITIAL / (open_0 * 100)
- 成本: 一次往返 (spread + 2×commission + 2×slip) 按 lot_eq 折算
- 日度 equity 用日 close 标价

n_trades = 1, 所以 PF / 平均 R:R 报 N/A; 单笔最大盈/亏 = 总 PnL
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (INITIAL_BALANCE, CONTRACT_SIZE, OUT_DIR, CACHE_DIR,
                     round_trip_cost_usd)
from _metrics import compute_metrics, print_metrics
from _price_series import build_daily_series

OUT_CSV = OUT_DIR / "baseline_1_buy_hold_equity.csv"
OUT_METRICS = OUT_DIR / "baseline_1_buy_hold_metrics.csv"


def main():
    px = build_daily_series()  # Date, Open, High, Low, Close, Volume, source
    px = px.sort_values("Date").reset_index(drop=True)

    open_0  = float(px["Close"].iloc[0])   # 用首日 close 作为买入价 (论文惯例)
    close_T = float(px["Close"].iloc[-1])

    # 等效手数: 初始资金 / 名义价值 (1 lot = 100 oz × price)
    lot_eq = INITIAL_BALANCE / (open_0 * CONTRACT_SIZE)
    cost   = round_trip_cost_usd(lot_eq)

    print(f"[B&H] 首日 {px['Date'].iloc[0].date()} close = ${open_0:,.2f}")
    print(f"      末日 {px['Date'].iloc[-1].date()} close = ${close_T:,.2f}")
    print(f"      等效持仓 lot_eq = {lot_eq:.6f} lot ({lot_eq*CONTRACT_SIZE:.4f} oz)")
    print(f"      往返成本 = ${cost:.4f}")

    # ---- 日度 equity = INITIAL × (close_t / open_0) - 期末扣一次成本 ----
    px["equity"] = INITIAL_BALANCE * (px["Close"] / open_0)
    # 期末扣成本: 最后一行减去全部 round-trip cost (近似)
    px.loc[px.index[-1], "equity"] = px["equity"].iloc[-1] - cost

    daily_equity = px.set_index("Date")["equity"]
    total_pnl    = daily_equity.iloc[-1] - INITIAL_BALANCE

    # 单笔 PnL = 总 PnL (B&H 只有 1 笔)
    trades_pnl = pd.Series([total_pnl], name="pnl")

    m = compute_metrics(daily_equity, trades_pnl=trades_pnl,
                        initial_balance=INITIAL_BALANCE,
                        label="Baseline 1: Buy & Hold")
    print_metrics(m)

    # 保存
    px[["Date", "Close", "equity"]].to_csv(OUT_CSV, index=False, encoding="utf-8")
    pd.DataFrame([m]).to_csv(OUT_METRICS, index=False, encoding="utf-8")
    print(f"\n  equity -> {OUT_CSV.name}")
    print(f"  metrics -> {OUT_METRICS.name}")
    return m


if __name__ == "__main__":
    main()
