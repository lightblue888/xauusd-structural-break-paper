# -*- coding: utf-8 -*-
"""
Baseline 6 — 子样本三段稳健性检验 (Phase 2: Python V10/V11 数据)
========================================================================
切 Python V10 / V11 trades 到 2024 / 2025 / 2026Q1+ 三段,
每段独立跑 10 指标, 看 V11 的 alpha 是不是稳定。
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (INITIAL_BALANCE, OUT_DIR, SUBSAMPLE_WINDOWS)
from _metrics import compute_metrics, print_metrics

PY_V10_CSV = OUT_DIR / "python_v10_trades.csv"
PY_V11_CSV = OUT_DIR / "python_v11_trades.csv"
OUT_TABLE  = OUT_DIR / "baseline_6_subsample_metrics.csv"


def load_python(path):
    df = pd.read_csv(path, encoding="utf-8")
    df["open_time"]  = pd.to_datetime(df["open_time"])
    df["close_time"] = pd.to_datetime(df["final_close_time"])
    df["pnl"] = pd.to_numeric(df["total_pnl_usd"], errors="coerce")
    df["sl_dist"] = (df["open_price"] - df["sl"]).abs()
    df["tp_dist"] = (df["tp"] - df["open_price"]).abs()
    df["rr"] = df["tp_dist"] / df["sl_dist"]
    return df


def slice_metrics(df: pd.DataFrame, start: str, end: str, label: str) -> dict:
    sub = df[(df["close_time"] >= start) & (df["close_time"] <= end + " 23:59:59")].copy()
    sub = sub.sort_values("close_time").reset_index(drop=True)
    if len(sub) == 0:
        return None
    daily_pnl = sub.groupby(sub["close_time"].dt.normalize())["pnl"].sum()
    daily_equity = INITIAL_BALANCE + daily_pnl.cumsum()
    m = compute_metrics(daily_equity, trades_pnl=sub["pnl"], trades_rr=sub["rr"],
                        initial_balance=INITIAL_BALANCE, label=label)
    return m


def main():
    v10 = load_python(PY_V10_CSV)
    v11 = load_python(PY_V11_CSV)
    rows = []
    for tag, s, e in SUBSAMPLE_WINDOWS:
        for strat_name, df in [("V10", v10), ("V11", v11)]:
            m = slice_metrics(df, s, e, label=f"{strat_name} [{tag}]")
            if m is None:
                continue
            print_metrics(m)
            rows.append(m)

    pd.DataFrame(rows).to_csv(OUT_TABLE, index=False, encoding="utf-8")
    print(f"\n  saved -> {OUT_TABLE.name}")
    return rows


if __name__ == "__main__":
    main()
