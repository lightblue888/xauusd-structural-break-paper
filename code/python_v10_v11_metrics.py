# -*- coding: utf-8 -*-
"""
Python V10 / V11 metrics (同口径 daily annualized × √252)
========================================================================
直接基于 Python V10/V11 trades 计算 10 项指标, 输出到 data/python_v10_v11_metrics.csv
被 summary_table.py 引用作为论文 hero 数字.
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import INITIAL_BALANCE, OUT_DIR
from _metrics import compute_metrics, print_metrics

OUT_METRICS = OUT_DIR / "python_v10_v11_metrics.csv"


def to_daily_equity(trades_df, pnl_col="total_pnl_usd",
                    close_col="final_close_time"):
    df = trades_df.copy()
    df[close_col] = pd.to_datetime(df[close_col])
    df["pnl"] = pd.to_numeric(df[pnl_col], errors="coerce")
    df = df.sort_values(close_col)
    daily_pnl = df.groupby(df[close_col].dt.normalize())["pnl"].sum()
    eq = INITIAL_BALANCE + daily_pnl.cumsum()
    return eq, df["pnl"]


def compute_rr(df):
    df = df.copy()
    df["sl_dist"] = (df["open_price"] - df["sl"]).abs()
    df["tp_dist"] = (df["tp"] - df["open_price"]).abs()
    df["rr"] = df["tp_dist"] / df["sl_dist"]
    return df["rr"]


def main():
    py_v10 = pd.read_csv(OUT_DIR / "python_v10_trades.csv", encoding="utf-8")
    py_v11 = pd.read_csv(OUT_DIR / "python_v11_trades.csv", encoding="utf-8")

    rows = []
    for label, df in [("Python V10", py_v10), ("Python V11", py_v11)]:
        eq, pnl = to_daily_equity(df)
        rr = compute_rr(df)
        m = compute_metrics(eq, trades_pnl=pnl, trades_rr=rr,
                            initial_balance=INITIAL_BALANCE, label=label)
        print_metrics(m)
        rows.append(m)

    pd.DataFrame(rows).to_csv(OUT_METRICS, index=False, encoding="utf-8")
    print(f"\n  saved -> {OUT_METRICS.name}")


if __name__ == "__main__":
    main()
