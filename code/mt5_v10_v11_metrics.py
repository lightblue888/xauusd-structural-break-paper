# -*- coding: utf-8 -*-
"""
MT5 V10 / V11 ground-truth metrics
========================================================================
读 v10_mt5_trades.csv + v11_mt5_trades.csv (来源: MT5 tester log parser),
用统一 _metrics.py 算 10 项指标, 论文 hero 数字.
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import INITIAL_BALANCE, OUT_DIR
from _metrics import compute_metrics, print_metrics

OUT_METRICS = OUT_DIR / "mt5_v10_v11_metrics.csv"


def load_mt5(path: Path):
    df = pd.read_csv(path, encoding="utf-8")
    # MT5 风格的日期 "2024.01.02 17:00:20" → 标准 datetime
    df["open_time"]  = pd.to_datetime(df["open_time"], format="%Y.%m.%d %H:%M:%S")
    df["close_time"] = pd.to_datetime(df["final_close_time"], format="%Y.%m.%d %H:%M:%S")
    df["pnl"] = pd.to_numeric(df["total_pnl_usd"], errors="coerce")
    df["sl_dist"] = (df["open_price"] - df["sl"]).abs()
    df["tp_dist"] = (df["tp"] - df["open_price"]).abs()
    df["rr"] = df["tp_dist"] / df["sl_dist"]
    return df


def to_daily_equity(df):
    df = df.sort_values("close_time").reset_index(drop=True)
    daily_pnl = df.groupby(df["close_time"].dt.normalize())["pnl"].sum()
    eq = INITIAL_BALANCE + daily_pnl.cumsum()
    return eq


def main():
    v10 = load_mt5(OUT_DIR / "v10_mt5_trades.csv")
    v11 = load_mt5(OUT_DIR / "v11_mt5_trades.csv")

    rows = []
    for label, df in [("V10 (MT5, fixed ATR sizing)", v10),
                      ("V11 (MT5, sqrt sizing)", v11)]:
        eq = to_daily_equity(df)
        m = compute_metrics(eq, trades_pnl=df["pnl"], trades_rr=df["rr"],
                            initial_balance=INITIAL_BALANCE, label=label)
        print_metrics(m)
        rows.append(m)

    pd.DataFrame(rows).to_csv(OUT_METRICS, index=False, encoding="utf-8")
    print(f"\n  saved -> {OUT_METRICS.name}")

    # 同时导出 daily equity csv (给 summary_plots 用)
    for name, df in [("v10", v10), ("v11", v11)]:
        eq = to_daily_equity(df)
        out = OUT_DIR / f"mt5_{name}_equity.csv"
        pd.DataFrame({"Date": eq.index, "equity": eq.values}).to_csv(
            out, index=False, encoding="utf-8")
        print(f"  saved -> {out.name}  ({len(eq)} days, final ${eq.iloc[-1]:,.2f})")


if __name__ == "__main__":
    main()
