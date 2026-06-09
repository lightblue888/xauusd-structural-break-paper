# -*- coding: utf-8 -*-
"""导出 Python V10/V11 daily equity csv (给 summary_plots 用)"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import INITIAL_BALANCE, OUT_DIR

for name in ["v10", "v11"]:
    df = pd.read_csv(OUT_DIR / f"python_{name}_trades.csv", encoding="utf-8")
    df["final_close_time"] = pd.to_datetime(df["final_close_time"])
    df["pnl"] = pd.to_numeric(df["total_pnl_usd"], errors="coerce")
    df = df.sort_values("final_close_time")
    daily_pnl = df.groupby(df["final_close_time"].dt.normalize())["pnl"].sum()
    eq = INITIAL_BALANCE + daily_pnl.cumsum()
    out = OUT_DIR / f"python_{name}_equity.csv"
    pd.DataFrame({"Date": eq.index, "equity": eq.values}).to_csv(
        out, index=False, encoding="utf-8")
    print(f"  saved -> {out.name}  ({len(eq)} days, final ${eq.iloc[-1]:,.2f})")
