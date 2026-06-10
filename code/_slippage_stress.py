# -*- coding: utf-8 -*-
"""
滑点压力测试 (仅针对 contribution (a): V10/V11 真 tick edge 的成本稳健性)
========================================================================
在 MT5 mode2 真 tick 的 ground-truth 交易上后处理: 每个 fill 扣保守滑点,
重算 Sharpe/Return/Calmar。证明 edge 不是"忽略执行成本"刷出来的。

模型 (conservative, per-executed-volume):
  每个 fill 成本 = slip_per_oz × 100(合约 oz/lot) × fill_lot
  一笔交易 fills = 入场(lot) + 半仓(half_exit_lot) + 终仓(final_close_lot)
  → round-trip 总量 = 2×lot (entry 一次 + exit 一次, exit 可能拆两 fill)
  零返佣假设 (最保守); 真实 ECMarkets 80% 返佣会进一步降低净成本。
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import INITIAL_BALANCE, OUT_DIR
from _metrics import compute_metrics

CONTRACT_OZ = 100.0            # XAUUSD: 1.0 lot = 100 oz
SLIP_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]   # 美元/oz, 每个 fill


def load(path):
    df = pd.read_csv(path, encoding="utf-8")
    df["close_time"] = pd.to_datetime(df["final_close_time"], errors="coerce")
    for c in ["lot", "half_exit_lot", "final_close_lot", "total_pnl_usd",
              "open_price", "sl", "tp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["fill_lots"] = df["lot"] + df["half_exit_lot"].fillna(0) + df["final_close_lot"].fillna(0)
    df["rr"] = (df["tp"] - df["open_price"]).abs() / (df["open_price"] - df["sl"]).abs()
    return df


def metrics_at(df, slip, label):
    cost = slip * CONTRACT_OZ * df["fill_lots"]
    pnl = df["total_pnl_usd"] - cost
    d = df.assign(pnl=pnl).sort_values("close_time")
    daily = d.groupby(d["close_time"].dt.normalize())["pnl"].sum()
    eq = INITIAL_BALANCE + daily.cumsum()
    m = compute_metrics(eq, trades_pnl=pnl, trades_rr=df["rr"],
                        initial_balance=INITIAL_BALANCE, label=label)
    m["slip_per_oz"] = slip
    m["total_slip_cost"] = float(cost.sum())
    return m


def main():
    rows = []
    for ver in ["v10", "v11"]:
        df = load(OUT_DIR / f"{ver}_realticks_trades.csv")
        print(f"\n=== {ver.upper()} (n={len(df)}, 总 fill 量={df['fill_lots'].sum():.1f} lots) ===")
        for slip in SLIP_LEVELS:
            m = metrics_at(df, slip, f"{ver}_slip{slip}")
            rows.append({"ver": ver.upper(), "slip_$/oz": slip,
                         "Sharpe": round(m["sharpe"], 3),
                         "Return%": round(m["total_return_pct"], 1),
                         "Calmar": round(m["calmar"], 3),
                         "Sortino": round(m["sortino"], 3),
                         "MaxDD%": round(m["max_dd_pct"], 1),
                         "slip_cost$": round(m["total_slip_cost"], 0)})
            print(f"  slip ${slip:.2f}/oz: Sharpe={m['sharpe']:.3f}  "
                  f"Return={m['total_return_pct']:.1f}%  Calmar={m['calmar']:.3f}  "
                  f"(滑点总成本 ${m['total_slip_cost']:,.0f})")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "slippage_stress.csv", index=False, encoding="utf-8")
    print("\nsaved -> slippage_stress.csv")
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
