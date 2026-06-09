# -*- coding: utf-8 -*-
"""
Baseline 3 — Classical Trend-Following: MA20/50 Crossover + V11 sqrt sizing
========================================================================
入场: D1 上 MA20 上穿 MA50 (bullish crossover) -> BUY
      MA20 下穿 MA50 (bearish crossover) -> SELL
出场: 反向 crossover 平仓 (反手)
SL/TP: SL = 2 × ATR(14), TP = 4 × ATR(14)  (1:2 R:R, 经典趋势设定)
仓位: RiskPercent×Equity / SL_USD × V11_sqrt_multiplier
       multiplier 用同日 D1 atr_ratio + trend (跟 V11 MQL5 一致, sqrt 公式)

数据预热: 用 _daily_extended.py 提前到 2022-06 保证 MA200 + ATR60 满血
交易窗口: 2024-01-02 ~ 2026-04-29 (跟 V10/V11 严格对齐)
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (INITIAL_BALANCE, RISK_PERCENT, CONTRACT_SIZE,
                     PIP_VALUE_PER_LOT, OUT_DIR, round_trip_cost_usd,
                     DATE_START, DATE_END)
from _metrics import compute_metrics, print_metrics
from _v11_recompute import v11_sqrt_multiplier
from _daily_extended import build_daily_extended

OUT_TRADES  = OUT_DIR / "baseline_3_ma_cross_trades.csv"
OUT_EQUITY  = OUT_DIR / "baseline_3_ma_cross_equity.csv"
OUT_METRICS = OUT_DIR / "baseline_3_ma_cross_metrics.csv"

# 100 oz/lot, 价格单位 USD/oz -> 1 USD 价格变动 = lot * 100 USD pnl
USD_PER_LOT_PER_DOLLAR = CONTRACT_SIZE   # = 100


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # ATR(14) — TR = max(H-L, |H-prevC|, |L-prevC|)
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    df["ATR60"] = tr.rolling(60).mean()
    df["atr_ratio"] = df["ATR14"] / df["ATR60"]

    # trend state (照搬 V11 MQL5 GetTrendState_D1 逻辑, 见源码 124-156)
    def label_trend(row):
        ma20, ma50, ma200, close = row["MA20"], row["MA50"], row["MA200"], row["Close"]
        if any(pd.isna(x) for x in [ma20, ma50, ma200]):
            return None
        # 多头排列
        if ma20 > ma50 > ma200 and close > ma20:
            return "strong_up"
        # 空头排列
        if ma20 < ma50 < ma200 and close < ma20:
            return "strong_down"
        # 弱空: MA20 < MA50, 但未空头排列
        if ma20 < ma50 and not (ma20 < ma50 < ma200):
            return "weak_down"
        # 震荡: MA 之间距离很近
        spread = max(ma20, ma50, ma200) - min(ma20, ma50, ma200)
        if spread / close < 0.01:
            return "range"
        return "mixed"
    df["trend"] = df.apply(label_trend, axis=1)
    return df


def run_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """单一持仓: 反向 cross 时平仓反手, 或 SL/TP 触发先平仓; 下一个 cross 再开."""
    trades = []
    pos = None  # dict(side, open_date, open_price, lot, sl, tp, mult, atr_at_entry, equity_at_entry)
    equity = INITIAL_BALANCE

    # 信号: MA20 cross MA50
    df["cross"] = 0
    above = df["MA20"] > df["MA50"]
    df.loc[above & (~above.shift(1).fillna(False)), "cross"] = 1   # bullish crossover: MA20 上穿 MA50
    df.loc[(~above) & (above.shift(1).fillna(False)), "cross"] = -1  # bearish crossover: MA20 下穿 MA50

    for i, row in df.iterrows():
        date, h, l, c, o = row["Date"], row["High"], row["Low"], row["Close"], row["Open"]
        atr = row["ATR14"]

        # 1) 若有持仓, 检查 SL/TP (用当日 H/L 模拟; 若都触发, 保守按 SL 先)
        if pos is not None:
            close_reason = None
            close_price = None
            if pos["side"] == "BUY":
                sl_hit = l <= pos["sl"]
                tp_hit = h >= pos["tp"]
                if sl_hit and tp_hit:
                    close_reason, close_price = "SL", pos["sl"]
                elif sl_hit:
                    close_reason, close_price = "SL", pos["sl"]
                elif tp_hit:
                    close_reason, close_price = "TP", pos["tp"]
            else:  # SELL
                sl_hit = h >= pos["sl"]
                tp_hit = l <= pos["tp"]
                if sl_hit and tp_hit:
                    close_reason, close_price = "SL", pos["sl"]
                elif sl_hit:
                    close_reason, close_price = "SL", pos["sl"]
                elif tp_hit:
                    close_reason, close_price = "TP", pos["tp"]

            if close_reason:
                direction = 1 if pos["side"] == "BUY" else -1
                pnl_gross = direction * (close_price - pos["open_price"]) * pos["lot"] * CONTRACT_SIZE
                cost = round_trip_cost_usd(pos["lot"])
                pnl_net = pnl_gross - cost
                equity += pnl_net
                trades.append({
                    "side": pos["side"], "open_date": pos["open_date"], "open_price": pos["open_price"],
                    "close_date": date, "close_price": close_price, "lot": pos["lot"],
                    "sl": pos["sl"], "tp": pos["tp"], "mult": pos["mult"],
                    "reason": close_reason, "pnl": pnl_net,
                    "rr": (pos["tp"] - pos["open_price"]) / (pos["open_price"] - pos["sl"]) if pos["side"]=="BUY"
                          else (pos["open_price"] - pos["tp"]) / (pos["sl"] - pos["open_price"]),
                })
                pos = None

        # 2) cross 信号 -> 反向平仓 / 开新仓 (用当日 close 作为入场, 简化)
        cross = row["cross"]
        if cross != 0 and pd.notna(atr) and pd.notna(row["atr_ratio"]) and row["trend"]:
            # 反手: 若已有持仓且方向相反, 按 close 平仓
            if pos is not None:
                want_side = "BUY" if cross == 1 else "SELL"
                if pos["side"] != want_side:
                    direction = 1 if pos["side"] == "BUY" else -1
                    pnl_gross = direction * (c - pos["open_price"]) * pos["lot"] * CONTRACT_SIZE
                    cost = round_trip_cost_usd(pos["lot"])
                    pnl_net = pnl_gross - cost
                    equity += pnl_net
                    trades.append({
                        "side": pos["side"], "open_date": pos["open_date"], "open_price": pos["open_price"],
                        "close_date": date, "close_price": c, "lot": pos["lot"],
                        "sl": pos["sl"], "tp": pos["tp"], "mult": pos["mult"],
                        "reason": "REVERSE", "pnl": pnl_net,
                        "rr": (pos["tp"] - pos["open_price"]) / (pos["open_price"] - pos["sl"]) if pos["side"]=="BUY"
                              else (pos["open_price"] - pos["tp"]) / (pos["sl"] - pos["open_price"]),
                    })
                    pos = None

            # 开新仓
            if pos is None and equity > 0:
                side = "BUY" if cross == 1 else "SELL"
                mult = v11_sqrt_multiplier(row["atr_ratio"], row["trend"])
                sl_dist = 2.0 * atr
                tp_dist = 4.0 * atr
                # lot: RiskPercent × equity = sl_dist × lot × CONTRACT_SIZE
                # → lot = (RiskPercent × equity) / (sl_dist × 100) × mult
                base_lot = (RISK_PERCENT * equity) / (sl_dist * CONTRACT_SIZE)
                lot = round(max(0.01, base_lot * mult), 2)
                if side == "BUY":
                    sl = c - sl_dist
                    tp = c + tp_dist
                else:
                    sl = c + sl_dist
                    tp = c - tp_dist
                pos = dict(side=side, open_date=date, open_price=c, lot=lot,
                           sl=sl, tp=tp, mult=mult)

    # 末日强制平仓
    if pos is not None:
        last = df.iloc[-1]
        direction = 1 if pos["side"] == "BUY" else -1
        pnl_gross = direction * (last["Close"] - pos["open_price"]) * pos["lot"] * CONTRACT_SIZE
        cost = round_trip_cost_usd(pos["lot"])
        pnl_net = pnl_gross - cost
        equity += pnl_net
        trades.append({
            "side": pos["side"], "open_date": pos["open_date"], "open_price": pos["open_price"],
            "close_date": last["Date"], "close_price": last["Close"], "lot": pos["lot"],
            "sl": pos["sl"], "tp": pos["tp"], "mult": pos["mult"],
            "reason": "EOD", "pnl": pnl_net,
            "rr": np.nan,
        })

    return pd.DataFrame(trades)


def main():
    print("[MA cross] 加载 daily K (含预热)")
    px = build_daily_extended()
    px = px.sort_values("Date").reset_index(drop=True)

    print("[MA cross] 计算指标 (MA20/50/200, ATR14/60, atr_ratio, trend)")
    px = compute_indicators(px)

    # 限定交易窗口
    px_win = px[(px["Date"] >= DATE_START) & (px["Date"] <= DATE_END)].reset_index(drop=True)
    print(f"           交易窗口 {len(px_win)} 日 ({px_win['Date'].min().date()} ~ {px_win['Date'].max().date()})")

    trades = run_strategy(px_win)
    print(f"\n[MA cross] 共 {len(trades)} 笔交易")
    print(f"           BUY={len(trades[trades['side']=='BUY'])}  SELL={len(trades[trades['side']=='SELL'])}")
    print(f"           SL/TP/REVERSE/EOD 分布:")
    print(trades["reason"].value_counts().to_string())

    # 日度 equity
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades_sorted = trades.sort_values("close_date").reset_index(drop=True)
    trades_sorted["equity"] = INITIAL_BALANCE + trades_sorted["pnl"].cumsum()
    # 用 close_date 当日聚合
    daily_pnl = trades_sorted.groupby(trades_sorted["close_date"].dt.normalize())["pnl"].sum()
    daily_equity = (INITIAL_BALANCE + daily_pnl.cumsum())
    # 用 daily K 索引补齐 (没交易的日子用前一日 equity)
    full_idx = px_win.set_index("Date").index
    daily_equity = daily_equity.reindex(full_idx).ffill().fillna(INITIAL_BALANCE)

    m = compute_metrics(daily_equity, trades_pnl=trades["pnl"], trades_rr=trades["rr"],
                        initial_balance=INITIAL_BALANCE,
                        label="Baseline 3: MA Cross + V11 sizing")
    print_metrics(m)

    trades.to_csv(OUT_TRADES, index=False, encoding="utf-8")
    pd.DataFrame({"Date": daily_equity.index, "equity": daily_equity.values}).to_csv(
        OUT_EQUITY, index=False, encoding="utf-8")
    pd.DataFrame([m]).to_csv(OUT_METRICS, index=False, encoding="utf-8")
    print(f"\n  saved -> {OUT_TRADES.name} / {OUT_EQUITY.name} / {OUT_METRICS.name}")
    return m


if __name__ == "__main__":
    main()
