# -*- coding: utf-8 -*-
"""
解析 MT5 Strategy Tester agent log → trade-level CSV  (V3: state machine)
========================================================================
V11 EA 单仓: 任何时刻最多 1 个 position 在场.
按时间顺序遍历所有 deal_performed 事件, state machine 配对成完整 trade.
不依赖 MT5 内部 deal#/order#/position# 这三个独立编号系统 (易混淆).

字段输出 schema 跟 V10_1507笔交易.csv 对齐 (用于 baseline_*.py 复用).
"""
import sys
import os
import re
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import INITIAL_BALANCE, OUT_DIR

# 本地: 设环境变量 MT5_HASH (或直接设 MT5_TESTER_DIR), 见 .env.example
MT5_TERMINAL_HASH = os.environ.get("MT5_HASH", "YOUR_TERMINAL_HASH_HERE")
MT5_TESTER_DIR = Path(os.environ.get(
    "MT5_TESTER_DIR",
    str(Path.home() / "AppData/Roaming/MetaQuotes/Tester" / MT5_TERMINAL_HASH / "Agent-127.0.0.1-3001/logs")))

# 正则: 只需要 4 种事件
RE_SESSION = re.compile(
    r"testing of Experts\\(\S+\.ex5).*from\s+(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2})\s+to\s+(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2})"
)
RE_DEAL = re.compile(
    r"(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2})\s+deal performed \[#\d+\s+(buy|sell)\s+([\d.]+)\s+\S+\s+at\s+([\d.]+)\]"
)
# SL/TP triggered: 提取 sl/tp 值 + reason (close 的 lot 单独从 deal 里拿)
RE_SLTP = re.compile(
    r"(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2})\s+(stop loss|take profit) triggered #\d+\s+(buy|sell)\s+([\d.]+)\s+\S+\s+([\d.]+)\s+sl:\s+([\d.]+)\s+tp:\s+([\d.]+)"
)
# 末日强平
RE_EOD = re.compile(
    r"(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2})\s+position closed due end of test"
)
RE_FINAL_BAL = re.compile(r"final balance\s+([\d.]+)")


def find_latest_log() -> Path:
    if not MT5_TESTER_DIR.exists():
        raise FileNotFoundError(MT5_TESTER_DIR)
    logs = sorted(MT5_TESTER_DIR.glob("*.log"))
    if not logs:
        raise FileNotFoundError(f"无 .log 在 {MT5_TESTER_DIR}")
    return logs[-1]


def read_log(path: Path) -> list:
    text = path.read_bytes().decode("utf-16-le", errors="replace")
    return text.splitlines()


def find_sessions(lines: list) -> list:
    starts = []
    for i, line in enumerate(lines):
        m = RE_SESSION.search(line)
        if m:
            starts.append((i, f"{m.group(2)} → {m.group(3)}"))
    sessions = []
    for k, (idx, label) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        sessions.append((idx, end, label))
    return sessions


def parse_session(lines: list) -> pd.DataFrame:
    """
    State machine over chronologically-ordered events.
    Events: DEAL (open/close), SLTP marker (gives sl/tp + reason), EOD marker.
    """
    events = []   # (dt, kind, payload)
    for line in lines:
        m = RE_DEAL.search(line)
        if m:
            events.append((m.group(1), "DEAL", dict(
                side=m.group(2), lot=float(m.group(3)), price=float(m.group(4)),
            )))
            continue
        m = RE_SLTP.search(line)
        if m:
            events.append((m.group(1), "SLTP", dict(
                reason="SL" if m.group(2) == "stop loss" else "TP",
                pos_side=m.group(3),  # 持仓方向 (close 之前)
                sl=float(m.group(6)), tp=float(m.group(7)),
            )))
            continue
        m = RE_EOD.search(line)
        if m:
            events.append((m.group(1), "EOD", {}))

    # 排序: 按时间 + log 顺序 (log 内同 timestamp 已是 chronological)
    # events 列表已是 log 顺序, 不重排

    rows = []
    running_bal = INITIAL_BALANCE
    pos_counter = 0

    # state machine
    in_position = False
    pos_open_dt = None
    pos_open_price = None
    pos_open_lot_orig = None       # 原始开仓 lot
    pos_remaining_lot = None       # 剩余 lot (半仓后 = orig - half)
    pos_dir = None                 # 1=BUY 持仓, -1=SELL 持仓
    pos_sl = 0.0
    pos_tp = 0.0
    pos_half = None                # dict with half_dt/lot/price/pnl or None

    pending_close_reason = None    # "SL" / "TP" / "EOD" — 由 marker 设置, 下个 DEAL 关闭时用
    pending_sl_tp = None           # (sl, tp) 从 marker

    for dt_str, kind, p in events:
        if kind == "SLTP":
            # 仅做标记, 下一个 DEAL 会平仓
            pending_close_reason = p["reason"]
            pending_sl_tp = (p["sl"], p["tp"])
            # 同时也存 pos_sl / pos_tp (用于半仓后 SL=保本场景, 这个 marker 是最终触发的 sl 值)
            if not pos_sl:
                pos_sl = p["sl"]
                pos_tp = p["tp"]
            else:
                pos_sl = p["sl"]   # 半仓后 SL 已移到保本, 触发时记的就是保本价
                pos_tp = p["tp"]
            continue
        if kind == "EOD":
            pending_close_reason = "EOD"
            continue
        if kind != "DEAL":
            continue

        deal_side, deal_lot, deal_price = p["side"], p["lot"], p["price"]

        if not in_position:
            # 开仓 deal
            in_position = True
            pos_open_dt = dt_str
            pos_open_price = deal_price
            pos_open_lot_orig = deal_lot
            pos_remaining_lot = deal_lot
            pos_dir = 1 if deal_side == "buy" else -1
            pos_sl = 0.0
            pos_tp = 0.0
            pos_half = None
            pending_close_reason = None
        else:
            # close deal (要么是半仓, 要么是终仓)
            # deal_lot 跟 pos_remaining_lot 比较
            is_partial = deal_lot < pos_remaining_lot - 0.001 and pos_half is None
            if is_partial:
                # 半仓
                half_pnl_gross = pos_dir * (deal_price - pos_open_price) * deal_lot * 100
                pos_half = dict(
                    dt=dt_str, lot=deal_lot, price=deal_price, pnl=half_pnl_gross,
                )
                pos_remaining_lot -= deal_lot
            else:
                # 终仓
                final_pnl_gross = pos_dir * (deal_price - pos_open_price) * pos_remaining_lot * 100
                # 注意: MT5 backtest 成交价已含 broker 的 spread/commission,
                # 不再额外扣 cost (否则会双重扣).
                half_pnl = pos_half["pnl"] if pos_half else 0.0
                total_pnl = half_pnl + final_pnl_gross
                running_bal += total_pnl

                pos_counter += 1
                reason = pending_close_reason or "OTHER"
                hold_min = (datetime.strptime(dt_str, "%Y.%m.%d %H:%M:%S")
                            - datetime.strptime(pos_open_dt, "%Y.%m.%d %H:%M:%S")
                            ).total_seconds() / 60.0

                rows.append({
                    "pos_id":            pos_counter,
                    "side":              "BUY" if pos_dir == 1 else "SELL",
                    "open_time":         pos_open_dt,
                    "open_price":        pos_open_price,
                    "lot":               pos_open_lot_orig,
                    "sl":                pos_sl,
                    "tp":                pos_tp,
                    "half_exit_time":    pos_half["dt"] if pos_half else "",
                    "half_exit_lot":     pos_half["lot"] if pos_half else 0.0,
                    "half_exit_price":   pos_half["price"] if pos_half else 0.0,
                    "pnl_half_usd":      round(half_pnl, 2),
                    "final_close_time":  dt_str,
                    "final_close_price": deal_price,
                    "final_close_lot":   pos_remaining_lot,
                    "final_reason":      reason,
                    "pnl_final_usd":     round(final_pnl_gross, 2),
                    "hold_min":          round(hold_min, 2),
                    "total_pnl_usd":     round(total_pnl, 2),
                    "running_balance":   round(running_bal, 2),
                })

                # reset position state
                in_position = False
                pos_open_dt = pos_open_price = None
                pos_open_lot_orig = pos_remaining_lot = None
                pos_dir = None
                pos_half = None
                pending_close_reason = None
                pos_sl = pos_tp = 0.0

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, default=-1)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--out", type=str, default="v11_mt5_trades.csv")
    args = ap.parse_args()

    log_path = args.log or find_latest_log()
    print(f"[parse v3] log: {log_path.name}")
    lines = read_log(log_path)

    sessions = find_sessions(lines)
    print(f"           {len(sessions)} 个 sessions:")
    for i, (s, e, label) in enumerate(sessions):
        print(f"             #{i}: 行 {s:>5}~{e:>5}  {label}")

    sel = sessions[args.session]
    print(f"\n           解析 session {args.session}: {sel[2]}")
    session_lines = lines[sel[0]:sel[1]]
    df = parse_session(session_lines)
    print(f"           {len(df)} 笔交易")

    mt5_final = None
    for l in session_lines:
        m = RE_FINAL_BAL.search(l)
        if m:
            mt5_final = float(m.group(1))
    if mt5_final is not None:
        parser_total = df["total_pnl_usd"].sum()
        mt5_total = mt5_final - INITIAL_BALANCE
        diff = parser_total - mt5_total
        pct = abs(diff) / abs(mt5_total) * 100 if mt5_total else 0
        print(f"\n           [对账] MT5 final ${mt5_final:,.2f}  PnL ${mt5_total:+,.2f}")
        print(f"                  Parser PnL ${parser_total:+,.2f}  差 ${diff:+,.2f} ({pct:.2f}%)")
        if pct < 5:
            print(f"                  ✅ 对账通过 (< 5%)")
        else:
            print(f"                  ⚠️ 差异较大, 可能成本模型不一致")

    out = OUT_DIR / args.out
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n           saved -> {out}")
    if len(df) > 0:
        print(f"\n           reason 分布: {df['final_reason'].value_counts().to_dict()}")
        half_count = (df["half_exit_lot"] > 0).sum()
        print(f"           有半仓 trade: {half_count} / {len(df)}")
        print(f"           BUY/SELL: {(df['side']=='BUY').sum()}/{(df['side']=='SELL').sum()}")


if __name__ == "__main__":
    main()
