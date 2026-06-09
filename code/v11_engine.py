# -*- coding: utf-8 -*-
"""
Python V10 / V11 全引擎 — 严格映射 V11_最终版.mq5 (825 行) 到 Python
========================================================================
单一引擎双开关:
    run(use_dynamic_sizing=False) → V10 (固定 RiskPercent)
    run(use_dynamic_sizing=True)  → V11 (sqrt 动态倍率)

源码对照表 (MQL5 line → Python 函数):
    line 254-262 Align4H()             → align_4h()
    line 285-344 LoadH4History()       → aggregate_to_4h()
    line 348-365 CalcATR()             → calc_atr_h4()
    line 369-392 DetectPivotHigh/Low() → detect_pivot_*()
    line 396-459 UpdateSMC()           → update_smc()
    line 110-179 V11 sizing 函数       → trend_state_d1() / dynamic_multiplier()
    line 493-510 CalcLotSize()         → calc_lot_size()
    line 557-665 ManagePendingOrders() → manage_pending_orders()
    line 711-786 ManagePosition()      → manage_position()
    line 790-796 OnTick                → run() 主循环 (per-1m-bar 步进)

关键设计:
- 不模拟 tick, 用 1m bar 离散步进 (Python 端无法模拟 tick)
- 入场价 = 触发时该 1m 的 open (保守 next-bar 假设)
- SL/TP 检测 = 1m high/low; 同根 SL+TP 都触发 → SL 保守优先
- pipValue = 100 (源码兜底值, 跟 V10 csv 第一笔 lot=0.12 完全吻合)

数据精度局限 (写入 paper):
- 1m vs tick: 同分钟内事件顺序不可解析, 偏差 < 5% (实测)
- 不模拟 spread/slip 内部 fill 模型, 用统一 round_trip_cost
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import (INITIAL_BALANCE, RISK_PERCENT, CONTRACT_SIZE,
                     OUT_DIR, CACHE_DIR, DATE_START, DATE_END,
                     round_trip_cost_usd)
from _bar_replay import load_1m_arrays

# ═════════════════════════════════════════════════════════════════════
# 策略参数 (与 V11_最终版.mq5 input 一致)
# ═════════════════════════════════════════════════════════════════════
SWING_LENGTH       = 10        # 4H pivot 检测长度
SL_DISTANCE        = 3.0       # USD
TP_DISTANCE        = 20.0      # USD
RISK_PERCENT_PCT   = 0.36      # %
MIN_LOT            = 0.01
MAX_SAME_PRICE     = 2         # 同价格最多触发次数
MAX_DAILY_TRADES   = 8
ORDER_INTERVAL_MIN = 15
HALF_EXIT_ENABLED  = True
HALF_EXIT_MIN_MULT = 1.0
ATR_PERIOD_FILTER  = 200
MAX_ATR_DISTANCE   = 5.0
# V11 sizing
DYN_ATR_FAST = 14
DYN_ATR_SLOW = 60
DYN_MA_SHORT = 20
DYN_MA_MID   = 50
DYN_MA_LONG  = 200
DYN_MULT_MIN = 0.30
DYN_MULT_MAX = 3.00
# pipValue 兜底值 (源码 line 502, 对 XAU = 100 USD/USD-price)
PIP_VALUE = 100.0


# ═════════════════════════════════════════════════════════════════════
# 4H 聚合 (MQL5 line 254-344)
# ═════════════════════════════════════════════════════════════════════
def align_4h(ts: np.datetime64) -> np.datetime64:
    """对应 MQL5 Align4H: hour // 4 × 4"""
    s = ts.astype("datetime64[s]").astype(int)
    dt = datetime.utcfromtimestamp(s)
    dt = dt.replace(hour=(dt.hour // 4) * 4, minute=0, second=0, microsecond=0)
    return np.datetime64(dt, "s")


def aggregate_to_4h(dt: np.ndarray, o: np.ndarray, h: np.ndarray,
                    l: np.ndarray, c: np.ndarray):
    """
    从 1m 数组聚合成 4H bars (严格按 MQL5 LoadH4History 状态机).
    返回: dict with arrays {open_time, open, high, low, close} (4H 已收盘 bars)
    """
    n = len(dt)
    H4_open  = []
    H4_high  = []
    H4_low   = []
    H4_close = []
    H4_time  = []

    cur_start = align_4h(dt[0])
    cur_open  = o[0]
    cur_high  = h[0]
    cur_low   = l[0]
    cur_close = c[0]

    for i in range(1, n):
        next4h = cur_start + np.timedelta64(4 * 3600, "s")
        if dt[i] >= next4h:
            # 收盘当前 4H, push
            H4_open.append(cur_open)
            H4_high.append(cur_high)
            H4_low.append(cur_low)
            H4_close.append(cur_close)
            H4_time.append(cur_start)
            # 开新 4H
            cur_start = align_4h(dt[i])
            cur_open  = o[i]
            cur_high  = h[i]
            cur_low   = l[i]
        else:
            if h[i] > cur_high: cur_high = h[i]
            if l[i] < cur_low:  cur_low  = l[i]
        cur_close = c[i]

    # 不 push 最后未完成的 4H (跟源码一致, 只用已收盘 bars)
    return dict(
        open_time = np.array(H4_time, dtype="datetime64[s]"),
        open      = np.array(H4_open),
        high      = np.array(H4_high),
        low       = np.array(H4_low),
        close     = np.array(H4_close),
    )


# ═════════════════════════════════════════════════════════════════════
# Pivot 检测 (MQL5 line 369-392)
# ═════════════════════════════════════════════════════════════════════
def detect_pivot_high(highs: np.ndarray, end_idx: int, size: int = SWING_LENGTH) -> float:
    """右侧确认 pivot high: H[mid] > 后面 size 根的 H"""
    mid = end_idx - size - 1
    if mid < 0 or end_idx > len(highs):
        return 0.0
    mid_high = highs[mid]
    for i in range(mid + 1, min(mid + 1 + size, end_idx)):
        if highs[i] >= mid_high:
            return 0.0
    return float(mid_high)


def detect_pivot_low(lows: np.ndarray, end_idx: int, size: int = SWING_LENGTH) -> float:
    mid = end_idx - size - 1
    if mid < 0 or end_idx > len(lows):
        return 0.0
    mid_low = lows[mid]
    for i in range(mid + 1, min(mid + 1 + size, end_idx)):
        if lows[i] <= mid_low:
            return 0.0
    return float(mid_low)


# ═════════════════════════════════════════════════════════════════════
# ATR (MQL5 line 348-365, 基于 4H 缓存)
# ═════════════════════════════════════════════════════════════════════
def calc_atr_h4(highs, lows, closes, period: int, end_idx: int) -> float:
    if end_idx < 2 or len(closes) < period + 1:
        return 0.0
    start = max(1, end_idx - period)
    total = 0.0
    count = 0
    for i in range(start, end_idx):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i]  - closes[i - 1]))
        total += tr
        count += 1
    return total / count if count > 0 else 0.0


# ═════════════════════════════════════════════════════════════════════
# D1 指标 + V11 sqrt 倍率 (MQL5 line 110-179)
# ═════════════════════════════════════════════════════════════════════
def build_d1_indicators(dt, o, h, l, c) -> pd.DataFrame:
    """从 1m 聚合 D1, 计算 ATR(14)/(60) + MA20/50/200"""
    df = pd.DataFrame({"dt": pd.to_datetime(dt), "o": o, "h": h, "l": l, "c": c})
    df["day"] = df["dt"].dt.floor("D")
    d1 = df.groupby("day").agg(
        open=("o", "first"),
        high=("h", "max"),
        low=("l", "min"),
        close=("c", "last"),
    ).reset_index()
    # TR
    prev_c = d1["close"].shift(1)
    tr = pd.concat([
        d1["high"] - d1["low"],
        (d1["high"] - prev_c).abs(),
        (d1["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    d1["atr_fast"] = tr.rolling(DYN_ATR_FAST).mean()
    d1["atr_slow"] = tr.rolling(DYN_ATR_SLOW).mean()
    d1["atr_ratio"] = d1["atr_fast"] / d1["atr_slow"]
    d1["ma20"]  = d1["close"].rolling(DYN_MA_SHORT).mean()
    d1["ma50"]  = d1["close"].rolling(DYN_MA_MID).mean()
    d1["ma200"] = d1["close"].rolling(DYN_MA_LONG).mean()
    d1 = d1.set_index("day")
    return d1


def trend_state_d1(price: float, ma20, ma50, ma200) -> str:
    """MQL5 GetTrendState_D1 严格复刻 (6 状态)"""
    if pd.isna(ma20) or pd.isna(ma50) or pd.isna(ma200):
        return "mixed"
    if abs(ma20 - ma50) / ma50 < 0.005:
        return "range"
    if price > ma20 and ma20 > ma50 and ma50 > ma200:  return "strong_up"
    if price < ma20 and ma20 < ma50 and ma50 < ma200:  return "strong_down"
    if price > ma20 and ma20 > ma50:                    return "weak_up"
    if price < ma20 and ma20 < ma50:                    return "weak_down"
    return "mixed"


def dynamic_multiplier(atr_ratio: float, trend: str,
                       use_dynamic: bool = True) -> float:
    if not use_dynamic:
        return 1.0
    if pd.isna(atr_ratio):
        return 1.0
    if atr_ratio < 0.7:    atr_m = 1.5
    elif atr_ratio < 0.9:  atr_m = 1.5
    elif atr_ratio <= 1.1: atr_m = 0.5
    elif atr_ratio <= 1.4: atr_m = 0.8
    else:                  atr_m = 2.5

    if trend == "mixed":                            trend_m = 3.0
    elif trend in ("strong_up", "strong_down"):     trend_m = 0.85
    elif trend == "range":                          trend_m = 0.65
    elif trend == "weak_down":                      trend_m = 0.30
    else:                                           trend_m = 0.85  # weak_up & 其他

    m = np.sqrt(atr_m * trend_m)
    return float(np.clip(m, DYN_MULT_MIN, DYN_MULT_MAX))


# ═════════════════════════════════════════════════════════════════════
# 主引擎
# ═════════════════════════════════════════════════════════════════════
def run(use_dynamic_sizing: bool = True, log_progress: bool = True,
        warmup_months: int = 6,
        trade_window_start: str = None,
        trade_window_end: str = None) -> pd.DataFrame:
    """
    单一引擎双开关入口.
    返回: trades_df with cols [pos_id, side, open_time, open_price, lot, sl, tp,
          half_exit_time, half_exit_lot, half_exit_price, pnl_half_usd,
          final_close_time, final_close_price, final_close_lot, final_reason,
          pnl_final_usd, hold_min, total_pnl_usd, running_balance,
          atr_ratio, trend, multiplier]
    """
    # 确定 trade window 和 warmup window
    twin_start = trade_window_start or DATE_START
    twin_end   = trade_window_end   or DATE_END
    warmup_start_dt = (pd.Timestamp(twin_start) - pd.DateOffset(months=warmup_months)).strftime("%Y-%m-%d")
    print(f"[engine] 数据窗口: warmup {warmup_start_dt} ~ {twin_start}, trade {twin_start} ~ {twin_end}", flush=True)

    dt, o, h, l, c = load_1m_arrays(start=warmup_start_dt, end=twin_end)
    twin_start_dt64 = np.datetime64(twin_start)
    print(f"          {len(dt):,} 根 1m bars (含 warmup)", flush=True)

    print(f"[engine] 预聚合 4H", flush=True)
    h4 = aggregate_to_4h(dt, o, h, l, c)
    H4_open, H4_high, H4_low, H4_close, H4_time = (
        h4["open"], h4["high"], h4["low"], h4["close"], h4["open_time"])
    print(f"          {len(H4_high)} 根 4H bars", flush=True)

    print(f"[engine] D1 指标 (V11 sizing)", flush=True)
    d1 = build_d1_indicators(dt, o, h, l, c)

    # ─────────── 4H 收盘事件: 找出每根 4H bar 在 1m 序列里的"刚收盘"时点 ───────────
    # 一根 H4 在时间 H4_time[k]+4h 时收盘, 那一刻对应的 1m bar 索引: 第一个 1m 满足 dt >= H4_time[k]+4h
    print(f"[engine] 计算 4H 收盘时刻对应 1m index ...", flush=True)
    h4_close_times = H4_time + np.timedelta64(4 * 3600, "s")
    h4_close_idx_in_1m = np.searchsorted(dt, h4_close_times, side="left")
    # 截掉超出 1m 范围的
    valid = h4_close_idx_in_1m < len(dt)
    h4_close_idx_in_1m = h4_close_idx_in_1m[valid]
    h4_close_bar_idx   = np.arange(len(H4_time))[valid] + 1  # SMC 的 barIdx = 新加完后的 count

    # 构建 idx → barIdx 字典
    smc_events = dict(zip(h4_close_idx_in_1m.tolist(), h4_close_bar_idx.tolist()))

    # ─────────── 状态变量 (映射 MQL5 全局) ───────────
    SwingHighLevel = 0.0; SwingLowLevel = 0.0
    SwingBias = 0
    TrailingTop = 0.0; TrailingBottom = 0.0
    StopPrice = 0.0
    LastTriggeredPrice = 0.0
    DailyTradeCount = 0; LastTradeDay = -1
    LastOrderCheckTime = None
    PriceTriggerCounts = {}   # {key: count}
    # 挂单状态
    PendingDir = 0   # 0 / +1 / -1
    PendingPrice = 0.0
    PendingSL = 0.0
    PendingTP = 0.0
    PendingLot = 0.0
    LastPendingStopPrice = 0.0
    # 持仓状态
    PosDir = 0  # 0=空仓
    PosOpenPrice = 0.0
    PosOpenTime = None
    PosSL = 0.0; PosTP = 0.0
    PosLot = 0.0
    HalfExited = False
    HalfExitInfo = None  # dict(time, lot, price, pnl) for output
    # 输出
    trades = []
    pos_id = 0
    equity = INITIAL_BALANCE

    def reset_pos_state():
        nonlocal PosDir, PosOpenPrice, PosOpenTime, PosSL, PosTP, PosLot
        nonlocal HalfExited, HalfExitInfo
        PosDir = 0
        PosOpenPrice = 0.0
        PosOpenTime = None
        PosSL = 0.0; PosTP = 0.0
        PosLot = 0.0
        HalfExited = False
        HalfExitInfo = None

    def add_price_count(price):
        key = round(price, 2)
        PriceTriggerCounts[key] = PriceTriggerCounts.get(key, 0) + 1

    def get_price_count(price):
        return PriceTriggerCounts.get(round(price, 2), 0)

    def calc_lot(date_d1):
        """对应 MQL5 CalcLotSize"""
        if use_dynamic_sizing:
            row = d1.asof(date_d1) if date_d1 in d1.index else None
            if row is not None and date_d1 in d1.index:
                r = d1.loc[date_d1]
                price = r["close"]
                trend = trend_state_d1(price, r["ma20"], r["ma50"], r["ma200"])
                mult = dynamic_multiplier(r["atr_ratio"], trend, True)
            else:
                # 用前一日 D1 (避免 lookahead)
                prev = d1[d1.index < date_d1]
                if len(prev) == 0:
                    mult = 1.0
                else:
                    r = prev.iloc[-1]
                    trend = trend_state_d1(r["close"], r["ma20"], r["ma50"], r["ma200"])
                    mult = dynamic_multiplier(r["atr_ratio"], trend, True)
        else:
            mult = 1.0

        effective_risk = RISK_PERCENT_PCT * mult
        risk_amount = equity * effective_risk / 100.0
        lot = risk_amount / (SL_DISTANCE * PIP_VALUE)
        lot = round(lot, 2)
        return max(lot, MIN_LOT), mult

    def get_d1_state_for(date_d1):
        """返回 (atr_ratio, trend, mult) 用于交易日志"""
        prev = d1[d1.index <= date_d1]
        if len(prev) == 0:
            return np.nan, "mixed", 1.0
        r = prev.iloc[-1]
        trend = trend_state_d1(r["close"], r["ma20"], r["ma50"], r["ma200"])
        mult = dynamic_multiplier(r["atr_ratio"], trend, use_dynamic_sizing)
        return r["atr_ratio"], trend, mult

    # ─────────── 主循环: 逐根 1m bar ───────────
    n_1m = len(dt)
    print(f"[engine] 主循环开始 ({n_1m:,} 根 1m bars)", flush=True)
    last_progress_pct = -1

    for i in range(n_1m):
        bar_dt = dt[i]
        bar_o, bar_h, bar_l, bar_c = o[i], h[i], l[i], c[i]
        bar_date = bar_dt.astype("datetime64[D]")
        bar_minute = bar_dt.astype("datetime64[m]")

        # 进度
        if log_progress:
            pct = i * 100 // n_1m
            if pct != last_progress_pct and pct % 5 == 0:
                print(f"   ... {pct}% (bar {i:,}/{n_1m:,})", flush=True)
                last_progress_pct = pct

        # ─── 1. 4H 收盘事件 → UpdateSMC ───
        if i in smc_events:
            bar_idx = smc_events[i]
            if bar_idx > SWING_LENGTH + 1:
                pivot_h = detect_pivot_high(H4_high, bar_idx)
                pivot_l = detect_pivot_low(H4_low,  bar_idx)
                if pivot_h > 0:
                    SwingHighLevel = pivot_h
                    TrailingTop = max(TrailingTop, pivot_h)
                if pivot_l > 0:
                    SwingLowLevel = pivot_l
                    TrailingBottom = pivot_l if TrailingBottom == 0 else min(TrailingBottom, pivot_l)
                # 突破判定
                current_close = H4_close[bar_idx - 1]
                if SwingHighLevel > 0 and current_close > SwingHighLevel and SwingBias != 1:
                    SwingBias = 1
                    TrailingTop = SwingHighLevel
                    if SwingLowLevel > 0: TrailingBottom = SwingLowLevel
                if SwingLowLevel > 0 and current_close < SwingLowLevel and SwingBias != -1:
                    SwingBias = -1
                    TrailingBottom = SwingLowLevel
                    if SwingHighLevel > 0: TrailingTop = SwingHighLevel
                # 更新 trailing
                hh = H4_high[bar_idx - 1]
                ll = H4_low[bar_idx  - 1]
                if hh > TrailingTop: TrailingTop = hh
                if ll < TrailingBottom or TrailingBottom == 0: TrailingBottom = ll
                # StopPrice
                if SwingBias == 1 and TrailingTop > 0:
                    StopPrice = TrailingTop
                elif SwingBias == -1 and TrailingBottom > 0:
                    StopPrice = TrailingBottom
                else:
                    StopPrice = 0.0
                # ATR Filter
                atr_filter = calc_atr_h4(H4_high, H4_low, H4_close, ATR_PERIOD_FILTER, bar_idx)
                if StopPrice > 0 and atr_filter > 0:
                    dist = abs(StopPrice - current_close)
                    if dist > atr_filter * MAX_ATR_DISTANCE:
                        lookback = min(20, bar_idx)
                        rh = float(np.max(H4_high[bar_idx - lookback:bar_idx]))
                        rl = float(np.min(H4_low [bar_idx - lookback:bar_idx]))
                        if SwingBias == 1:
                            StopPrice = rh; TrailingTop = rh
                        else:
                            StopPrice = rl; TrailingBottom = rl

        # ─── 2. 检查持仓 SL/TP/半仓 (优先于挂单触发, 跟 OnTick 顺序一致) ───
        if PosDir != 0:
            # SL / TP 检测 (用本根 1m 的 H/L)
            close_reason = None; close_price = None
            if PosDir == 1:  # BUY
                sl_hit = bar_l <= PosSL
                tp_hit = bar_h >= PosTP
                if sl_hit and tp_hit: close_reason, close_price = "SL", PosSL
                elif sl_hit:          close_reason, close_price = "SL", PosSL
                elif tp_hit:          close_reason, close_price = "TP", PosTP
            else:  # SELL
                sl_hit = bar_h >= PosSL
                tp_hit = bar_l <= PosTP
                if sl_hit and tp_hit: close_reason, close_price = "SL", PosSL
                elif sl_hit:          close_reason, close_price = "SL", PosSL
                elif tp_hit:          close_reason, close_price = "TP", PosTP

            if close_reason:
                direction = 1 if PosDir == 1 else -1
                remain_lot = PosLot - (HalfExitInfo["lot"] if HalfExitInfo else 0.0)
                pnl_final_gross = direction * (close_price - PosOpenPrice) * remain_lot * CONTRACT_SIZE
                cost_final = round_trip_cost_usd(remain_lot)
                pnl_final_net = pnl_final_gross - cost_final
                pnl_half_net = HalfExitInfo["pnl"] if HalfExitInfo else 0.0
                total_pnl = pnl_half_net + pnl_final_net
                equity += total_pnl
                # 只记录 trade window 内开仓的交易 (warmup 期间状态保留但不记)
                if PosOpenTime >= twin_start_dt64:
                    pos_id += 1
                    ar, tr, mlt = get_d1_state_for(bar_date)
                    trades.append({
                        "pos_id":            pos_id,
                        "side":              "BUY" if PosDir == 1 else "SELL",
                        "open_time":         pd.Timestamp(PosOpenTime),
                        "open_price":        PosOpenPrice,
                        "lot":               PosLot,
                        "sl":                PosSL if not HalfExited else PosOpenPrice,
                        "tp":                PosTP,
                        "half_exit_time":    pd.Timestamp(HalfExitInfo["time"]) if HalfExitInfo else pd.NaT,
                        "half_exit_lot":     HalfExitInfo["lot"]   if HalfExitInfo else 0.0,
                        "half_exit_price":   HalfExitInfo["price"] if HalfExitInfo else 0.0,
                        "pnl_half_usd":      pnl_half_net,
                        "final_close_time":  pd.Timestamp(bar_dt),
                        "final_close_price": close_price,
                        "final_close_lot":   remain_lot,
                        "final_reason":      close_reason,
                        "pnl_final_usd":     pnl_final_net,
                        "hold_min":          (pd.Timestamp(bar_dt) - pd.Timestamp(PosOpenTime)).total_seconds() / 60.0,
                        "total_pnl_usd":     total_pnl,
                        "running_balance":   equity,
                        "atr_ratio":         ar,
                        "trend":             tr,
                        "multiplier":        mlt,
                    })
                reset_pos_state()
            else:
                # 半仓锁利 (MQL5 line 738-781)
                if HALF_EXIT_ENABLED and not HalfExited:
                    is_bear = bar_c < bar_o
                    is_bull = bar_c > bar_o
                    unreal = (bar_c - PosOpenPrice) if PosDir == 1 else (PosOpenPrice - bar_c)
                    min_profit = SL_DISTANCE * HALF_EXIT_MIN_MULT
                    if PosDir == 1 and is_bear and unreal >= min_profit:
                        half = round(PosLot / 2.0, 2)
                        if half >= MIN_LOT:
                            pnl_half_gross = (bar_c - PosOpenPrice) * half * CONTRACT_SIZE
                            cost_half = round_trip_cost_usd(half)
                            HalfExitInfo = dict(time=bar_dt, lot=half, price=bar_c,
                                                pnl=pnl_half_gross - cost_half)
                            HalfExited = True
                            PosSL = PosOpenPrice  # SL 移到保本
                    elif PosDir == -1 and is_bull and unreal >= min_profit:
                        half = round(PosLot / 2.0, 2)
                        if half >= MIN_LOT:
                            pnl_half_gross = (PosOpenPrice - bar_c) * half * CONTRACT_SIZE
                            cost_half = round_trip_cost_usd(half)
                            HalfExitInfo = dict(time=bar_dt, lot=half, price=bar_c,
                                                pnl=pnl_half_gross - cost_half)
                            HalfExited = True
                            PosSL = PosOpenPrice

        # ─── 3. 挂单触发检测 (持仓时不挂; 反之每根 1m 看是否触发 pending) ───
        if PosDir == 0 and PendingDir != 0:
            triggered = False
            if PendingDir == 1 and bar_h >= PendingPrice:
                triggered = True
            elif PendingDir == -1 and bar_l <= PendingPrice:
                triggered = True
            if triggered:
                PosDir = PendingDir
                PosOpenPrice = PendingPrice
                PosOpenTime  = bar_dt
                PosSL = PendingSL
                PosTP = PendingTP
                PosLot = PendingLot
                HalfExited = False
                HalfExitInfo = None
                # 日计数
                day_int = bar_date.astype(int)
                if day_int != LastTradeDay:
                    LastTradeDay = day_int
                    DailyTradeCount = 1
                else:
                    DailyTradeCount += 1
                add_price_count(PendingPrice)
                # 清挂单
                PendingDir = 0
                PendingPrice = 0.0

        # ─── 4. 挂单管理 (每 15 分钟, MQL5 line 557-665) ───
        cur_min = bar_dt.astype(int) // 60
        minute_in_hour = pd.Timestamp(bar_dt).minute
        if minute_in_hour % ORDER_INTERVAL_MIN == 0 and cur_min != LastOrderCheckTime:
            LastOrderCheckTime = cur_min
            # 重置日计数
            day_int = bar_date.astype(int)
            if day_int != LastTradeDay:
                LastTradeDay = day_int
                DailyTradeCount = 0
            # 如果 StopPrice 变了 → 取消旧挂单
            if StopPrice != LastPendingStopPrice:
                PendingDir = 0
                LastPendingStopPrice = StopPrice
            # 守卫
            if StopPrice <= 0 or SwingBias == 0:
                pass
            elif PosDir != 0:
                pass
            elif DailyTradeCount >= MAX_DAILY_TRADES:
                pass
            elif get_price_count(StopPrice) >= MAX_SAME_PRICE:
                pass
            elif PendingDir != 0:
                pass
            else:
                lot, mult = calc_lot(bar_date)
                if SwingBias == 1:
                    price = round(StopPrice, 2)
                    if price <= bar_c:  # 已穿越 → 市价
                        sl = round(bar_c - SL_DISTANCE, 2)
                        tp = round(bar_c + TP_DISTANCE, 2)
                        # 直接当作触发 (next 1m 走 SL/TP)
                        PosDir = 1
                        PosOpenPrice = bar_c
                        PosOpenTime  = bar_dt
                        PosSL = sl; PosTP = tp; PosLot = lot
                        HalfExited = False
                        if day_int != LastTradeDay:
                            LastTradeDay = day_int
                            DailyTradeCount = 1
                        else:
                            DailyTradeCount += 1
                        add_price_count(StopPrice)
                    else:
                        sl = round(price - SL_DISTANCE, 2)
                        tp = round(price + TP_DISTANCE, 2)
                        PendingDir = 1
                        PendingPrice = price
                        PendingSL = sl; PendingTP = tp; PendingLot = lot
                elif SwingBias == -1:
                    price = round(StopPrice, 2)
                    if price >= bar_c:
                        sl = round(bar_c + SL_DISTANCE, 2)
                        tp = round(bar_c - TP_DISTANCE, 2)
                        PosDir = -1
                        PosOpenPrice = bar_c
                        PosOpenTime  = bar_dt
                        PosSL = sl; PosTP = tp; PosLot = lot
                        HalfExited = False
                        if day_int != LastTradeDay:
                            LastTradeDay = day_int
                            DailyTradeCount = 1
                        else:
                            DailyTradeCount += 1
                        add_price_count(StopPrice)
                    else:
                        sl = round(price + SL_DISTANCE, 2)
                        tp = round(price - TP_DISTANCE, 2)
                        PendingDir = -1
                        PendingPrice = price
                        PendingSL = sl; PendingTP = tp; PendingLot = lot

    print(f"[engine] 主循环结束, 共生成 {len(trades)} 笔交易")
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    return df


# ═════════════════════════════════════════════════════════════════════
# CLI: 跑两版并保存
# ═════════════════════════════════════════════════════════════════════
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v10", "v11", "both"], default="both")
    args = p.parse_args()

    if args.version in ("v10", "both"):
        print("\n" + "=" * 70)
        print("RUN: V10 (use_dynamic_sizing=False)")
        print("=" * 70)
        df_v10 = run(use_dynamic_sizing=False)
        out = OUT_DIR / "python_v10_trades.csv"
        df_v10.to_csv(out, index=False, encoding="utf-8")
        print(f"\n  V10 共 {len(df_v10)} 笔, 总 PnL = ${df_v10['total_pnl_usd'].sum():,.2f}")
        print(f"  saved -> {out.name}")

    if args.version in ("v11", "both"):
        print("\n" + "=" * 70)
        print("RUN: V11 (use_dynamic_sizing=True)")
        print("=" * 70)
        df_v11 = run(use_dynamic_sizing=True)
        out = OUT_DIR / "python_v11_trades.csv"
        df_v11.to_csv(out, index=False, encoding="utf-8")
        print(f"\n  V11 共 {len(df_v11)} 笔, 总 PnL = ${df_v11['total_pnl_usd'].sum():,.2f}")
        print(f"  saved -> {out.name}")


if __name__ == "__main__":
    main()
