# -*- coding: utf-8 -*-
"""
1-minute bar SL/TP 回放器 (供 Baseline 4 / 5 共用)
========================================================================
功能:
- 一次加载并缓存指定时段的 1m XAUUSD OHLC 到 numpy 数组
- 提供 replay_sl_tp(entry_dt, side, open_price, sl_price, tp_price,
                    max_minutes) -> (close_dt, close_price, reason, hold_min)
- 在每根 1m 内, 同时触发 SL 与 TP 时按保守规则: SL 优先 (符合
  Bailey & López de Prado 2014 的最差路径假设, 论文里也这么写)

数据精度局限 (写进论文 Section 3.4):
- 1m bar 内不能解析 SL 与 TP 谁先到, 偏差上界由 1m 内高低差决定
- 周末/隔夜跳空跨多分钟时, 用首根有效 bar 的 open 作为成交价
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import XAU_1M_CSV, CACHE_DIR, DATE_START, DATE_END

CACHE = CACHE_DIR / "xau_1m_array.npz"


def load_1m_arrays(force: bool = False, start: str = None, end: str = None):
    """
    加载 1m OHLC 为 numpy 数组. 缓存包含 [配置 DATE_START - 12 月预热, DATE_END] 的全部数据,
    便于 warmup; caller 可传 start/end 二次切片. 首次需 ~60s.
    """
    if CACHE.exists() and not force:
        data = np.load(CACHE, allow_pickle=False)
        dt_arr, o_arr, h_arr, l_arr, c_arr = data["dt"], data["o"], data["h"], data["l"], data["c"]
    else:
        print(f"[bar_replay] 首次加载 1m csv ({XAU_1M_CSV.name}) ...", flush=True)
        df = pd.read_csv(XAU_1M_CSV, sep=";", encoding="utf-8")
        df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
        # 缓存包含 DATE_START 之前 12 个月预热数据
        warmup_start = (pd.Timestamp(DATE_START) - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
        mask = (df["Date"] >= warmup_start) & (df["Date"] <= DATE_END + " 23:59:59")
        df = df.loc[mask].sort_values("Date").reset_index(drop=True)
        print(f"            截取窗口 (含 12 月预热) {len(df):,} 根 1m bar")

        dt_arr = df["Date"].values.astype("datetime64[s]")
        o_arr = df["Open"].values.astype(np.float64)
        h_arr = df["High"].values.astype(np.float64)
        l_arr = df["Low"].values.astype(np.float64)
        c_arr = df["Close"].values.astype(np.float64)
        np.savez(CACHE, dt=dt_arr, o=o_arr, h=h_arr, l=l_arr, c=c_arr)
        print(f"            缓存 -> {CACHE.name}")

    # 可选二次切片 (caller 传 start/end)
    if start is not None or end is not None:
        s = np.datetime64(start) if start else dt_arr[0]
        e = np.datetime64(end + " 23:59:59") if end else dt_arr[-1]
        mask = (dt_arr >= s) & (dt_arr <= e)
        return dt_arr[mask], o_arr[mask], h_arr[mask], l_arr[mask], c_arr[mask]
    return dt_arr, o_arr, h_arr, l_arr, c_arr


def _find_start_idx(dt_array: np.ndarray, entry_dt: np.datetime64) -> int:
    """二分查找首根 dt >= entry_dt 的 bar 索引. 找不到返回 -1."""
    idx = int(np.searchsorted(dt_array, entry_dt, side="left"))
    if idx >= len(dt_array):
        return -1
    return idx


def replay_sl_tp(
    dt: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    entry_dt: np.datetime64,
    side: int,                 # 1 = BUY, -1 = SELL
    open_price: float,
    sl_price: float,
    tp_price: float,
    max_minutes: int = 1440,   # 默认最长持仓 24h
):
    """
    从 entry_dt 起, 在 1m 数据上往后扫, 谁先碰 SL 或 TP 谁触发。
    返回: dict(close_dt, close_price, reason in {'SL','TP','TIMEOUT'}, hold_min)
    若 entry_dt 超出数据范围或 max_minutes 内未触发 → 'TIMEOUT' + 最末根 close.
    """
    start = _find_start_idx(dt, entry_dt)
    if start < 0:
        return None
    end = min(start + max_minutes, len(dt) - 1)
    seg_h = h[start:end + 1]
    seg_l = l[start:end + 1]

    if side == 1:  # BUY: low <= sl -> SL ; high >= tp -> TP
        sl_hit = seg_l <= sl_price
        tp_hit = seg_h >= tp_price
    else:          # SELL: high >= sl -> SL ; low <= tp -> TP
        sl_hit = seg_h >= sl_price
        tp_hit = seg_l <= tp_price

    any_hit = sl_hit | tp_hit
    if not any_hit.any():
        # TIMEOUT: 用 max_minutes 末根 close 平仓
        close_dt = dt[end]
        close_price = c[end]
        hold = int((close_dt - entry_dt) / np.timedelta64(1, "m"))
        return dict(close_dt=close_dt, close_price=close_price, reason="TIMEOUT", hold_min=hold)

    first_idx = int(np.argmax(any_hit))   # 第一个 True 的位置
    # 若同根同时触发 SL+TP → 保守 SL 优先
    if sl_hit[first_idx]:
        reason = "SL"; close_price = sl_price
    else:
        reason = "TP"; close_price = tp_price
    close_dt = dt[start + first_idx]
    hold = int((close_dt - entry_dt) / np.timedelta64(1, "m"))
    return dict(close_dt=close_dt, close_price=close_price, reason=reason, hold_min=hold)


if __name__ == "__main__":
    dt, o, h, l, c = load_1m_arrays(force=False)
    print(f"  数组长度 {len(dt):,} 根 1m bar")
    print(f"  范围 {pd.Timestamp(dt[0])} ~ {pd.Timestamp(dt[-1])}")
    # 自测: 2024-01-02 18:00 SELL @ 2058.31, SL=2061.31, TP=2038.31
    test_dt = np.datetime64("2024-01-02T18:00:10", "s")
    r = replay_sl_tp(dt, h, l, c, test_dt, side=-1, open_price=2058.31, sl_price=2061.31, tp_price=2038.31)
    print(f"  测试回放: {r}")
