# -*- coding: utf-8 -*-
"""
扩展版 daily K (含 200 日 MA 预热): 2022-06-01 ~ 2026-04-30
仅 baseline_3 (MA cross + V11 sizing) 用得到。
独立缓存 _cache/xau_daily_extended.csv 避免污染 baseline_1 用的窗口数据。
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import XAU_1M_CSV, CACHE_DIR
from _price_series import resample_1m_to_daily, fetch_yahoo_gc

EXT_CACHE = CACHE_DIR / "xau_daily_extended.csv"
WARMUP_START = "2022-06-01"   # 250 日预热 + 200 日 MA


def build_daily_extended(force: bool = False) -> pd.DataFrame:
    if EXT_CACHE.exists() and not force:
        df = pd.read_csv(EXT_CACHE, encoding="utf-8")
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    print(f"[daily_extended] 重采样 {WARMUP_START} ~ 2025-12-31 + Yahoo 2026Q1+")
    part_a = resample_1m_to_daily(XAU_1M_CSV, WARMUP_START, "2025-12-31")
    part_b = fetch_yahoo_gc("2026-01-01", "2026-04-30")
    df = pd.concat([part_a, part_b], ignore_index=True).sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Date"], keep="first")
    df.to_csv(EXT_CACHE, index=False, encoding="utf-8")
    print(f"  保存 -> {EXT_CACHE.name}  ({len(df)} 日)")
    return df


if __name__ == "__main__":
    df = build_daily_extended(force=True)
    print(df.head(2).to_string(index=False))
    print(df.tail(2).to_string(index=False))
