# -*- coding: utf-8 -*-
"""
拼接 XAUUSD daily 价格序列 (2024-01-01 ~ 2026-04-30)

来源:
    A. 本机 1m csv -> 日 K (2024-01-02 ~ 2025-12-31)
    B. Yahoo Finance GC=F daily close (2026-01-01 ~ 2026-04-30)

输出: _cache/xau_daily.csv  (Date, Open, High, Low, Close, source)
首次运行会拉网 + 重采样, 之后直接读缓存。
"""
import sys
import io
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import XAU_1M_CSV, CACHE_DIR, DATE_START, DATE_END

CACHE = CACHE_DIR / "xau_daily.csv"
YAHOO_CACHE = CACHE_DIR / "yahoo_gc_2026.csv"


def resample_1m_to_daily(src: Path, start: str, end: str) -> pd.DataFrame:
    """读 1m csv 重采样到日 K (使用 UTC date)。仅读所需窗口节省内存。"""
    print(f"  [A] 1m -> daily, 读取 {src.name} ...", flush=True)
    # 1m csv: Date;Open;High;Low;Close;Volume  分号分隔
    df = pd.read_csv(src, sep=";", encoding="utf-8")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
    mask = (df["Date"] >= start) & (df["Date"] <= end + " 23:59:59")
    df = df.loc[mask].copy()
    print(f"        过滤后 {len(df):,} 根 1m K 线")

    df["d"] = df["Date"].dt.date
    daily = df.groupby("d").agg(
        Open  = ("Open",   "first"),
        High  = ("High",   "max"),
        Low   = ("Low",    "min"),
        Close = ("Close",  "last"),
        Volume= ("Volume", "sum"),
    ).reset_index().rename(columns={"d": "Date"})
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["source"] = "MT5_1m_resample"
    print(f"        重采样 -> {len(daily)} 日 K")
    return daily


def fetch_yahoo_gc(start: str, end: str) -> pd.DataFrame:
    """yfinance 拉 GC=F daily, 仅 2026-01~04 这一段缺口。"""
    if YAHOO_CACHE.exists():
        print(f"  [B] Yahoo daily 命中缓存 {YAHOO_CACHE.name}", flush=True)
        df = pd.read_csv(YAHOO_CACHE, encoding="utf-8")
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    print(f"  [B] yfinance 拉 GC=F {start} ~ {end} ...", flush=True)
    import yfinance as yf
    raw = yf.download("GC=F", start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"yfinance 返回空数据 (GC=F {start} ~ {end})")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.reset_index().rename(columns={
        "Date": "Date", "Open": "Open", "High": "High",
        "Low": "Low", "Close": "Close", "Volume": "Volume",
    })
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
    raw["source"] = "yahoo_GC=F"
    raw.to_csv(YAHOO_CACHE, index=False, encoding="utf-8")
    print(f"        拉到 {len(raw)} 行 -> 缓存 {YAHOO_CACHE.name}")
    return raw[["Date", "Open", "High", "Low", "Close", "Volume", "source"]]


def build_daily_series(force: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not force:
        print(f"[price_series] 命中缓存 {CACHE.name}")
        df = pd.read_csv(CACHE, encoding="utf-8")
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    print(f"[price_series] 构建 daily 价格序列 {DATE_START} ~ {DATE_END}")
    part_a = resample_1m_to_daily(XAU_1M_CSV, DATE_START, "2025-12-31")
    part_b = fetch_yahoo_gc("2026-01-01", DATE_END)

    df = pd.concat([part_a, part_b], ignore_index=True).sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Date"], keep="first")

    # 偏置检查: 两段交界处价格不要跳变得离谱
    last_a = part_a["Close"].iloc[-1] if len(part_a) else None
    first_b = part_b["Close"].iloc[0] if len(part_b) else None
    if last_a and first_b:
        gap = abs(first_b - last_a) / last_a * 100
        print(f"  交界检查: 2025-12-31 close=${last_a:,.2f}  vs  Yahoo 首日 close=${first_b:,.2f}  Δ={gap:.2f}%")

    df.to_csv(CACHE, index=False, encoding="utf-8")
    print(f"  保存 -> {CACHE}")
    print(f"  总长度 {len(df)} 日, 范围 {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    return df


if __name__ == "__main__":
    df = build_daily_series(force=True)
    print("\n  前 3 行:")
    print(df.head(3).to_string(index=False))
    print("\n  后 3 行:")
    print(df.tail(3).to_string(index=False))
