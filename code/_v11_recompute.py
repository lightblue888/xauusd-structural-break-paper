# -*- coding: utf-8 -*-
"""
论文级 V11 重算: 从 V10 csv 取每笔交易的 atr_ratio + trend (来自 V11_老 csv),
应用 MQL5 V11_最终版.mq5 里的 sqrt 公式生成 multiplier, 重新计算每笔 PnL。

输出: _cache/V11_sqrt.csv  (论文 ground truth 的 V11 数据)

公式 (源码 line 158-176, 825 行权威版):
    atr_m:
        atr < 0.7   -> 1.5
        atr < 0.9   -> 1.5
        atr <= 1.1  -> 0.5
        atr <= 1.4  -> 0.8
        atr > 1.4   -> 2.5   (增强: 从 1.5 提到 2.5)
    trend_m:
        mixed           -> 3.0    (增强: 从 2.0 提到 3.0)
        strong_up/down  -> 0.85
        range           -> 0.65
        weak_down       -> 0.30
        else            -> 0.85
    multiplier = clip(sqrt(atr_m * trend_m), 0.30, 3.00)
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import V10_CSV, V11_OLD_CSV, CACHE_DIR

OUT = CACHE_DIR / "V11_sqrt.csv"
MULT_MIN, MULT_MAX = 0.30, 3.00


def atr_m_lookup(atr: float) -> float:
    """V11 MQL5 atr_m 查表 (增强规则)"""
    if pd.isna(atr):
        return np.nan
    if atr < 0.7:    return 1.5
    if atr < 0.9:    return 1.5
    if atr <= 1.1:   return 0.5
    if atr <= 1.4:   return 0.8
    return 2.5


def trend_m_lookup(trend: str) -> float:
    """V11 MQL5 trend_m 查表 (增强规则)"""
    if trend == "mixed":                              return 3.0
    if trend in ("strong_up", "strong_down"):         return 0.85
    if trend == "range":                              return 0.65
    if trend == "weak_down":                          return 0.30
    return 0.85  # weak_up / 其他默认


def v11_sqrt_multiplier(atr_ratio: float, trend: str) -> float:
    """合成 sqrt 倍率, 夹紧到 [0.30, 3.00]"""
    a = atr_m_lookup(atr_ratio)
    t = trend_m_lookup(trend)
    if pd.isna(a) or pd.isna(t):
        return 1.0
    m = float(np.sqrt(a * t))
    return float(np.clip(m, MULT_MIN, MULT_MAX))


def main():
    # 主数据用 V10 (干净), atr_ratio + trend 从 V11_老 csv 取
    print(f"[1/3] 加载 V10 ({V10_CSV.name})")
    v10 = pd.read_csv(V10_CSV, encoding="utf-8-sig")
    print(f"      {len(v10)} 笔")

    print(f"[2/3] 加载 V11_老 csv 抽 atr_ratio + trend 列")
    v11_old = pd.read_csv(V11_OLD_CSV, encoding="utf-8-sig")
    keep = ["pos_id", "atr_ratio", "trend", "fed", "dxy", "vol_ratio"]
    v11_feat = v11_old[keep]
    df = v10.merge(v11_feat, on="pos_id", how="left")
    miss = df["atr_ratio"].isna().sum()
    print(f"      atr_ratio 缺失 {miss} 笔 (将用 multiplier=1.0)")

    print(f"[3/3] 应用 sqrt 公式")
    df["mult_sqrt"] = [v11_sqrt_multiplier(a, t) for a, t in zip(df["atr_ratio"], df["trend"])]
    df["atr_m"]    = df["atr_ratio"].apply(atr_m_lookup)
    df["trend_m"]  = df["trend"].apply(trend_m_lookup)
    df["pnl_v10"]      = pd.to_numeric(df["total_pnl_usd"], errors="coerce")
    df["pnl_v11_sqrt"] = df["pnl_v10"] * df["mult_sqrt"]

    # 统计
    print(f"\n  multiplier 描述:")
    print(df["mult_sqrt"].describe().to_string())
    print(f"\n  V10 总 PnL:        ${df['pnl_v10'].sum():,.2f}")
    print(f"  V11(sqrt) 总 PnL:  ${df['pnl_v11_sqrt'].sum():,.2f}")
    delta = df["pnl_v11_sqrt"].sum() - df["pnl_v10"].sum()
    print(f"  差异:              ${delta:+,.2f}  ({delta / abs(df['pnl_v10'].sum()) * 100:+.2f}%)")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n  保存 -> {OUT}")
    return df


if __name__ == "__main__":
    main()
