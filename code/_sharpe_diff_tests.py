# -*- coding: utf-8 -*-
"""
Sharpe 差异显著性检验 (reviewer #1/#2)
========================================================================
对配对策略做 Sharpe 差异的 stationary block bootstrap 检验 (Politis-Romano 1994,
跟论文 §4.5 一致)。配对重采样 (同一组 block 索引同时作用于两条日收益序列) 以保留
两策略间的强相关 (同信号不同 sizing)。

输出: 每个比较的 Sharpe_A, Sharpe_B, 差异, 95% CI, bootstrap 双侧 p, 是否显著。
- (c): V10 vs {V11, ATR_only, Trend_only, VolTargeting}
- (b): V11 vs V11_FixedR (mode2)
- (d): V11 vs V11_FixedR 在 mode1 与 mode2
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import INITIAL_BALANCE, OUT_DIR

ANN = 252
B = 5000              # bootstrap 次数
BLOCK_L = 10          # 期望 block 长度 (交易日)
SEED = 12345


def parse_dt(s):
    out = pd.to_datetime(s, format="%Y.%m.%d %H:%M:%S", errors="coerce")
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(s, errors="coerce")
    return out


def daily_pnl(path):
    df = pd.read_csv(path)
    ct = parse_dt(df["final_close_time"])
    pnl = pd.to_numeric(df["total_pnl_usd"], errors="coerce")
    d = pd.DataFrame({"d": ct.dt.normalize(), "pnl": pnl}).dropna()
    return d.groupby("d")["pnl"].sum().sort_index()


def returns_on_grid(dp, idx):
    p = dp.reindex(idx, fill_value=0.0)
    eq = INITIAL_BALANCE + p.cumsum()
    eq_prev = eq.shift(1).fillna(INITIAL_BALANCE)
    return (p / eq_prev).values


def sharpe(r):
    sd = r.std(ddof=1)
    return np.sqrt(ANN) * r.mean() / sd if sd > 0 else np.nan


def stationary_idx(n, L, rng):
    p = 1.0 / L
    idx = np.empty(n, dtype=np.int64)
    i = rng.integers(0, n)
    rand = rng.random(n)
    for t in range(n):
        idx[t] = i
        i = rng.integers(0, n) if rand[t] < p else (i + 1) % n
    return idx


def compare(name_a, path_a, name_b, path_b, rng):
    da, db = daily_pnl(path_a), daily_pnl(path_b)
    idx = da.index.union(db.index)
    ra, rb = returns_on_grid(da, idx), returns_on_grid(db, idx)
    n = len(idx)
    sa, sb = sharpe(ra), sharpe(rb)
    obs = sa - sb
    diffs = np.empty(B)
    for b in range(B):
        bi = stationary_idx(n, BLOCK_L, rng)
        diffs[b] = sharpe(ra[bi]) - sharpe(rb[bi])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # 双侧 bootstrap p: 围绕 0 的不对称程度
    frac_neg = np.mean(diffs < 0)
    p = 2 * min(frac_neg, 1 - frac_neg)
    sig = "YES" if (lo > 0 or hi < 0) else "no"
    return dict(comp=f"{name_a} − {name_b}", Sa=round(sa, 3), Sb=round(sb, 3),
                diff=round(obs, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                p=round(p, 3), sig=sig, n_days=n)


def main():
    D = OUT_DIR
    rng = np.random.default_rng(SEED)
    rows = []

    print("=== (c) V10 vs 各 sizing 规则 (mode2) ===")
    for nm, f in [("V11", "v11_realticks_trades.csv"),
                  ("ATR_Only", "v11_atr_only_realticks_trades.csv"),
                  ("Trend_Only", "v11_trend_only_realticks_trades.csv"),
                  ("VolTarget", "v11_voltargeting_realticks_trades.csv")]:
        r = compare("V10", D / "v10_realticks_trades.csv", nm, D / f, rng)
        rows.append(r); print(r)

    print("\n=== (b) V11 vs V11_FixedR (mode2) ===")
    r = compare("V11", D / "v11_realticks_trades.csv",
                "V11_FixedR", D / "v11_fixedr_realticks_trades.csv", rng)
    rows.append(r); print(r)

    print("\n=== (d) V11 vs V11_FixedR: mode1 与 mode2 ===")
    r1 = compare("V11_m1", D / "v11_mt5_trades.csv",
                 "V11_FixedR_m1", D / "v11_fixedr_trades.csv", rng)
    rows.append(r1); print(r1)
    r2 = compare("V11_m2", D / "v11_realticks_trades.csv",
                 "V11_FixedR_m2", D / "v11_fixedr_realticks_trades.csv", rng)
    rows.append(r2); print(r2)

    out = pd.DataFrame(rows)
    out.to_csv(D / "sharpe_diff_tests.csv", index=False, encoding="utf-8")
    print("\nsaved -> sharpe_diff_tests.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
