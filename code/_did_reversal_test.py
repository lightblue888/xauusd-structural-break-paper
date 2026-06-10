# -*- coding: utf-8 -*-
"""
(d) 反转的 difference-in-differences 检验
========================================================================
反转幅度 DiD = (Sharpe_V11 - Sharpe_FixedR)_mode2 - (Sharpe_V11 - Sharpe_FixedR)_mode1
4 条日收益序列放在同一日历网格, stationary block bootstrap 用**同一组 block 索引**
同时作用于 4 条序列 (保留 mode1↔mode2 同交易的强相关 → DiD 差掉共同方差)。
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import INITIAL_BALANCE, OUT_DIR

ANN = 252
B = 10000
BLOCK_L = 10
SEED = 2024


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


def main():
    D = OUT_DIR
    paths = {
        "V11_m1": D / "v11_mt5_trades.csv",
        "FixedR_m1": D / "v11_fixedr_trades.csv",
        "V11_m2": D / "v11_realticks_trades.csv",
        "FixedR_m2": D / "v11_fixedr_realticks_trades.csv",
    }
    dps = {k: daily_pnl(p) for k, p in paths.items()}
    idx = dps["V11_m1"].index
    for k in dps:
        idx = idx.union(dps[k].index)
    R = {k: returns_on_grid(dps[k], idx) for k in dps}
    n = len(idx)

    def did(r):
        return (sharpe(r["V11_m2"]) - sharpe(r["FixedR_m2"])) - \
               (sharpe(r["V11_m1"]) - sharpe(r["FixedR_m1"]))

    s = {k: sharpe(R[k]) for k in R}
    gap_m1 = s["V11_m1"] - s["FixedR_m1"]
    gap_m2 = s["V11_m2"] - s["FixedR_m2"]
    obs = gap_m2 - gap_m1

    print(f"n_days={n}")
    print(f"Sharpe: V11_m1={s['V11_m1']:.3f} FixedR_m1={s['FixedR_m1']:.3f} "
          f"V11_m2={s['V11_m2']:.3f} FixedR_m2={s['FixedR_m2']:.3f}")
    print(f"gap mode1 (V11-FixedR) = {gap_m1:+.3f}")
    print(f"gap mode2 (V11-FixedR) = {gap_m2:+.3f}")
    print(f"DiD (reversal magnitude) = {obs:+.3f}")

    rng = np.random.default_rng(SEED)
    dids = np.empty(B)
    for b in range(B):
        bi = stationary_idx(n, BLOCK_L, rng)
        rb = {k: R[k][bi] for k in R}
        dids[b] = did(rb)
    lo, hi = np.percentile(dids, [2.5, 97.5])
    frac_neg = np.mean(dids < 0)
    p_two = 2 * min(frac_neg, 1 - frac_neg)
    p_one = frac_neg  # H1: DiD>0  → p = P(boot<=0) 近似
    print(f"\n95% CI of DiD = [{lo:+.3f}, {hi:+.3f}]")
    print(f"two-sided bootstrap p = {p_two:.3f}")
    print(f"one-sided p (H1: reversal>0) = {p_one:.3f}")
    print(f"significant @5% (CI excl 0)? {'YES' if (lo>0 or hi<0) else 'no'}")

    pd.DataFrame([{"gap_mode1": gap_m1, "gap_mode2": gap_m2, "DiD": obs,
                   "ci_lo": lo, "ci_hi": hi, "p_two": p_two, "p_one": p_one,
                   "n_days": n}]).to_csv(D / "did_reversal_test.csv",
                                         index=False, encoding="utf-8")
    print("\nsaved -> did_reversal_test.csv")


if __name__ == "__main__":
    main()
