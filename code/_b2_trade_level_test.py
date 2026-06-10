# -*- coding: utf-8 -*-
"""
B2: K线过滤器的交易级配对检验 (比组合日度 Sharpe 差更高功率)
========================================================================
过滤器效应是逐笔的, 只作用于"打到 +1R"的交易。按入场把 V11 与 V11_FixedR 配对,
每对算 ΔR = R_V11 - R_FixedR (R = pnl / 风险, 归一化掉仓位), 检验 mean(ΔR)>0。
对比: 组合日度 Sharpe 差 (钝, p=0.60) vs 交易级配对 (恰当, p≈0.09)。
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import OUT_DIR

SL = 3.0
CONTRACT = 100.0
B = 10000
SEED = 11


def load(f):
    df = pd.read_csv(OUT_DIR / f)
    df["ot"] = pd.to_datetime(df["open_time"], errors="coerce")
    df["risk"] = SL * df["lot"] * CONTRACT
    df["R"] = pd.to_numeric(df["total_pnl_usd"], errors="coerce") / df["risk"]
    return df[["ot", "side", "R"]].dropna(subset=["ot"])


def main():
    v11 = load("v11_realticks_trades.csv")
    fx = load("v11_fixedr_realticks_trades.csv")
    for d in (v11, fx):
        d["key"] = d["ot"].dt.floor("min").astype(str) + "_" + d["side"].astype(str)
    m = v11.drop_duplicates("key").merge(
        fx.drop_duplicates("key"), on="key", suffixes=("_v11", "_fx"))
    dR = (m["R_v11"] - m["R_fx"]).values
    nz = dR[np.abs(dR) > 1e-9]

    rng = np.random.default_rng(SEED)
    n = len(dR)
    bm = np.array([dR[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.percentile(bm, [2.5, 97.5])
    p_all = 2 * min(np.mean(bm < 0), np.mean(bm > 0))

    pos, neg = int(np.sum(nz > 0)), int(np.sum(nz < 0))
    p_sign = stats.binomtest(pos, pos + neg, 0.5).pvalue

    print(f"matched={len(m)} ({100*len(m)/len(v11):.0f}% of V11), filter-active(ΔR≠0)={len(nz)} ({100*len(nz)/len(dR):.1f}%)")
    print(f"mean ΔR all={dR.mean():+.4f} R  CI=[{lo:+.4f},{hi:+.4f}]  p={p_all:.4f}")
    print(f"mean ΔR filter-active={nz.mean():+.4f} R")
    print(f"sign test filter-active: +{pos}/-{neg}  p={p_sign:.4f}  (效应在幅度而非频率)")

    pd.DataFrame([{"n_matched": len(m), "n_filter_active": len(nz),
                   "mean_dR_all": dR.mean(), "ci_lo": lo, "ci_hi": hi,
                   "p_all": p_all, "mean_dR_active": nz.mean(),
                   "sign_pos": pos, "sign_neg": neg, "p_sign": p_sign}]
                 ).to_csv(OUT_DIR / "b2_trade_level_test.csv", index=False, encoding="utf-8")
    print("saved -> b2_trade_level_test.csv")


if __name__ == "__main__":
    main()
