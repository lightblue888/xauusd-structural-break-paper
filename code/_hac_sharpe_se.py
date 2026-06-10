# -*- coding: utf-8 -*-
"""
_hac_sharpe_se.py  —  Lo (2002) HAC-adjusted Sharpe SE + stationary block bootstrap CI.

Reviewer #24 reproducibility check for paper SS 4.5.

statsmodels is NOT installed in this environment, so Lo (2002) is implemented in pure
numpy (formula below) and the stationary block bootstrap (Politis & Romano 1994) is
implemented directly. No external dependency beyond numpy / pandas.

----------------------------------------------------------------------------------------
Lo, A. W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts Journal 58(4).

For a return series with autocorrelation, the iid asymptotic SE of the Sharpe ratio
SR is  sqrt((1 + 0.5 SR^2) / T).  Under serial correlation Lo multiplies the iid
variance by  eta^2 = sum_{k=-(q-1)}^{q-1} (1 - |k|/q) rho_k  (a Newey-West / Bartlett
weighted sum of autocorrelations), giving

    SE_HAC(SR) = sqrt(eta^2) * SE_iid(SR),   factor = sqrt(eta^2).

Here SR and SE are computed at the *sampling frequency* of the return series, then both
the point estimate and the SE are annualised by the same sqrt(periods_per_year) factor,
so the t-statistic t = SR_ann / SE_ann = SR_per / SE_per is scale-invariant.

The paper SS 4.5 measured the autocorrelation on the TRADE-LEVEL PnL series
(robustness_ljungbox.md: rho1=-0.108, rho3=+0.123, rho5=+0.113, Lo factor ~1.46),
so the Lo adjustment is applied to the per-trade Sharpe and then annualised by the
realised trades-per-year. We also report the daily-equity annualised Sharpe (the
headline 3.06 / 2.61) for cross-reference.
----------------------------------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("XAU_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
TRADING_DAYS = 252
RNG = np.random.default_rng(20260610)


# ----------------------------------------------------------------------------- helpers
def load_trades(tag):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{tag}_realticks_trades.csv"))
    df["open_dt"] = pd.to_datetime(df["open_time"], format="%Y.%m.%d %H:%M:%S")
    df["close_dt"] = pd.to_datetime(df["final_close_time"], format="%Y.%m.%d %H:%M:%S")
    return df


def trade_returns(df):
    """Per-trade simple return = total_pnl / balance_before_trade."""
    bal_before = df["running_balance"].values - df["total_pnl_usd"].values
    return df["total_pnl_usd"].values / bal_before


def daily_ann_sharpe(df, initial=10000.0):
    """Headline definition: daily equity returns, annualised by sqrt(252)."""
    eq = df.set_index("close_dt")["running_balance"].copy()
    eq = eq[~eq.index.duplicated(keep="last")]
    ts0 = eq.index.min() - pd.Timedelta(days=1)
    eq = pd.concat([pd.Series([initial], index=[ts0]), eq]).sort_index()
    d = eq.resample("1D").last().dropna()
    r = (d / d.shift(1) - 1).dropna()
    sr = r.mean() / r.std(ddof=1)
    return sr * np.sqrt(TRADING_DAYS), r


def autocorr(x, k):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if k >= n:
        return 0.0
    num = np.sum(x[:n - k] * x[k:])
    den = np.sum(x * x)
    return num / den


def lo_factor(returns, q):
    """sqrt(eta^2) with Bartlett (Newey-West) weights, lags 1..q-1."""
    eta2 = 1.0
    for k in range(1, q):
        w = 1.0 - k / q
        eta2 += 2.0 * w * autocorr(returns, k)
    eta2 = max(eta2, 1e-9)
    return np.sqrt(eta2), eta2


def sharpe_se(returns, q, periods_per_year):
    """Return (SR_ann, SE_iid_ann, SE_hac_ann, t, factor)."""
    r = np.asarray(returns, float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    se_iid = np.sqrt((1.0 + 0.5 * sr ** 2) / T)
    fac, _ = lo_factor(r, q)
    se_hac = fac * se_iid
    ann = np.sqrt(periods_per_year)
    sr_ann = sr * ann
    se_iid_ann = se_iid * ann
    se_hac_ann = se_hac * ann
    t = sr / se_hac  # scale-invariant
    return sr_ann, se_iid_ann, se_hac_ann, t, fac


def stationary_block_bootstrap_sharpe(returns, periods_per_year, n_boot=10000,
                                      mean_block=20):
    """Politis-Romano (1994) stationary bootstrap CI for the ANNUALISED Sharpe."""
    r = np.asarray(returns, float)
    T = len(r)
    p = 1.0 / mean_block
    ann = np.sqrt(periods_per_year)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(T, dtype=int)
        i = RNG.integers(0, T)
        for t in range(T):
            idx[t] = i
            if RNG.random() < p:
                i = RNG.integers(0, T)
            else:
                i = (i + 1) % T
        s = r[idx]
        sd = s.std(ddof=1)
        out[b] = (s.mean() / sd) * ann if sd > 0 else np.nan
    out = out[~np.isnan(out)]
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def nw_auto_lag(T):
    """Newey-West (1994) automatic lag q ~ floor(4*(T/100)^(2/9))."""
    return max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))


# ------------------------------------------------------------------------------- driver
def run(tag, q=None, n_boot=10000, mean_block=20):
    """SS4.5 object = the headline DAILY annualised Sharpe (3.06 / 2.61).

    Lo(2002) HAC SE and the stationary block bootstrap CI are therefore both
    computed on the DAILY equity-return series, matching _metrics.py. The
    per-trade PnL autocorrelations are reported as diagnostics only (these are
    the rho's quoted in robustness_ljungbox.md).
    """
    df = load_trades(tag)
    rd = daily_returns_series(df)          # daily equity returns (the headline object)
    rt = trade_returns(df)                 # per-trade returns (diagnostic only)
    pnl = df["total_pnl_usd"].to_numpy()   # raw per-trade PnL (Ljung-Box object)

    T = len(rd)
    if q is None:
        q = nw_auto_lag(T)
    sr_ann, se_iid, se_hac, t, fac = sharpe_se(rd, q, TRADING_DAYS)
    lo, hi = stationary_block_bootstrap_sharpe(rd, TRADING_DAYS, n_boot, mean_block)

    rho_pnl = [autocorr(pnl, k) for k in range(1, 6)]
    return dict(tag=tag, n_days=T, n_trades=len(rt), q=q,
                sr_daily_ann=sr_ann, se_iid=se_iid, se_hac=se_hac,
                t=t, factor=fac, ci=(lo, hi), rho_pnl=rho_pnl)


def daily_returns_series(df, initial=10000.0):
    eq = df.set_index("close_dt")["running_balance"].copy()
    eq = eq[~eq.index.duplicated(keep="last")]
    ts0 = eq.index.min() - pd.Timedelta(days=1)
    eq = pd.concat([pd.Series([initial], index=[ts0]), eq]).sort_index()
    d = eq.resample("1D").last().dropna()
    return (d / d.shift(1) - 1).dropna().to_numpy()


if __name__ == "__main__":
    sm = __import__("importlib").util.find_spec("statsmodels") is not None
    print("statsmodels available:", sm, "(pure-numpy fallback used)")
    print("SS4.5 object = DAILY annualised Sharpe; Lo(2002) HAC SE (Bartlett, "
          "NW auto lag); stationary block bootstrap mean_block=20, n_boot=10000, "
          "seed=20260610\n")
    hdr = (f"{'EA':<5}{'Tdays':>7}{'q':>4}{'SR_ann':>8}{'SE_iid':>9}"
           f"{'SE_HAC':>9}{'factor':>9}{'t':>7}   95% CI")
    print(hdr)
    print("-" * len(hdr))
    for tag in ("v10", "v11"):
        r = run(tag)
        print(f"{tag.upper():<5}{r['n_days']:>7}{r['q']:>4}{r['sr_daily_ann']:>8.2f}"
              f"{r['se_iid']:>9.3f}{r['se_hac']:>9.3f}{r['factor']:>9.3f}"
              f"{r['t']:>7.2f}   [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]")
        print(f"      per-trade PnL rho1..5 = "
              + ", ".join(f"{x:+.3f}" for x in r['rho_pnl'])
              + f"   (n_trades={r['n_trades']})")
