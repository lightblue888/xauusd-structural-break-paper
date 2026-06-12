# -*- coding: utf-8 -*-
"""P0-2 v2: 乘数映射扰动敏感性 (统一日收益年化口径)。
族 A: 全表值 (m_ATR × m_Trend) ×{0.8, 1.0, 1.2} (对称, 一族即可)
族 B: rho 分桶边界 ×{0.8, 1.0, 1.2}
族 C: ATRh=2.5 单独 ±20% (助教真正点名的两个校准值之一)
族 D: mix=3.0 单独 ±20%
对照 = V10 (use_dynamic=False)。引擎 broker_tz=Europe/Athens。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v11_engine as eng
from _config import INITIAL_BALANCE, OUT_DIR
from _metrics import compute_metrics
from python_v10_v11_metrics import to_daily_equity, compute_rr

def run_variant(label, patch_fn):
    orig = eng.dynamic_multiplier
    eng.dynamic_multiplier = patch_fn
    try:
        trades = eng.run(use_dynamic_sizing=True, log_progress=False, broker_tz="Europe/Athens")
    finally:
        eng.dynamic_multiplier = orig
    if trades is None or len(trades) == 0:
        return {'label': label, 'n': 0, 'sharpe': np.nan, 'ret': np.nan, 'calmar': np.nan, 'maxdd': np.nan}
    eq, pnl = to_daily_equity(trades)
    m = compute_metrics(eq, trades_pnl=pnl, initial_balance=INITIAL_BALANCE, label=label)
    return {'label': label, 'n': len(trades), 'sharpe': m['sharpe'],
            'ret': m['total_return_pct'], 'calmar': m['calmar'], 'maxdd': m['max_dd_pct']}

def make_map_fn(scale):
    """Scale ALL map values by `scale` (m_ATR × m_Trend symmetric)."""
    def fn(atr_ratio, trend, use_dynamic=True):
        if not use_dynamic or pd.isna(atr_ratio): return 1.0
        if atr_ratio < 0.7:    atr_m = 1.5
        elif atr_ratio < 0.9:  atr_m = 1.5
        elif atr_ratio <= 1.1: atr_m = 0.5
        elif atr_ratio <= 1.4: atr_m = 0.8
        else:                  atr_m = 2.5
        trend_m = {'mixed': 3.0, 'strong_up': 0.85, 'strong_down': 0.85,
                   'range': 0.65, 'weak_down': 0.30}.get(trend, 0.85)
        return float(np.clip(np.sqrt(atr_m * trend_m * scale), eng.DYN_MULT_MIN, eng.DYN_MULT_MAX))
    return fn

def make_bound_fn(scale):
    b = [0.9 * scale, 1.1 * scale, 1.4 * scale]
    def fn(atr_ratio, trend, use_dynamic=True):
        if not use_dynamic or pd.isna(atr_ratio): return 1.0
        if atr_ratio < 0.7:    atr_m = 1.5
        elif atr_ratio < b[0]: atr_m = 1.5
        elif atr_ratio <= b[1]:atr_m = 0.5
        elif atr_ratio <= b[2]:atr_m = 0.8
        else:                  atr_m = 2.5
        trend_m = {'mixed': 3.0, 'strong_up': 0.85, 'strong_down': 0.85,
                   'range': 0.65, 'weak_down': 0.30}.get(trend, 0.85)
        return float(np.clip(np.sqrt(atr_m * trend_m), eng.DYN_MULT_MIN, eng.DYN_MULT_MAX))
    return fn

def make_atrh_fn(atrh_val):
    """Perturb only ATRh (the >1.4 bucket value), keep rest at deployed."""
    def fn(atr_ratio, trend, use_dynamic=True):
        if not use_dynamic or pd.isna(atr_ratio): return 1.0
        if atr_ratio < 0.7:    atr_m = 1.5
        elif atr_ratio < 0.9:  atr_m = 1.5
        elif atr_ratio <= 1.1: atr_m = 0.5
        elif atr_ratio <= 1.4: atr_m = 0.8
        else:                  atr_m = atrh_val  # perturbed
        trend_m = {'mixed': 3.0, 'strong_up': 0.85, 'strong_down': 0.85,
                   'range': 0.65, 'weak_down': 0.30}.get(trend, 0.85)
        return float(np.clip(np.sqrt(atr_m * trend_m), eng.DYN_MULT_MIN, eng.DYN_MULT_MAX))
    return fn

def make_mix_fn(mix_val):
    """Perturb only m_Trend(mixed), keep rest at deployed."""
    def fn(atr_ratio, trend, use_dynamic=True):
        if not use_dynamic or pd.isna(atr_ratio): return 1.0
        if atr_ratio < 0.7:    atr_m = 1.5
        elif atr_ratio < 0.9:  atr_m = 1.5
        elif atr_ratio <= 1.1: atr_m = 0.5
        elif atr_ratio <= 1.4: atr_m = 0.8
        else:                  atr_m = 2.5
        tmap = {'mixed': mix_val, 'strong_up': 0.85, 'strong_down': 0.85,
                'range': 0.65, 'weak_down': 0.30}
        trend_m = tmap.get(trend, 0.85)
        return float(np.clip(np.sqrt(atr_m * trend_m), eng.DYN_MULT_MIN, eng.DYN_MULT_MAX))
    return fn

results = []

# V10 baseline
print('Running V10 baseline...')
trades_v10 = eng.run(use_dynamic_sizing=False, log_progress=False, broker_tz="Europe/Athens")
eq_v10, pnl_v10 = to_daily_equity(trades_v10)
m0 = compute_metrics(eq_v10, trades_pnl=pnl_v10, initial_balance=INITIAL_BALANCE, label='V10')
results.append({'label': 'V10 baseline', 'family': '---', 'perturbation': '---',
                'n': len(trades_v10), 'sharpe': m0['sharpe'], 'ret': m0['total_return_pct'],
                'calmar': m0['calmar'], 'maxdd': m0['max_dd_pct']})
print(f"  V10: Sharpe {m0['sharpe']}")

# V11 baseline (unperturbed)
print('Running V11 baseline...')
r = run_variant('V11 baseline', eng.dynamic_multiplier)
r['family'] = '---'; r['perturbation'] = 'deployed'
results.append(r)
print(f"  V11: Sharpe {r['sharpe']}")

TESTS = [
    ('Map values', [0.8, 1.2], make_map_fn),
    ('Rho breakpoints', [0.8, 1.2], make_bound_fn),
    ('ATRh (>1.4 bucket)', [2.5*0.8, 2.5*1.2], make_atrh_fn),
    ('mix (mixed trend)', [3.0*0.8, 3.0*1.2], make_mix_fn),
]

for fam, vals, maker in TESTS:
    for v in vals:
        label = f'{fam} = {v:.1f}'
        print(f'Running {label}...')
        r = run_variant(label, maker(v))
        r['family'] = fam
        r['perturbation'] = f'{v:.1f}'
        results.append(r)
        print(f"  {label}: Sharpe {r['sharpe']}")

df = pd.DataFrame(results)
out = OUT_DIR / 'multiplier_sensitivity.csv'
df.to_csv(out, index=False, encoding='utf-8')
print(f'\nSaved -> {out}')
print(df[['label','family','perturbation','n','sharpe','ret','maxdd']].to_string(index=False))
print(f"\nV10 baseline Sharpe: {m0['sharpe']}")
print('Any variant beats V10?', (df[df.label != 'V10 baseline']['sharpe'] > m0['sharpe']).any())
