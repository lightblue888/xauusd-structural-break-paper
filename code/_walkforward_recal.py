# -*- coding: utf-8 -*-
"""P2-1: 递归 walk-forward — 只对乘数映射做 train/test。
训练 12 个月 → grid 重校准 m_ATR / m_Trend 值 → 冻结 → 测 6 个月 → 滚动 6 个月。
2024.01–2026.06 → 3 个测试窗:
  Window 1: train 2024.01–2024.12, test 2025.01–2025.06
  Window 2: train 2024.07–2025.06, test 2025.07–2025.12
  Window 3: train 2025.01–2025.12, test 2026.01–2026.06
每窗报 V11_recal vs V10 的 Sharpe。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v11_engine as eng
from _config import INITIAL_BALANCE
from _metrics import compute_metrics
from python_v10_v11_metrics import to_daily_equity

WINDOWS = [
    {'name': '2025H1', 'train': ('2024-01-01', '2024-12-31'), 'test': ('2025-01-01', '2025-06-30')},
    {'name': '2025H2', 'train': ('2024-07-01', '2025-06-30'), 'test': ('2025-07-01', '2025-12-31')},
    {'name': '2026H1', 'train': ('2025-01-01', '2025-12-31'), 'test': ('2026-01-01', '2026-06-01')},
]

# Grid: scale factors to try for m_ATR and m_Trend (single combined scale)
SCALE_GRID = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

def make_scaled_fn(atr_scale, trend_scale):
    def fn(atr_ratio, trend, use_dynamic=True):
        if not use_dynamic or pd.isna(atr_ratio): return 1.0
        if atr_ratio < 0.7:    atr_m = 1.5 * atr_scale
        elif atr_ratio < 0.9:  atr_m = 1.5 * atr_scale
        elif atr_ratio <= 1.1: atr_m = 0.5 * atr_scale
        elif atr_ratio <= 1.4: atr_m = 0.8 * atr_scale
        else:                  atr_m = 2.5 * atr_scale
        base = {'mixed': 3.0, 'strong_up': 0.85, 'strong_down': 0.85,
                'range': 0.65, 'weak_down': 0.30}.get(trend, 0.85)
        return float(np.clip(np.sqrt(atr_m * base * trend_scale),
                             eng.DYN_MULT_MIN, eng.DYN_MULT_MAX))
    return fn

def run_window(trade_start, trade_end, mult_fn=None):
    orig = eng.dynamic_multiplier
    if mult_fn is not None:
        eng.dynamic_multiplier = mult_fn
    try:
        trades = eng.run(use_dynamic_sizing=(mult_fn is not None),
                         log_progress=False, broker_tz="Europe/Athens",
                         trade_window_start=trade_start, trade_window_end=trade_end)
    finally:
        eng.dynamic_multiplier = orig
    if trades is None or len(trades) == 0:
        return None, 0
    eq, pnl = to_daily_equity(trades)
    m = compute_metrics(eq, trades_pnl=pnl,
                        initial_balance=INITIAL_BALANCE, label='wf')
    return m, len(trades)

results = []
for w in WINDOWS:
    print(f"\n=== {w['name']} ===")
    print(f"  Train: {w['train'][0]} → {w['train'][1]}")
    print(f"  Test:  {w['test'][0]} → {w['test'][1]}")

    # --- TRAIN: grid search best (atr_scale, trend_scale) on train window ---
    best_sharpe = -999
    best_params = (1.0, 1.0)
    for asc in SCALE_GRID:
        for tsc in SCALE_GRID:
            fn = make_scaled_fn(asc, tsc)
            m, n = run_window(w['train'][0], w['train'][1], fn)
            if m is not None and m['sharpe'] is not None and not np.isnan(m['sharpe']):
                if m['sharpe'] > best_sharpe:
                    best_sharpe = m['sharpe']
                    best_params = (asc, tsc)
    print(f"  Train best: atr_scale={best_params[0]}, trend_scale={best_params[1]}, Sharpe={best_sharpe:.3f}")

    # --- TEST: freeze best params, run on test window ---
    fn_best = make_scaled_fn(*best_params)
    m_recal, n_recal = run_window(w['test'][0], w['test'][1], fn_best)
    # V10 on same test window
    m_v10, n_v10 = run_window(w['test'][0], w['test'][1], None)

    sh_recal = m_recal['sharpe'] if m_recal else np.nan
    sh_v10 = m_v10['sharpe'] if m_v10 else np.nan
    ret_recal = m_recal['total_return_pct'] if m_recal else np.nan
    ret_v10 = m_v10['total_return_pct'] if m_v10 else np.nan

    print(f"  Test V11_recal: Sharpe {sh_recal}, Return {ret_recal}%, n={n_recal}")
    print(f"  Test V10:       Sharpe {sh_v10}, Return {ret_v10}%, n={n_v10}")
    print(f"  V11_recal beats V10? {sh_recal > sh_v10 if not np.isnan(sh_recal) and not np.isnan(sh_v10) else 'N/A'}")

    results.append({
        'window': w['name'],
        'train_best_atr_scale': best_params[0],
        'train_best_trend_scale': best_params[1],
        'train_sharpe': round(best_sharpe, 3),
        'test_v11_recal_sharpe': round(sh_recal, 3) if not np.isnan(sh_recal) else None,
        'test_v11_recal_return': round(ret_recal, 1) if not np.isnan(ret_recal) else None,
        'test_v11_recal_n': n_recal,
        'test_v10_sharpe': round(sh_v10, 3) if not np.isnan(sh_v10) else None,
        'test_v10_return': round(ret_v10, 1) if not np.isnan(ret_v10) else None,
        'test_v10_n': n_v10,
        'recal_beats_v10': bool(sh_recal > sh_v10) if not np.isnan(sh_recal) and not np.isnan(sh_v10) else None,
    })

df = pd.DataFrame(results)
out = Path('../data/walkforward_recal.csv')
df.to_csv(out, index=False, encoding='utf-8')
print(f"\nSaved -> {out}")
print(df.to_string(index=False))
print(f"\nOverall: V11_recal beats V10 in {sum(r.get('recal_beats_v10',False) for r in results)}/{len(results)} windows")
