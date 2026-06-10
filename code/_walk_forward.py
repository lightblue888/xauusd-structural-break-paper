# -*- coding: utf-8 -*-
"""
Walk-forward / 滚动窗口 OOS 稳定性 (#35, reviewer #4/#11)
========================================================================
把 MT5 真 tick 交易按 6 个月窗口切分, 算各窗口 Sharpe/Return。
V10 无任何调参 → 干净 OOS; V11 唯一调参是 sqrt sizing (= 负面发现 c, 样本内调参
反而使其更保守)。证明入场/半仓 edge 不依赖单一市场 regime。
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import INITIAL_BALANCE, OUT_DIR

ANN = 252
WINS = [('2024-01', '2024-07', '24H1 (disinflation)'),
        ('2024-07', '2025-01', '24H2'),
        ('2025-01', '2025-07', '25H1 (geopolitical)'),
        ('2025-07', '2026-01', '25H2'),
        ('2026-01', '2026-07', '26H1 (tariff / real-tick)')]


def load(f):
    df = pd.read_csv(OUT_DIR / f)
    df['ct'] = pd.to_datetime(df['final_close_time'], errors='coerce')
    df['pnl'] = pd.to_numeric(df['total_pnl_usd'], errors='coerce')
    return df.dropna(subset=['ct'])


def win_metrics(df, a, b):
    d = df[(df['ct'] >= a) & (df['ct'] < b)].sort_values('ct')
    if len(d) < 10:
        return None
    daily = d.groupby(d['ct'].dt.normalize())['pnl'].sum()
    eq = INITIAL_BALANCE + daily.cumsum()
    r = daily / eq.shift(1).fillna(INITIAL_BALANCE)
    sh = np.sqrt(ANN) * r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
    return dict(n=len(d), sharpe=round(sh, 2), ret=round(100 * daily.sum() / INITIAL_BALANCE, 1))


def main():
    rows = []
    for ver, f in [('V10', 'v10_realticks_trades.csv'), ('V11', 'v11_realticks_trades.csv')]:
        df = load(f)
        print(f"=== {ver} rolling windows (MT5 real-tick; V10 has zero tuned params = clean OOS) ===")
        for a, b, lbl in WINS:
            m = win_metrics(df, pd.Timestamp(a), pd.Timestamp(b))
            if m:
                print(f"  {lbl:28} n={m['n']:4}  Sharpe={m['sharpe']:5}  Return={m['ret']:6}%")
                rows.append({'ver': ver, 'window': lbl, **m})
    pd.DataFrame(rows).to_csv(OUT_DIR / 'walk_forward.csv', index=False, encoding='utf-8')
    print('saved -> walk_forward.csv')


if __name__ == '__main__':
    main()
