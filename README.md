# Structural-Break Entry and Asymmetric Partial-Exit on XAUUSD

Replication code and trade-level data for the paper:

> **Structural-Break Entry and Asymmetric Partial-Exit Strategy for XAUUSD: A Tick-Level Empirical Reassessment of Volatility-Adaptive Sizing and Backtest Bias**

This repository contains the **Python analysis pipeline** and **trade-level output data** for all backtested specifications and baselines. The proprietary MQL5 strategy source is **not** included (the strategy is under active commercial deployment); the complete strategy logic is specified as language-agnostic pseudocode in the paper's Appendix A, sufficient for independent reimplementation.

## What's here

```
code/   Python pipeline: MT5 tester-log parser, unified metrics,
        6 baselines, Monte Carlo placebo, sizing-variant analysis, plots
data/   Trade-level CSVs for V10/V11/V10_2x/V11_2x, sizing variants,
        half-exit ablation, and Baselines 1-6 (both backtest modes)
.env.example   Environment-variable template (see Configuration)
```

## Four contributions (reproducible from this repo)

1. **Entry-signal alpha** — structural-break entry beats six baselines (Sharpe 2.6–3.1).
2. **Asymmetric partial-exit (negative finding)** — a +1R half-exit with breakeven stop reshapes the outcome distribution (win rate 17%→38%, drawdown cut ~1/3) but institutionalizes the disposition effect; a faithful real-tick ablation shows no gain in any risk-adjusted measure (Sharpe/Sortino/Calmar all statistically unchanged).
3. **Negative finding** — regime-conditional position sizing (sqrt-product, ATR-only, trend-only, Moreira–Muir volatility targeting) generates no Sharpe alpha; apparent gains are covert leverage.
4. **Methodological finding** — backtest tick-resolution can reverse the directional conclusion about which sub-component generates alpha.

## Requirements

- Python 3.10+
- `pandas`, `numpy`, `matplotlib`, `scipy`

```bash
pip install pandas numpy matplotlib scipy
```

## Configuration

Most analysis runs directly on the released CSVs under `data/`. To re-parse raw MT5 tester logs or regenerate baselines from raw data, set the environment variables in `.env.example`:

```bash
cp .env.example .env   # then edit paths
# key variables:
#   XAU_DATA_DIR     raw-data directory (V10/V11/1m CSVs)
#   MT5_HASH         MT5 terminal directory hash (for log parsing)
#   MT5_TESTER_DIR   or the full agent-log directory
```

## Reproducing key results

```bash
# Unified metrics for all MT5 real-tick specifications
python -X utf8 code/mt5_v10_v11_metrics.py

# Baseline comparison table + figures
python -X utf8 code/summary_table.py
python -X utf8 code/summary_plots.py

# Parse a raw MT5 tester log into trade-level CSV (requires MT5_* env vars)
python -X utf8 code/_parse_tester_log.py --session -1 --out v11_trades.csv
```

## Performance metrics

All metrics follow standard academic conventions: Sharpe and Sortino are daily,
annualized by √252; Calmar is annualized return over maximum drawdown.

## Data notes

- Trade-level CSVs are derived from MT5 Strategy Tester runs (ECMarkets broker)
  over 2024-01-01 to 2026-06-01. Two modeling modes are provided:
  interpolated-tick ("Mode 1") and real-tick ("Mode 2").
- The custom parser reconciles against MT5-reported final balances with
  discrepancies below 3.3% across all specifications.

## License

MIT (code). Trade-level data released for academic reproduction.

## Citation

```bibtex
@article{quan2026structuralbreak,
  title  = {Structural-Break Entry and Asymmetric Partial-Exit Strategy for XAUUSD:
            A Tick-Level Empirical Reassessment of Volatility-Adaptive Sizing and Backtest Bias},
  author = {Quan, Yongjun and Jiang, Yonghong},
  year   = {2026},
  note   = {Working paper}
}
```
