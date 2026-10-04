# Backtest results: eight strategies and the four-strategy blend

Written by `python scripts/make_results_summary.py` from a price cache of 6,663 sessions, 2000-01-03 to 2026-07-01. Every figure is computed by the code in this repository at each module's default parameters; none is typed in by hand. `summary.json` holds the same figures.

| Strategy | Status | Sample | CAGR | Volatility | Sharpe full / in sample / out of sample | Max drawdown | Turnover | Cost drag |
| --- | --- | --- | ---: | ---: | :---: | ---: | ---: | ---: |
| Dip buying (`mean_reversion`) | kept | 2000-11-10 to 2026-07-01 | 4.97% | 6.81% | 0.50 / 0.69 / 0.23 | -14.42% | 20.3x | 1.05% |
| Turn of month (`seasonality_flows`) | kept | 2000-01-03 to 2026-07-01 | 7.35% | 10.08% | 0.58 / 0.58 / 0.58 | -13.47% | 24.1x | 0.59% |
| Trend following (`tsmom_trend`) | kept | 2001-02-28 to 2026-07-01 | 5.58% | 6.25% | 0.64 / 0.77 / 0.42 | -14.67% | 2.0x | 0.18% |
| ETF momentum rotation (`xsec_etf_mom`) | kept | 2002-03-28 to 2026-07-01 | 13.74% | 20.86% | 0.65 / 0.74 / 0.49 | -27.85% | 5.7x | 0.26% |
| **Blend of the four kept strategies** (`ensemble`) | kept | 2000-01-03 to 2026-07-01 | 6.59% | 6.35% | 0.76 / 0.90 / 0.51 | -11.33% | 12.7x | 0.72% |
| Stock momentum (biased universe) (`xsec_stock_mom`) | rejected | 2001-01-31 to 2026-07-01 | 20.25% | 26.18% | 0.77 / 0.75 / 0.82 | -41.79% | 13.7x | 2.51% |
| VIX regime switch (`vol_regime`) | rejected | 2006-07-24 to 2026-07-01 | 11.95% | 15.79% | 0.70 / 0.79 / 0.62 | -40.95% | 12.0x | 0.41% |
| ETF pairs (`pairs_statarb`) | rejected | 2000-06-30 to 2026-07-01 | 2.37% | 3.90% | 0.18 / 0.48 / -0.54 | -9.98% | 18.4x | 1.26% |
| Trend at 1.64x leverage (`tsmom_voltarget`) | rejected | 2001-02-28 to 2026-07-01 | 7.72% | 10.29% | 0.62 / 0.76 / 0.37 | -23.47% | 3.7x | 0.25% |

How to read it:

- Returns are net of modelled commissions, fees and slippage and of 30% withholding on inferred dividends (Treasury funds exempt). Idle cash earns the 13-week bill yield less 10 bps. These are modelling assumptions.
- Sharpe ratios are in excess of that cash rate. In sample is before 2018-01-01; out of sample is from that date. The later period was inspected during development, so it is not an untouched holdout.
- Turnover is the yearly sum of absolute weight changes: buys plus sells. Cost drag is the yearly return lost to commissions, fees and slippage.
- A sample starts at the strategy's first position. The blend starts with the earliest strategy and holds Treasury bills for the others until they start.
- Rejected strategies are listed for the record. Their figures are not investable results; each module's docstring gives the reason:
  - `xsec_stock_mom`: Monthly stock momentum with a reversal tilt. Rejected: biased universe.
  - `vol_regime`: VIX term-structure regime switch, QQQ or IEF. Rejected: no timing value.
  - `pairs_statarb`: ETF pairs spread reversion. Rejected: negative out of sample after costs.
  - `tsmom_voltarget`: Volatility-target and concentration study on the trend rule. Rejected.

The blend:

- Weights, from the inverse of in-sample volatility and then fixed: `mean_reversion` 34.2%, `seasonality_flows` 19.8%, `tsmom_trend` 35.2%, `xsec_etf_mom` 10.9%.
- Each strategy trades its share of $100,000, where order minimums weigh more. That costs 0.18% a year against running each at the full amount, and is included above.
- The blend's turnover is the trading inside the strategies. Moving capital between them is not costed: the split is reset at every close. Full-sample Sharpe with the split reset every close / at month-ends / never: 0.76 / 0.75 / 0.66.

Charts: `equity_production.png` (the kept strategies and the blend), `drawdown_ensemble.png` (the blend's drawdown) and `equity_rejected.png` (the rejected strategies).
