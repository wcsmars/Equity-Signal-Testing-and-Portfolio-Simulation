# Systematic Trading Research

Python tools for daily equity and ETF research: signal construction,
transaction-cost modelling, walk-forward evaluation, data-quality checks,
and reproducible experiment reports.

**Author:** Chung Shing Mars Wong

The repository contains two implementations:

- **Alpha Lab** is the configurable research library. Its default example
  uses generated prices, requires no data download, and records configuration,
  returns, metrics, and reports for each run.
- **QCore** contains the earlier ETF research engine and eight strategy
  examples. These use a locally downloaded market-data cache and retain the
  historical execution assumptions described below.

## Example output

![Synthetic momentum demonstration: equity before and after transaction costs, with net drawdown below](examples/equity.png)

Generated prices, fixed seed, and the [base configuration](alpha_lab/configs/base.yaml).
Only the walk-forward evaluation period is plotted; the training warm-up is
excluded. These synthetic results demonstrate the pipeline, not a profitable
trading strategy. Browse the [sample report](examples/synthetic_report.md) on
GitHub, or download the [self-contained HTML report](examples/synthetic_report.html)
and open it in a browser.

## Implementation and design choices

The project covers the research workflow from data validation and signals to
portfolio construction, evaluation, and saved experiment reports. Alpha Lab
separates these components so signals and cost assumptions can be changed
without rewriting the backtest.

Three decisions are central to the implementation:

- **Make execution timing explicit.** The default delays close-based signals
  until the next close, and costs use prices and liquidity estimates available
  at the modeled execution time.
- **Test for specific research errors.** Regression tests plant future-data
  leaks, same-day cost inputs, price-adjustment defects, and universe-membership
  violations, alongside clean controls that must pass.
- **Keep runs inspectable.** Each experiment saves its resolved configuration,
  environment versions, result tables, metrics, and reports. Gross and net
  results expose the effect of the assumed transaction costs.

## Run the offline example

Use Python 3.11 or 3.12. From this repository's root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
python alpha_lab/scripts/run_backtest.py --config alpha_lab/configs/base.yaml
```

The example generates 20 synthetic assets with a fixed seed, computes
12-1 momentum, constructs a constrained long-short portfolio, and evaluates
rolling training/test windows with transaction costs. The default execution
lag is two daily rows: signals formed at one close execute at the following
close, before earning the next close-to-close return.

Runs are saved under `alpha_lab/runs/`, including `config.yaml`,
`metrics.json`, persisted result tables, and HTML/Markdown reports. The command
prints the run directory. Rebuild an existing report with:

```bash
python alpha_lab/scripts/make_report.py --run alpha_lab/runs/<run_id>
```

The [sample report](examples/synthetic_report.md) includes metrics,
equity/drawdown charts, turnover, and evaluation windows produced by this
configuration. See [the timing conventions](alpha_lab/CONVENTIONS.md)
for the detailed data and execution contract.

## Explore market-data strategies

The optional downloader retrieves daily prices through `yfinance`. It requires
network access and may be affected by provider revisions or rate limits. Obtain
data under the provider's applicable terms; no downloaded market dataset is
distributed here.

```bash
python src/download_data.py
python scripts/data_quality.py
python src/strategies/mean_reversion.py
```

Run the data-quality command separately and inspect its findings before
continuing: exit code `0` means no warning/failure, `1` means warnings, and `2`
means failures. It writes `results/data_quality.json`. Strategy scripts print
computed metrics. `pairs_statarb.py` also saves `results/pairs_statarb.json`,
`tsmom_voltarget.py` runs its variant study and saves
`results/tsmom_voltarget*.{csv,json}`, and the `--sweep` options write
variant tables in `results/`.

| Script in `src/strategies/` | Research example |
| --- | --- |
| `mean_reversion.py` | Short-horizon RSI reversal with a trend filter |
| `tsmom_trend.py` | Monthly multi-asset trend and inverse-volatility weights |
| `tsmom_voltarget.py` | Trend concentration and volatility-scaling variants |
| `xsec_etf_mom.py` | ETF momentum rotation with rank hysteresis |
| `xsec_stock_mom.py` | Stock momentum with a reversal component |
| `seasonality_flows.py` | Turn-of-month positioning |
| `vol_regime.py` | VIX term-structure regime rules |
| `pairs_statarb.py` | Rolling hedge ratios and spread reversion |

## Research limits

- QCore uses close-of-day signals with execution at that same close and
  next-day returns. Lagging returns does not make final closing prices available
  before an order deadline. Treat these results as an execution approximation;
  Alpha Lab's two-row default provides a delayed-close example.
- The fixed stock universe is survivorship-biased. Provider price histories do
  not supply historical index membership or a complete delisting dataset.
- QCore labels results before and after 2018 as `in_sample` and `out_of_sample`.
  The research process included later changes informed by inspected results;
  the later period is not an untouched holdout for the final strategy choices.
- Fee, slippage, financing, cash-yield, and withholding parameters are modelling
  assumptions. They are not a verified current broker schedule or a complete
  account-specific cost model. QCore does not charge margin interest centrally;
  the volatility-scaling example adds its own financing approximation.
- Share counts for commissions come from a split-adjusted close, so stocks
  with large later splits pay the 1%-of-notional commission cap in early years
  (costs overstated), and reverse splits understate them. Dividends are
  inferred from adjusted versus unadjusted prices, so spin-off adjustments can
  be charged withholding as if they were dividends.
- Missing-price treatment, synthetic universe membership, and simplified
  delisting behaviour limit what the backtests establish. Passing tests checks
  implemented behaviours and known failure cases, not profitability or the
  absence of every research bias.

## Code map

```text
alpha_lab/alpha_lab/   data, features, signals, portfolios, backtests, reports
alpha_lab/configs/     reproducible example configuration
alpha_lab/scripts/     experiment and report commands
alpha_lab/tests/       unit, timing, and bias regression tests
src/qcore/            original backtest, costs, calendar, statistics, quality checks
src/strategies/       market-data strategy examples
src/download_data.py  optional market-data downloader
scripts/data_quality.py
tests/               QCore engine, downloader and data-quality tests (synthetic data)
examples/            synthetic report and charts, viewable directly on GitHub
```

Local validation: 472 tests passed on Python 3.11.7 in isolated environments
using current compatible dependencies and the declared minimum versions,
without SciPy. The offline example, report rebuild, and package wheel build
and import completed from standalone copies in folders containing spaces.
The GitHub workflow is configured to run the tests and offline example on
Python 3.11 and 3.12. A small live downloader smoke test covered SPY, QQQ,
SGOV and two index series; it does not validate every instrument or historical bar.

Released under the [MIT License](LICENSE).
