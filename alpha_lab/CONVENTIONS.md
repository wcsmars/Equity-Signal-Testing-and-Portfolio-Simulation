# Timing and data conventions

These are the assumptions used by the implementation. The timing, bias, and
leakage tests check specific regressions; passing them does not establish that
an arbitrary dataset or custom signal is free of bias.

## Data

- Daily panels use ascending, unique, timezone-naive trading dates and ticker
  columns. All fields align to `close`.
- `close`, `open`, `high`, and `low` are expected to be split- and
  dividend-adjusted. Share commissions and dollar volume use raw
  `unadjusted_close` and raw-share `volume`. Missing raw prices trigger a
  warning and adjusted-price fallback in the cost model.
- A supplied boolean `universe` should describe historical membership known
  on each date. Without it, membership is inferred from price availability.
  Neither the CSV loader nor the synthetic demonstration establishes the
  historical completeness of a real investment universe.
- Price adjustment quality, historical constituents, delisting returns, and
  vendor revisions remain the data provider's responsibility.

## Timing

For `r_t = close_t / close_{t-1} - 1`:

1. Features `F_t` and target weights `W_t` may use information through close
   `t`. Features should be invariant when later rows are removed. Signals
   must score each date using its own available features, and fitting may
   use only the supplied training rows.
2. Holdings are `H_t = W_{t-lag}` and earn `gross_t = sum(H_t * r_t)`.
   **The default is `execution_lag=2`:** a target computed after Monday's
   close executes at Tuesday's close and first earns Wednesday's return.
   `lag=1` assumes the target can already be fixed before the same close
   used as its fill; it is unsuitable for features that require that final
   closing price. A positive lag alone does not guarantee executable timing.
   The schema and engine reject `lag < 1`.
3. The trade indexed by `t` executes at close `t-1`. Its size is
   `H_t - drift(H_{t-1})`, with drift computed using returns through `t-1`.
   Turnover is the sum of absolute traded weights. Drift adjustment is on
   by default and can be disabled.
4. Costs use raw prices and rolling liquidity/volatility estimates through
   `t-1`, and are charged to the return indexed by `t`:
   `net_t = gross_t - cost_t`. The spread, commission, and square-root impact
   parameters are assumptions, not calibrated execution estimates. The
   model uses a fixed reference portfolio value for participation and
   ignores financing, borrow availability, and borrow fees.

## Walk-forward and missing data

- Each chronological training window precedes its test window by the
  configured purge plus embargo gap. A fresh signal copy is fitted to each
  training window. Test scores are stitched before applying the holdings
  lag. Selecting parameters using these test results would invalidate an
  untouched out-of-sample interpretation.
- `walkforward: null` fits on the full sample and is a diagnostic mode.
- Unavailable scores or ineligible assets receive zero target weights;
  holdings still follow the configured lag. Prices are not forward-filled.
  Missing returns contribute zero, and lagged positions can persist after
  prices disappear. No delisting payout or forced liquidation is modeled;
  turnover costs still follow the configured cost model. Results involving
  missing prices require separate review.

## Paths and outputs

Paths in a YAML config (`data.path`, `experiment.runs_dir`, and
`experiment.feature_cache_dir`) are relative to that config's directory.
Dotted `--override` values use the same rule. The separate `--runs-dir` CLI
flag is relative to the shell's current directory. Absolute paths work in
all three cases. `config_from_dict` leaves paths as supplied.

The base example writes tracked results and HTML/Markdown reports to
`alpha_lab/runs/` relative to the public bundle. Synthetic results exercise
the software and are not evidence of investment performance.

For walk-forward runs, report charts, monthly returns, and headline metrics
start at the first test-window date. The training and purge warm-up remains
in the persisted result tables but is excluded from those summaries. The
window table still shows the training dates for context. Full-sample
diagnostic runs retain their complete date range.

## Numbered references in code comments

Some docstrings and comments cite an earlier numbering of the timing rules
("rule N" or "clause N"). They map to the Timing items above: 1-3 (features,
signal scoring and fitting, target weights) to item 1; 4-5 (holdings lag and
gross return) to item 2; 6 (trades, drift and turnover) to item 3; 7 (trade
execution and cost dating) to items 3 and 4; 8 (cost inputs through t-1) to
item 4.
