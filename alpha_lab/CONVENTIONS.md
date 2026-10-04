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
- Dates in CSV files and in config values (`data.start`, `data.end`,
  `data.synthetic.start`) must be ISO 8601: `YYYY-MM-DD`, optionally with a
  time of day and no UTC offset, or compact `YYYYMMDD` (quoted in YAML,
  where a bare `20240102` is a number). Any other spelling, such as
  `10/01/2024`, is rejected: the day/month order is never guessed.

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
   by default. `drift_adjust_turnover: false` is a comparison aid only:
   `gross_t` still assumes the book is reset to `H_t` every day, so the
   trades that undo each day's drift are then left out of turnover and
   cost, which are typically understated relative to the gross series.
4. Costs use raw prices and rolling liquidity/volatility estimates through
   `t-1`, and are charged to the return indexed by `t`:
   `net_t = gross_t - cost_t`. The spread, commission, and square-root impact
   parameters are assumptions, not calibrated execution estimates. The
  model uses a fixed reference portfolio value for participation and
  ignores financing, borrow availability, and borrow fees.
  Participation is capped at 2x average daily dollar volume, and a panel
  without `volume` is priced at a fixed 5% participation. Either way the
  modelled cost stops growing with the reference portfolio value (the
  model warns), so it is not a capacity estimate.
  Drift uses gross returns and costs are subtracted additively; fee-induced
  rescaling of holdings is not modeled. The engine rejects a loss of 100%
  or more, before or after costs, because insolvency is unsupported.

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
  prices disappear. A price gap inside a holding period makes two returns
  missing (the gap day and the day the price returns), so the move across
  the gap is dropped, not booked when the price reappears. No delisting
  payout or forced liquidation is modeled; turnover costs still follow the
  configured cost model. Results involving missing prices require separate
  review. A run warns and records `meta.missing_held_return_cells`
  whenever a held return is missing.

## Scores and portfolio construction

- Cross-sectional z-scores leave a date unscored when it has no dispersion:
  a standard deviation of at most 1e-12 times the date's largest absolute
  value, which is the rounding residue of equal values.
- A date whose valid scores are all equal gets zero target weights, in
  long-short and long-only books alike. Otherwise each bucket has
  `k = max(1, floor(n * quantile))` slots for the date's `n` valid scores.
- Ties never depend on column order. When the score at a bucket's edge is
  shared by names on both sides of the edge, all of them join the bucket.
  With `weighting: equal` they share the slots they straddle equally (`m`
  names over `j` slots get `j/m` of a slot each). With `weighting: score`
  they carry the bucket's least extreme score, which gets a near-zero
  weight (an equal split when the whole bucket is tied). A group tied
  across both edges is long and short at once and nets out, so that date
  can hold less than `gross_leverage`.
- `gross_leverage` is the gross of the book before vol targeting. With
  `vol_target` set, each date's book is multiplied by `target / estimate`
  clipped to [0, 3] and the per-name cap is applied again, so gross can
  fall below `gross_leverage` or reach three times it. The estimate uses
  each name's trailing volatility through the decision date and ignores
  correlations. A held name with fewer than `vol_lookback // 2` returns of
  its own is given the largest trailing volatility known that day. Dates
  before any name has an estimate keep the unscaled book.

## Paths and outputs

Paths in a YAML config (`data.path`, `experiment.runs_dir`, and
`experiment.feature_cache_dir`) are relative to that config's directory.
Dotted `--override` values use the same rule. The separate `--runs-dir` CLI
flag is relative to the shell's current directory. Absolute paths work in
all three cases. `config_from_dict` leaves paths as supplied.

A config file is one YAML mapping. An empty file means all defaults; any
other document (a list, a scalar) is an error. Unknown keys are errors, so
a YAML anchor must be declared on a value inside a schema section, not
under a scratch top-level key. Each alias gets its own copy of the anchored
value: an override applied through one path does not change the other.

The base example writes tracked results and HTML/Markdown reports to
`alpha_lab/runs/` relative to the project root. Synthetic results exercise
the software and are not evidence of investment performance.

Feature cache identities cover every input date and value. A disk entry is
also keyed by the feature's parameters and by its implementation: the
module-qualified name of the feature's class, the source text of that
class and of its base classes other than the `Feature` interface, and the
functions those classes define as loaded in the running process (their
bytecode, names and constants). Editing a feature class, or registering a
different class under the same name, is therefore a cache miss, also for a
class whose source cannot be read, such as one defined in an interactive
session. A session that imported a feature before its file was edited
writes under its own key, and entries are specific to the Python version
that compiled the code. Not covered: code the class only calls
(module-level helpers and constants, `MarketData` methods, pandas and
NumPy), and class attributes of a class without readable source. Clear the
cache after changing those. The in-memory memo of a `FeatureStore` is
keyed by parameters and data only. Cache files are written to a temporary
name and renamed into place; a file that cannot be read is reported with a
warning, recomputed and replaced.

Market-data snapshots are published only after all fields are written,
reject an existing snapshot id, and verify each recorded field checksum on
load. An empty panel cannot be snapshotted. A published snapshot directory
takes the permission bits of its dataset directory. Older snapshots without
manifests remain readable but cannot offer checksum verification. The
snapshot store is a standalone utility: no config key selects a snapshot,
and a run does not record a snapshot id.

For walk-forward runs, report charts, monthly returns, and headline metrics
start at the first test-window date. The training and purge warm-up remains
in the persisted result tables but is excluded from those summaries. The
window table still shows the training dates for context. Full-sample
diagnostic runs retain their complete date range: the report states how
many leading days precede the first position (signal warm-up and execution
lag), and those days count as flat days in every metric.

## Leakage checks

`alpha_lab.testing.checks` compares a computation on the full panel with
the same computation on the panel cut at a sampled date. Without explicit
`dates` the sample is eight dates spread from one sixth of the index to the
second-to-last date (cutting at the final date returns the whole panel and
cannot fail), plus the warm-up: the first two dates, the middle of the
first sixth, the first date on which the full-panel output holds a value,
and the first date on which it holds a non-zero value (target weights are
0.0, not missing, before the first position). The warm-up rows are where a
backfill would place future values.
The feature, signal and constructor checks raise `DataError` when the
full-panel output is blank (missing or 0.0) on every checked date, as on a
panel shorter than the lookback: no value would have been compared.
The cost check samples dates that carry a trade and raises `DataError`
when none of the checked dates does, because a date without a trade costs
nothing whatever the model reads. These are sampled probes, not proofs.

## Numbered references in code and error messages

Some docstrings, comments and error messages cite an earlier numbering of the
timing rules ("rule N" or "clause N"). They map to the Timing items above:
1-3 (features, signal scoring and fitting, target weights) to item 1; 4-5
(holdings lag and gross return) to item 2; 6 (trades, drift and turnover) to
item 3; 7 (trade execution and cost dating) to items 3 and 4; 8 (inputs known
only through t-1: the drift behind each trade and the cost inputs) to items 3
and 4.

## Forbidden patterns

Library code must avoid:

- `shift(-k)` / negative shifts anywhere in features, signals, portfolio,
  costs (test code may use them to *plant* leaks for the harness to catch,
  or to measure forward returns after the fact).
- Full-sample statistics inside features/signals (e.g. z-scoring a time
  series by its full-sample mean/std). Cross-sectional per-date
  standardization is fine; time-series standardization must be rolling or
  expanding.
- Fitting on test dates, peeking across the purge gap, or reusing test data
  to select hyperparameters presented as out-of-sample.
- Trading on same-day information the trade itself could not have known
  (Timing items 2-4 above).
