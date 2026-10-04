# notebooks/

Exploratory work lives here; anything worth keeping graduates into the
library with tests. Keep heavy outputs out of git (checkpoints are ignored).

Start with `scripts/run_backtest.py --config configs/base.yaml`, then load the
run directory it prints:

```python
from alpha_lab.core.results import BacktestResult
res = BacktestResult.load("../runs/<run_id>/result")
res.equity_curve().loc[res.windows[0].test_start:].plot()  # from the first test date, as in the report
```
