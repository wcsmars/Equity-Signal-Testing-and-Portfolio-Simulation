"""Illustrative transaction-cost assumptions for US stocks and ETFs.

Defaults model fixed per-share commissions with an order minimum and cap,
averaged buy/sell regulatory fees, and configurable slippage. The numerical
rates are retained research assumptions, not verified current broker prices.

Dividend withholding and Treasury-fund exemptions are simplified model
parameters. They do not implement fund/year-specific tax treatment and are
not legal or tax recommendations. Validate all inputs for the intended use.
"""

from dataclasses import dataclass


@dataclass
class IBKRHKCostModel:
    capital: float = 100_000.0          # USD account size assumption
    commission_per_share: float = 0.005
    min_commission: float = 1.00
    max_commission_pct: float = 0.01
    sec_fee_rate: float = 0.0000278     # sells only, on notional
    finra_taf_per_share: float = 0.000166
    slippage_bps: float = 3.0           # per side; override per universe

    def cost_bps_per_side(self, price: float, trade_notional: float) -> float:
        """All-in one-side cost in basis points of trade notional.

        trade_notional: dollar size of this single trade leg.
        price: share price, to convert per-share commission to bps.
        Use a price in the units of shares traded. The bundled downloader
        supplies split-adjusted closes, so historical per-share charges
        remain approximate even without dividend adjustment.
        Averages buy/sell regulatory fees (sells pay SEC+TAF, buys don't),
        which is exact over a round trip.
        """
        if trade_notional <= 0 or price <= 0:
            return 0.0
        shares = trade_notional / price
        commission = min(
            max(self.min_commission, self.commission_per_share * shares),
            self.max_commission_pct * trade_notional,
        )
        # halve sell-only fees to spread them across both sides
        regulatory = 0.5 * (
            self.sec_fee_rate * trade_notional
            + min(self.finra_taf_per_share * shares, 8.30)
        )
        return (commission + regulatory) / trade_notional * 1e4 + self.slippage_bps


# Fixed withholding assumption charged against inferred ex-date dividends;
# the helper below provides an approximate annual drag estimate.
US_DIV_WITHHOLDING = 0.30

# Fixed Treasury-ETF exemption list used by this model. Actual exemption
# fractions vary by fund/year and require distribution-level tax data.
QII_EXEMPT_TREASURY = frozenset({"SHY", "IEF", "TLT", "TIP", "SGOV", "BIL", "GOVT"})


def withholding_drag_annual(avg_dividend_yield: float, avg_exposure: float = 1.0) -> float:
    """Approximate annual return drag from 30% US dividend withholding."""
    return US_DIV_WITHHOLDING * avg_dividend_yield * avg_exposure
