"""Monthly sleeve returns from the raw Yahoo / FRED pulls."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# Public sleeves are fixed blends of ETFs. 'priv' is listed private equity (PSP): a market-priced
# proxy, so it carries the economic drawdown but is more volatile than reported fund NAVs.
EQUITY_MIX = {"SPY": 0.5, "EFA": 0.3, "EEM": 0.2}
SLEEVES = ["equity", "bonds", "real", "priv", "cash", "infl"]


def load_monthly():
    px = pd.read_csv(RAW / "prices_daily.csv", index_col=0, parse_dates=True)
    fr = pd.read_csv(RAW / "fred_monthly.csv", index_col=0, parse_dates=True)

    # month-end total-return prices; drop the current, incomplete month
    me = px.drop(columns=["^VIX"]).resample("ME").last()
    last_full = px.index.max() + pd.offsets.MonthEnd(0)
    if px.index.max() != px.index.max() + pd.offsets.MonthEnd(0):
        me = me.loc[: me.index[-2]]
    r = me.pct_change()

    fr.index = fr.index + pd.offsets.MonthEnd(0)
    cpi = fr["CPIAUCSL"].interpolate()          # Oct 2025 CPI was never published
    out = pd.DataFrame(index=r.index)
    out["equity"] = sum(w * r[k] for k, w in EQUITY_MIX.items())
    out["bonds"] = r["AGG"]
    out["real"] = r["VNQ"]
    out["priv"] = r["PSP"]
    out["cash"] = (fr["TB3MS"] / 100 / 12).reindex(r.index)
    out["infl"] = cpi.pct_change().reindex(r.index)
    out = out.loc["2006-11-30":].dropna()
    return out


if __name__ == "__main__":
    d = load_monthly()
    print(d.index[0].date(), "->", d.index[-1].date(), len(d), "months")
    print(d.describe().T[["mean", "std", "min", "max"]].round(4))
