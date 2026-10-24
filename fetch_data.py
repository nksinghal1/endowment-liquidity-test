"""Pull the data for the endowment liquidity stress test.

    set FRED_API_KEY=your_key        (Command Prompt)
    python fetch_data.py

Writes data/raw/prices_daily.csv (Yahoo, total-return adjusted closes) and
data/raw/fred_monthly.csv (T-bill rate, CPI, 10y yield).
"""
import os
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from fredapi import Fred

RAW = Path(__file__).resolve().parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
START = "1999-01-01"

# Yahoo proxies. Listed private equity (PSP) starts Oct 2006 and sets the common sample.
YAHOO = {
    "SPY": "US large-cap equity",
    "IWM": "US small-cap equity (PE/VC beta proxy)",
    "EFA": "Developed ex-US equity",
    "EEM": "Emerging market equity",
    "AGG": "US aggregate bonds",
    "IEF": "7-10y Treasuries",
    "VNQ": "US listed real estate (REITs)",
    "PSP": "Listed private equity",
    "QAI": "Multi-strategy hedge fund replication (starts 2009)",
    "^VIX": "VIX (diagnostic only)",
}
FRED = {
    "TB3MS": "3-month T-bill rate (%)",
    "CPIAUCSL": "CPI all urban consumers",
    "GS10": "10-year Treasury yield (%)",
}


def main():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        sys.exit("FRED_API_KEY is not set. In Command Prompt: set FRED_API_KEY=your_key")

    print("Pulling Yahoo prices...")
    px = yf.download(list(YAHOO), start=START, auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all").sort_index()
    px.to_csv(RAW / "prices_daily.csv")

    print("Pulling FRED...")
    fred = Fred(api_key=key)
    fr = pd.concat({k: fred.get_series(k, observation_start=START) for k in FRED}, axis=1)
    fr.to_csv(RAW / "fred_monthly.csv")

    print("\nYAHOO: first/last valid date, rows, share missing since 2006-10")
    for t, desc in YAHOO.items():
        s = px[t].dropna()
        since = px.loc["2006-10-01":, t]
        print(f"  {t:5s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):5d}  "
              f"missing since 2006-10: {since.isna().mean():.1%}   {desc}")
    print("\nFRED:")
    for t, desc in FRED.items():
        s = fr[t].dropna()
        print(f"  {t:9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):4d}   {desc}")
    print("\nSaved to", RAW)


if __name__ == "__main__":
    main()
