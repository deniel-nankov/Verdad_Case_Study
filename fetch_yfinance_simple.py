"""
Simple yfinance data fetcher for FX Carry Analysis
Fetches additional data to supplement your existing analysis
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

print("="*70)
print("FETCHING SUPPLEMENTARY DATA FROM YAHOO FINANCE")
print("="*70)

start_date = '2000-01-01'
end_date = '2025-01-01'

# 1. VIX (Volatility Index)
print("\n1. Fetching VIX...")
try:
    vix_ticker = yf.Ticker('^VIX')
    vix = vix_ticker.history(start=start_date, end=end_date)
    vix_data = vix['Close']
    vix_data.to_csv('yf_vix.csv')
    print(f"   ✓ VIX: {len(vix_data)} observations saved to yf_vix.csv")
except Exception as e:
    print(f"   ✗ VIX failed: {e}")

# 2. Dollar Index (DXY)
print("\n2. Fetching Dollar Index...")
try:
    dxy_ticker = yf.Ticker('DX-Y.NYB')
    dxy = dxy_ticker.history(start=start_date, end=end_date)
    dxy_data = dxy['Close']
    dxy_data.to_csv('yf_dollar_index.csv')
    print(f"   ✓ DXY: {len(dxy_data)} observations saved to yf_dollar_index.csv")
except Exception as e:
    print(f"   ✗ DXY failed: {e}")

# 3. Commodity Index (DBC ETF)
print("\n3. Fetching Commodity Index...")
try:
    dbc_ticker = yf.Ticker('DBC')
    dbc = dbc_ticker.history(start=start_date, end=end_date)
    dbc_data = dbc['Close']
    dbc_data.to_csv('yf_commodity_index.csv')
    print(f"   ✓ DBC: {len(dbc_data)} observations saved to yf_commodity_index.csv")
except Exception as e:
    print(f"   ✗ DBC failed: {e}")

# 4. Treasury Bond ETFs
print("\n4. Fetching Treasury/Bond ETFs...")
bond_etfs = {
    'TLT': '20Y Treasury',
    'IEF': '7-10Y Treasury', 
    'LQD': 'Investment Grade Bonds',
    'HYG': 'High Yield Bonds'
}

for ticker, name in bond_etfs.items():
    try:
        etf = yf.Ticker(ticker)
        df = etf.history(start=start_date, end=end_date)
        df['Close'].to_csv(f'yf_{ticker.lower()}.csv')
        print(f"   ✓ {name} ({ticker}): {len(df)} observations saved to yf_{ticker.lower()}.csv")
    except Exception as e:
        print(f"   ✗ {name} ({ticker}) failed: {e}")

# 5. Additional Equity Indices
print("\n5. Fetching Equity Indices...")
equity_indices = {
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones',
    '^FTSE': 'FTSE 100',
    '^N225': 'Nikkei 225',
    'EEM': 'Emerging Markets'
}

for ticker, name in equity_indices.items():
    try:
        idx = yf.Ticker(ticker)
        df = idx.history(start=start_date, end=end_date)
        df['Close'].to_csv(f'yf_{ticker.replace("^", "").lower()}.csv')
        print(f"   ✓ {name} ({ticker}): {len(df)} observations saved")
    except Exception as e:
        print(f"   ✗ {name} ({ticker}) failed: {e}")

# 6. Gold and Oil
print("\n6. Fetching Commodities...")
commodities = {
    'GC=F': 'Gold Futures',
    'CL=F': 'Crude Oil (WTI)',
    'SI=F': 'Silver Futures'
}

for ticker, name in commodities.items():
    try:
        comm = yf.Ticker(ticker)
        df = comm.history(start=start_date, end=end_date)
        df['Close'].to_csv(f'yf_{name.lower().replace(" ", "_").replace("(","").replace(")","")}.csv')
        print(f"   ✓ {name} ({ticker}): {len(df)} observations saved")
    except Exception as e:
        print(f"   ✗ {name} ({ticker}) failed: {e}")

print("\n" + "="*70)
print("DONE! All data files saved.")
print("="*70)
print("\nFiles created:")
print("  - yf_vix.csv")
print("  - yf_dollar_index.csv")
print("  - yf_commodity_index.csv")
print("  - yf_tlt.csv, yf_ief.csv, yf_lqd.csv, yf_hyg.csv")
print("  - yf_gspc.csv, yf_dji.csv, yf_ftse.csv, yf_n225.csv, yf_eem.csv")
print("  - yf_gold_futures.csv, yf_crude_oil_wti.csv, yf_silver_futures.csv")
print("\nYou can now load these in your Phase 2 notebook!")
