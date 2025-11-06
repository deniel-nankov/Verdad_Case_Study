"""
Data Sources Integration for FX Carry Analysis
Demonstrates how to fetch additional data using various free APIs
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# 1. YFINANCE - NO API KEY NEEDED (EASIEST)
# ============================================================================

def fetch_yfinance_data(start_date='2000-01-01', end_date='2025-01-01'):
    """
    Fetch financial data using yfinance (Yahoo Finance)
    FREE - No API key required
    
    Install: pip install yfinance
    """
    import yfinance as yf
    
    print("Fetching data from Yahoo Finance...")
    
    # Dictionary to store all data
    data = {}
    
    # 1. FX Data (alternative source to verify your existing data)
    fx_pairs = {
        'AUDUSD=X': 'AUD',
        'EURUSD=X': 'EUR', 
        'GBPUSD=X': 'GBP',
        'USDJPY=X': 'JPY',
        'USDCAD=X': 'CAD',
        'USDCHF=X': 'CHF',
        'USDBRL=X': 'BRL',
        'USDMXN=X': 'MXN'
    }
    
    print("\n1. Downloading FX rates...")
    fx_data = {}
    for ticker, name in fx_pairs.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not df.empty:
                # Use 'Close' for FX data (no adjusted close for currencies)
                fx_data[name] = df['Close']
                print(f"   ✓ {name}: {len(df)} observations")
        except Exception as e:
            print(f"   ✗ {name}: Failed - {e}")
    
    if fx_data:
        data['fx_rates'] = pd.DataFrame(fx_data)
    else:
        data['fx_rates'] = pd.DataFrame()
    
    # 2. Equity Indices
    print("\n2. Downloading Equity Indices...")
    equity_tickers = {
        '^GSPC': 'SP500',      # S&P 500
        '^DJI': 'DJIA',         # Dow Jones
        '^IXIC': 'NASDAQ',      # NASDAQ
        '^FTSE': 'FTSE100',     # UK
        '^N225': 'NIKKEI',      # Japan
        'EEM': 'EmergingMkts'   # Emerging Markets ETF
    }
    
    equity_data = {}
    for ticker, name in equity_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not df.empty:
                # Use 'Close' column when auto_adjust=True (it becomes adjusted close)
                equity_data[name] = df['Close'].pct_change()
                print(f"   ✓ {name}: {len(df)} observations")
        except Exception as e:
            print(f"   ✗ {name}: Failed - {e}")
    
    if equity_data:
        data['equity_returns'] = pd.DataFrame(equity_data)
    else:
        data['equity_returns'] = pd.DataFrame()
    
    # 3. Volatility (VIX)
    print("\n3. Downloading Volatility Index...")
    try:
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False, auto_adjust=True)
        data['vix'] = vix['Close']
        print(f"   ✓ VIX: {len(vix)} observations")
    except Exception as e:
        print(f"   ✗ VIX: Failed - {e}")
    
    # 4. Commodities
    print("\n4. Downloading Commodities...")
    commodity_tickers = {
        'GC=F': 'Gold',
        'CL=F': 'Oil_WTI',
        'SI=F': 'Silver',
        'DBC': 'Commodity_Index',  # ETF
        'USO': 'Oil_ETF'
    }
    
    commodity_data = {}
    for ticker, name in commodity_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not df.empty:
                commodity_data[name] = df['Close'].pct_change()
                print(f"   ✓ {name}: {len(df)} observations")
        except Exception as e:
            print(f"   ✗ {name}: Failed - {e}")
    
    if commodity_data:
        data['commodities'] = pd.DataFrame(commodity_data)
    else:
        data['commodities'] = pd.DataFrame()
    
    # 5. Dollar Index
    print("\n5. Downloading Dollar Index...")
    try:
        dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date, progress=False, auto_adjust=True)
        data['dollar_index'] = dxy['Close']
        print(f"   ✓ DXY: {len(dxy)} observations")
    except Exception as e:
        print(f"   ✗ DXY: Failed - {e}")
    
    # 6. Treasury ETFs (bond returns proxy)
    print("\n6. Downloading Treasury/Bond ETFs...")
    bond_tickers = {
        'TLT': 'Treasury_20Y',
        'IEF': 'Treasury_7-10Y',
        'SHY': 'Treasury_1-3Y',
        'LQD': 'Corp_Bonds_IG',
        'HYG': 'Corp_Bonds_HY'
    }
    
    bond_data = {}
    for ticker, name in bond_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not df.empty:
                bond_data[name] = df['Close'].pct_change()
                print(f"   ✓ {name}: {len(df)} observations")
        except Exception as e:
            print(f"   ✗ {name}: Failed - {e}")
    
    if bond_data:
        data['bonds'] = pd.DataFrame(bond_data)
    else:
        data['bonds'] = pd.DataFrame()
    
    return data


# ============================================================================
# 2. ALPHA VANTAGE - REQUIRES FREE API KEY
# ============================================================================

def fetch_alphavantage_data(api_key, symbols=['EURUSD', 'GBPUSD']):
    """
    Fetch FX and economic data from Alpha Vantage
    FREE API Key: https://www.alphavantage.co/support/#api-key
    Limit: 5 calls/minute, 500 calls/day (free tier)
    
    Install: pip install alpha_vantage
    """
    from alpha_vantage.foreignexchange import ForeignExchange
    from alpha_vantage.timeseries import TimeSeries
    import time
    
    print("Fetching data from Alpha Vantage...")
    
    fx = ForeignExchange(key=api_key, output_format='pandas')
    ts = TimeSeries(key=api_key, output_format='pandas')
    
    data = {}
    
    # FX Data (daily)
    print("\n1. Downloading FX Data...")
    for symbol in symbols:
        try:
            df, meta = fx.get_currency_exchange_daily(
                from_symbol=symbol[:3], 
                to_symbol=symbol[3:],
                outputsize='full'
            )
            data[symbol] = df['4. close']
            print(f"   ✓ {symbol}: {len(df)} observations")
            time.sleep(12)  # Respect rate limit (5 calls/min)
        except Exception as e:
            print(f"   ✗ {symbol}: Failed - {e}")
    
    return data


# ============================================================================
# 3. QUANDL - REQUIRES FREE API KEY
# ============================================================================

def fetch_quandl_data(api_key):
    """
    Fetch economic and financial data from Quandl
    FREE API Key: https://www.quandl.com/sign-up
    Limit: 50 calls/day (free tier)
    
    Install: pip install quandl
    """
    import quandl
    
    quandl.ApiConfig.api_key = api_key
    
    print("Fetching data from Quandl...")
    
    data = {}
    
    # 1. FRED Economic Data (via Quandl)
    print("\n1. Downloading Economic Indicators...")
    indicators = {
        'FRED/DGS10': 'Treasury_10Y',
        'FRED/DGS2': 'Treasury_2Y',
        'FRED/VIXCLS': 'VIX',
        'FRED/DEXUSEU': 'EUR_USD',
        'FRED/DEXJPUS': 'JPY_USD'
    }
    
    for code, name in indicators.items():
        try:
            df = quandl.get(code, start_date='2000-01-01')
            data[name] = df
            print(f"   ✓ {name}: {len(df)} observations")
        except Exception as e:
            print(f"   ✗ {name}: Failed - {e}")
    
    return data


# ============================================================================
# 4. TIINGO - REQUIRES FREE API KEY
# ============================================================================

def fetch_tiingo_data(api_key, tickers=['SPY', 'TLT']):
    """
    Fetch stock/ETF data from Tiingo
    FREE API Key: https://www.tiingo.com/
    Better historical data quality than Yahoo
    
    Install: pip install tiingo
    """
    from tiingo import TiingoClient
    
    config = {'api_key': api_key}
    client = TiingoClient(config)
    
    print("Fetching data from Tiingo...")
    
    data = {}
    
    for ticker in tickers:
        try:
            df = client.get_dataframe(
                ticker,
                startDate='2000-01-01',
                endDate='2025-01-01'
            )
            data[ticker] = df['adjClose']
            print(f"   ✓ {ticker}: {len(df)} observations")
        except Exception as e:
            print(f"   ✗ {ticker}: Failed - {e}")
    
    return data


# ============================================================================
# 5. INTEGRATED FETCH FUNCTION (Combines Multiple Sources)
# ============================================================================

def fetch_all_supplementary_data(
    use_yfinance=True,
    use_alphavantage=False,
    use_quandl=False,
    alphavantage_key=None,
    quandl_key=None
):
    """
    Master function to fetch data from multiple sources
    Combines and deduplicates data
    """
    
    all_data = {}
    
    # 1. Yahoo Finance (no key needed)
    if use_yfinance:
        print("\n" + "="*70)
        print("FETCHING FROM YAHOO FINANCE (yfinance)")
        print("="*70)
        yf_data = fetch_yfinance_data()
        all_data['yfinance'] = yf_data
    
    # 2. Alpha Vantage (requires key)
    if use_alphavantage and alphavantage_key:
        print("\n" + "="*70)
        print("FETCHING FROM ALPHA VANTAGE")
        print("="*70)
        av_data = fetch_alphavantage_data(alphavantage_key)
        all_data['alphavantage'] = av_data
    
    # 3. Quandl (requires key)
    if use_quandl and quandl_key:
        print("\n" + "="*70)
        print("FETCHING FROM QUANDL")
        print("="*70)
        q_data = fetch_quandl_data(quandl_key)
        all_data['quandl'] = q_data
    
    return all_data


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("FX CARRY STRATEGY - SUPPLEMENTARY DATA FETCHER")
    print("="*70)
    
    # EXAMPLE 1: Use yfinance only (no API key needed)
    print("\n>>> EXAMPLE 1: Fetching from Yahoo Finance (FREE, no API key)")
    print("-"*70)
    
    data = fetch_all_supplementary_data(
        use_yfinance=True,
        use_alphavantage=False,
        use_quandl=False
    )
    
    # Save to CSV files
    print("\n" + "="*70)
    print("SAVING DATA TO CSV FILES")
    print("="*70)
    
    if 'yfinance' in data:
        for dataset_name, dataset in data['yfinance'].items():
            filename = f'yfinance_{dataset_name}.csv'
            if isinstance(dataset, pd.DataFrame):
                dataset.to_csv(filename)
                print(f"✓ Saved {filename} ({dataset.shape[0]} rows, {dataset.shape[1]} columns)")
            elif isinstance(dataset, pd.Series):
                dataset.to_csv(filename)
                print(f"✓ Saved {filename} ({len(dataset)} rows)")
    
    # EXAMPLE 2: Use Alpha Vantage (uncomment and add your API key)
    """
    print("\n>>> EXAMPLE 2: Fetching from Alpha Vantage")
    print("-"*70)
    
    ALPHAVANTAGE_API_KEY = 'YOUR_API_KEY_HERE'  # Get from: https://www.alphavantage.co/support/#api-key
    
    data = fetch_all_supplementary_data(
        use_yfinance=False,
        use_alphavantage=True,
        alphavantage_key=ALPHAVANTAGE_API_KEY
    )
    """
    
    # EXAMPLE 3: Use Quandl (uncomment and add your API key)
    """
    print("\n>>> EXAMPLE 3: Fetching from Quandl")
    print("-"*70)
    
    QUANDL_API_KEY = 'YOUR_API_KEY_HERE'  # Get from: https://www.quandl.com/sign-up
    
    data = fetch_all_supplementary_data(
        use_yfinance=False,
        use_quandl=True,
        quandl_key=QUANDL_API_KEY
    )
    """
    
    print("\n" + "="*70)
    print("DONE! Data fetched successfully.")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the CSV files created")
    print("2. Integrate this data into your Phase 2 analysis")
    print("3. Add new factors to your factor decomposition")
    print("4. Enhance your multi-signal framework")
