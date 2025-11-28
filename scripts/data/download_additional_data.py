"""
Download Additional Macro Data for FX Carry Analysis Enhancement
Uses FRED API and Yahoo Finance to fetch factors and signals
"""

import pandas as pd
import yfinance as yf
from fredapi import Fred
import numpy as np
from datetime import datetime

# Configuration
FRED_API_KEY = 'b4a18aac3a462b6951ee89d9fef027cb'
START_DATE = '2000-01-01'
END_DATE = '2025-11-01'

def download_fred_data():
    """Download data from FRED API."""
    print("=" * 80)
    print("DOWNLOADING DATA FROM FRED API")
    print("=" * 80)
    
    fred = Fred(api_key=FRED_API_KEY)
    
    fred_series = {
        'VIX': 'VIXCLS',
        'Treasury_10Y': 'DGS10',
        'Treasury_2Y': 'DGS2',
        'Treasury_3M': 'DGS3MO',
        'Term_Spread': 'T10Y2Y',
        'IG_Spread': 'BAMLC0A0CM',
        'HY_Spread': 'BAMLH0A0HYM2',
        'TED_Spread': 'TEDRATE',
        'Dollar_Index': 'DTWEXBGS',
        'Fed_Funds': 'DFF',
        'CPI': 'CPIAUCSL',
        'Unemployment': 'UNRATE'
    }
    
    fred_data = {}
    
    for name, series_id in fred_series.items():
        try:
            print(f"  Downloading {name} ({series_id})...", end='')
            data = fred.get_series(series_id, observation_start=START_DATE)
            fred_data[name] = data
            print(f" ✓ ({len(data)} observations)")
        except Exception as e:
            print(f" ✗ Error: {str(e)}")
            fred_data[name] = pd.Series(dtype=float)
    
    return pd.DataFrame(fred_data)


def download_yahoo_data():
    """Download data from Yahoo Finance."""
    print("\n" + "=" * 80)
    print("DOWNLOADING DATA FROM YAHOO FINANCE")
    print("=" * 80)
    
    tickers = {
        'DXY': 'DX-Y.NYB',           # Dollar Index
        'Commodities': 'DBC',         # DB Commodity Index
        'Gold': 'GLD',                # Gold ETF
        'Oil': 'USO',                 # Oil ETF
        'Bonds_20Y': 'TLT',           # 20+ Year Treasury Bond ETF
        'Bonds_7_10Y': 'IEF',         # 7-10 Year Treasury Bond ETF
        'IG_Bonds': 'LQD',            # Investment Grade Corporate Bonds
        'HY_Bonds': 'HYG',            # High Yield Corporate Bonds
        'EM_Equities': 'EEM',         # Emerging Market Equities
        'Europe_Equities': 'VGK',     # European Equities
        'Asia_Equities': 'VPL',       # Asia-Pacific Equities
        'REITs': 'VNQ'                # Real Estate
    }
    
    yf_data = {}
    
    for name, ticker in tickers.items():
        try:
            print(f"  Downloading {name} ({ticker})...", end='')
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
            if not df.empty and 'Adj Close' in df.columns:
                yf_data[name] = df['Adj Close'].squeeze()
                print(f" ✓ ({len(df)} observations)")
            elif not df.empty and 'Close' in df.columns:
                yf_data[name] = df['Close'].squeeze()
                print(f" ✓ ({len(df)} observations)")
            else:
                print(f" ✗ No data")
                yf_data[name] = pd.Series(dtype=float, name=name)
        except Exception as e:
            print(f" ✗ Error: {str(e)}")
            yf_data[name] = pd.Series(dtype=float, name=name)
    
    # Align all series to same index before creating DataFrame
    if yf_data:
        all_indices = [s.index for s in yf_data.values() if isinstance(s, pd.Series) and not s.empty]
        if all_indices:
            combined_index = pd.Index(sorted(set().union(*all_indices)))
            for name in yf_data:
                if isinstance(yf_data[name], pd.Series):
                    yf_data[name] = yf_data[name].reindex(combined_index)
    
    return pd.DataFrame(yf_data)


def calculate_returns(df):
    """Calculate returns for price series."""
    print("\n" + "=" * 80)
    print("CALCULATING RETURNS")
    print("=" * 80)
    
    returns = pd.DataFrame()
    
    price_columns = ['DXY', 'Commodities', 'Gold', 'Oil', 'Bonds_20Y', 'Bonds_7_10Y',
                     'IG_Bonds', 'HY_Bonds', 'EM_Equities', 'Europe_Equities', 
                     'Asia_Equities', 'REITs']
    
    for col in price_columns:
        if col in df.columns and not df[col].isna().all():
            returns[col + '_Return'] = df[col].pct_change()
            print(f"  {col}_Return calculated ✓")
    
    return returns


def main():
    """Main function to download and save all data."""
    print("\n" + "=" * 80)
    print("FX CARRY ANALYSIS - ADDITIONAL DATA DOWNLOAD")
    print("=" * 80)
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print("=" * 80 + "\n")
    
    # Download data
    fred_df = download_fred_data()
    yahoo_df = download_yahoo_data()
    returns_df = calculate_returns(yahoo_df)
    
    # Combine all data
    all_data = pd.concat([fred_df, yahoo_df, returns_df], axis=1)
    
    # Summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total columns: {len(all_data.columns)}")
    print(f"Date range: {all_data.index.min()} to {all_data.index.max()}")
    print(f"Total rows: {len(all_data)}")
    print(f"\nColumns:\n{list(all_data.columns)}")
    
    # Data quality check
    print("\n" + "=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)
    missing_pct = (all_data.isna().sum() / len(all_data) * 100).round(2)
    print("\nMissing Data (%):")
    print(missing_pct.sort_values(ascending=False).head(10))
    
    # Save to CSV
    output_file = 'additional_macro_data.csv'
    all_data.to_csv(output_file)
    print(f"\n✅ Data saved to '{output_file}'")
    
    # Save metadata
    metadata = {
        'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'start_date': START_DATE,
        'end_date': END_DATE,
        'total_columns': len(all_data.columns),
        'total_rows': len(all_data),
        'columns': list(all_data.columns)
    }
    
    import json
    with open('data_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to 'data_metadata.json'")
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    
    return all_data


if __name__ == "__main__":
    data = main()
