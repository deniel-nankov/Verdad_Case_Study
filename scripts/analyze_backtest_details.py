
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def analyze_details():
    # 1. Load Backtest Results
    results_file = 'results/backtests/enhanced_strategy_results_20251127_224602.csv'
    df = pd.read_csv(results_file, index_col='Date', parse_dates=True)
    df['Returns'] = df['Equity'].pct_change()
    
    # 2. Load VIX Data
    print("Downloading VIX data...")
    vix = yf.download('^VIX', start=df.index[0], end=df.index[-1], progress=False)
    
    # Align data
    # Handle multi-level columns if present (yfinance update)
    if isinstance(vix.columns, pd.MultiIndex):
        vix_close = vix['Close'].iloc[:, 0]
    else:
        vix_close = vix['Close']
        
    df = df.join(vix_close.rename('VIX'), how='inner')
    
    # 3. Regime Analysis
    high_vol_mask = df['VIX'] > 20
    low_vol_mask = df['VIX'] <= 20
    
    high_vol_ret = df.loc[high_vol_mask, 'Returns'].mean() * 252
    low_vol_ret = df.loc[low_vol_mask, 'Returns'].mean() * 252
    
    print("\n=== Regime Analysis ===")
    print(f"High Volatility (>20) Annualized Return: {high_vol_ret:.4%}")
    print(f"Low Volatility (<=20) Annualized Return: {low_vol_ret:.4%}")
    
    # 4. Specific Examples
    # Find flat periods (avoided risk)
    df['Rolling_Vol'] = df['Returns'].rolling(window=5).std()
    flat_periods = df[df['Rolling_Vol'] < 1e-6]
    
    print("\n=== Potential 'Avoided' Periods (Flat Equity) ===")
    if not flat_periods.empty:
        print(flat_periods.head())
        print("...")
        print(flat_periods.tail())
    
    # Find best days
    print("\n=== Best Days ===")
    print(df.nlargest(5, 'Returns')[['Equity', 'Returns', 'VIX']])

if __name__ == "__main__":
    analyze_details()
