"""
Advanced Data Loader for ML FX Strategy
Fetches and caches data from multiple sources:
- FX rates from OANDA/Yahoo Finance
- Macro data from FRED
- Market data from Yahoo Finance
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class MLDataLoader:
    """Unified data loader for ML FX strategy"""
    
    def __init__(self, fred_api_key: str, cache_dir: str = "./data_cache"):
        self.fred = Fred(api_key=fred_api_key)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Currency pairs we trade
        self.currencies = ['AUD', 'BRL', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'MXN']
        
    def load_all_data(self, start_date: str = '2010-01-01', 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load all data needed for ML models
        
        Returns:
            DataFrame with datetime index and all features
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print("📊 Loading comprehensive dataset for ML...")
        
        # 1. FX Data
        print("  ├─ Loading FX rates...")
        fx_data = self._load_fx_data(start_date, end_date)
        
        # 2. Interest Rates
        print("  ├─ Loading interest rates...")
        rates_data = self._load_interest_rates(start_date, end_date)
        
        # 3. Market Data
        print("  ├─ Loading market data (equities, VIX, etc.)...")
        market_data = self._load_market_data(start_date, end_date)
        
        # 4. Macro Data
        print("  ├─ Loading macro data (GDP, inflation, etc.)...")
        macro_data = self._load_macro_data(start_date, end_date)
        
        # 5. Commodity Data
        print("  ├─ Loading commodity data...")
        commodity_data = self._load_commodity_data(start_date, end_date)
        
        # Merge all datasets
        print("  └─ Merging datasets...")
        data = fx_data
        for df in [rates_data, market_data, macro_data, commodity_data]:
            data = data.join(df, how='outer')
        
        # Forward fill missing values (weekends, holidays)
        data = data.fillna(method='ffill').dropna()
        
        print(f"✅ Loaded {len(data)} rows, {len(data.columns)} columns")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        
        return data
    
    def _load_fx_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load FX rates from Yahoo Finance"""
        cache_file = f"{self.cache_dir}/fx_rates.parquet"
        
        # Check cache
        if os.path.exists(cache_file):
            cached = pd.read_parquet(cache_file)
            if len(cached) > 0 and cached.index[-1] >= pd.Timestamp(end_date):
                return cached
        
        fx_data = pd.DataFrame()
        
        # Yahoo Finance FX tickers
        fx_tickers = {
            'AUD': 'AUDUSD=X',
            'BRL': 'BRL=X',  # Yahoo uses different format for some
            'CAD': 'CADUSD=X',
            'CHF': 'CHF=X',
            'EUR': 'EURUSD=X',
            'GBP': 'GBPUSD=X',
            'JPY': 'JPY=X',
            'MXN': 'MXN=X'
        }
        
        for currency, ticker in fx_tickers.items():
            try:
                data_df = yf.download(ticker, start=start_date, end=end_date, 
                                     progress=False)
                
                # Handle both single and multi-index columns
                if isinstance(data_df.columns, pd.MultiIndex):
                    df = data_df['Close']
                else:
                    df = data_df['Close'] if 'Close' in data_df.columns else data_df['Adj Close']
                
                # Convert to USD per foreign (inverse for most)
                if currency in ['EUR', 'GBP', 'AUD']:
                    fx_data[f'{currency}_USD'] = df
                else:
                    fx_data[f'{currency}_USD'] = 1.0 / df
                    
            except Exception as e:
                print(f"    Warning: Could not load {currency}: {e}")
        
        # Cache
        fx_data.to_parquet(cache_file)
        return fx_data
    
    def _load_interest_rates(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load interest rates from FRED"""
        cache_file = f"{self.cache_dir}/interest_rates.parquet"
        
        if os.path.exists(cache_file):
            cached = pd.read_parquet(cache_file)
            if len(cached) > 0 and cached.index[-1] >= pd.Timestamp(end_date) - timedelta(days=7):
                return cached
        
        rates = pd.DataFrame()
        
        # FRED series for interest rates
        rate_series = {
            'USD': 'DFF',           # Fed Funds Rate
            'EUR': 'ECBDFR',        # ECB Deposit Facility Rate
            'GBP': 'GBPONTD156N',   # UK Official Bank Rate
            'JPY': 'IRSTCI01JPM156N',  # Japan Call Rate
            'CAD': 'IRSTCI01CAM156N',  # Canada Overnight Rate
            'AUD': 'IRSTCI01AUM156N',  # Australia Cash Rate
            'CHF': 'IRSTCI01CHM156N',  # Swiss 3M Rate
        }
        
        for currency, series_id in rate_series.items():
            try:
                data = self.fred.get_series(series_id, 
                                           observation_start=start_date,
                                           observation_end=end_date)
                rates[f'{currency}_rate'] = data
            except Exception as e:
                print(f"    Warning: Could not load {currency} rate: {e}")
        
        # For missing currencies, use proxies or zeros
        for curr in self.currencies:
            if f'{curr}_rate' not in rates.columns:
                rates[f'{curr}_rate'] = 0.0
        
        rates.to_parquet(cache_file)
        return rates
    
    def _load_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load market data (equities, VIX, credit spreads, etc.)"""
        cache_file = f"{self.cache_dir}/market_data.parquet"
        
        if os.path.exists(cache_file):
            cached = pd.read_parquet(cache_file)
            if len(cached) > 0 and cached.index[-1] >= pd.Timestamp(end_date):
                return cached
        
        market = pd.DataFrame()
        
        # Download key market indicators
        tickers = {
            'SPX': '^GSPC',      # S&P 500
            'VIX': '^VIX',       # Volatility Index
            'DXY': 'DX-Y.NYB',   # Dollar Index
            'TLT': 'TLT',        # 20Y Treasury ETF
            'HYG': 'HYG',        # High Yield Bond ETF
            'LQD': 'LQD',        # Investment Grade Bond ETF
            'EEM': 'EEM',        # Emerging Markets ETF
        }
        
        for name, ticker in tickers.items():
            try:
                data_df = yf.download(ticker, start=start_date, end=end_date,
                                     progress=False)
                # Handle both single and multi-index columns
                if isinstance(data_df.columns, pd.MultiIndex):
                    df = data_df['Close'] if ('Close', ticker) in data_df.columns else data_df['Adj Close']
                else:
                    df = data_df['Close'] if 'Close' in data_df.columns else data_df.get('Adj Close', data_df.iloc[:, 0])
                market[name] = df
            except Exception as e:
                print(f"    Warning: Could not load {name}: {e}")
        
        # Calculate returns
        for col in market.columns:
            if not col.endswith('_ret'):  # Don't calculate returns on returns
                market[f'{col}_ret'] = market[col].pct_change()
        
        # FRED data for credit spreads
        try:
            market['credit_spread_ig'] = self.fred.get_series(
                'BAMLC0A0CM', observation_start=start_date, observation_end=end_date)
            market['credit_spread_hy'] = self.fred.get_series(
                'BAMLH0A0HYM2', observation_start=start_date, observation_end=end_date)
        except:
            pass
        
        # Term spread (10Y - 2Y)
        try:
            rate_10y = self.fred.get_series('DGS10', observation_start=start_date, 
                                           observation_end=end_date)
            rate_2y = self.fred.get_series('DGS2', observation_start=start_date,
                                          observation_end=end_date)
            market['term_spread'] = rate_10y - rate_2y
        except:
            pass
        
        market.to_parquet(cache_file)
        return market
    
    def _load_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load macro economic data from FRED"""
        cache_file = f"{self.cache_dir}/macro_data.parquet"
        
        if os.path.exists(cache_file):
            cached = pd.read_parquet(cache_file)
            if len(cached) > 0 and cached.index[-1] >= pd.Timestamp(end_date) - timedelta(days=30):
                return cached
        
        macro = pd.DataFrame()
        
        # Key macro series
        series = {
            'gdp_us': 'GDP',
            'cpi_us': 'CPIAUCSL',
            'unemployment_us': 'UNRATE',
            'pmi_us': 'MANEMP',
            'retail_sales_us': 'RSXFS',
        }
        
        for name, series_id in series.items():
            try:
                data = self.fred.get_series(series_id,
                                           observation_start=start_date,
                                           observation_end=end_date)
                macro[name] = data
                
                # Calculate YoY changes
                macro[f'{name}_yoy'] = data.pct_change(12)
            except Exception as e:
                print(f"    Warning: Could not load {name}: {e}")
        
        macro.to_parquet(cache_file)
        return macro
    
    def _load_commodity_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load commodity prices"""
        cache_file = f"{self.cache_dir}/commodity_data.parquet"
        
        if os.path.exists(cache_file):
            cached = pd.read_parquet(cache_file)
            if len(cached) > 0 and cached.index[-1] >= pd.Timestamp(end_date):
                return cached
        
        commodities = pd.DataFrame()
        
        tickers = {
            'gold': 'GC=F',
            'silver': 'SI=F',
            'oil': 'CL=F',
            'copper': 'HG=F',
        }
        
        for name, ticker in tickers.items():
            try:
                data_df = yf.download(ticker, start=start_date, end=end_date,
                                     progress=False)
                # Handle both single and multi-index columns
                if isinstance(data_df.columns, pd.MultiIndex):
                    df = data_df['Close'] if ('Close', ticker) in data_df.columns else data_df['Adj Close']
                else:
                    df = data_df['Close'] if 'Close' in data_df.columns else data_df.get('Adj Close', data_df.iloc[:, 0])
                commodities[name] = df
                commodities[f'{name}_ret'] = df.pct_change()
            except Exception as e:
                print(f"    Warning: Could not load {name}: {e}")
        
        commodities.to_parquet(cache_file)
        return commodities
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get most recent data for live trading"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        data = self.load_all_data(start_date, end_date)
        return data.tail(252)  # Last year of data
    
    def clear_cache(self):
        """Clear all cached data"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        print("✅ Cache cleared")


if __name__ == "__main__":
    # Test data loader
    from dotenv import load_dotenv
    load_dotenv()
    
    fred_key = os.getenv('FRED_API_KEY', 'b4a18aac3a462b6951ee89d9fef027cb')
    
    loader = MLDataLoader(fred_api_key=fred_key)
    
    # Load data
    data = loader.load_all_data(start_date='2015-01-01')
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Shape: {data.shape}")
    print(f"   Columns: {list(data.columns[:10])}...")
    print(f"\n   Sample (last 5 rows):")
    print(data.tail())
