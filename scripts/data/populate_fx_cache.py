"""
Populate FX cache with real data from yfinance
This creates a persistent cache so the trading system can run even without OANDA access
"""
import json
import pandas as pd
import yfinance as yf
from datetime import datetime

def get_latest_fx_rates():
    """Fetch latest FX rates from Yahoo Finance"""
    
    print("🔄 Fetching real-time FX rates from Yahoo Finance...")
    print("=" * 60)
    
    # Map currency pairs to Yahoo Finance tickers
    fx_tickers = {
        'USDEUR': 'EURUSD=X',
        'USDGBP': 'GBPUSD=X',
        'USDJPY': 'JPY=X',
        'USDCAD': 'CAD=X',
        'USDAUD': 'AUDUSD=X',
        'USDNZD': 'NZDUSD=X',
        'USDCHF': 'CHF=X',
        'USDSEK': 'SEK=X',
        'USDBRL': 'BRL=X',
        'USDMXN': 'MXN=X'
    }
    
    cache_data = {}
    
    for pair, ticker in fx_tickers.items():
        try:
            # Download recent data (last 5 days to ensure we get latest)
            data = yf.download(ticker, period='5d', progress=False)
            
            if not data.empty:
                # Get latest close price
                latest_close = data['Close'].iloc[-1]
                
                # For pairs quoted as CCY/USD, take inverse
                if pair in ['USDEUR', 'USDGBP', 'USDAUD', 'USDNZD']:
                    rate = 1 / latest_close
                else:
                    rate = latest_close
                
                # Create cache entry
                cache_data[f"fx_{pair}"] = {
                    'rate': float(rate),
                    'bid': float(rate * 0.9995),  # Approximate spread
                    'ask': float(rate * 1.0005),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'YAHOO_FINANCE',
                    'ticker': ticker,
                    'last_updated': data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                }
                
                print(f"✅ {pair:8s} = {rate:.4f} (from {ticker})")
            else:
                print(f"❌ {pair:8s} - No data available")
                
        except Exception as e:
            print(f"❌ {pair:8s} - Error: {e}")
    
    return cache_data

def save_cache(cache_data, filename='fx_data_cache.json'):
    """Save cache to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"\n✅ Saved {len(cache_data)} FX rates to {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Error saving cache: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("FX Data Cache Population Tool")
    print("=" * 60)
    print("\nThis tool fetches real FX rates from Yahoo Finance")
    print("and creates a persistent cache for the trading system.\n")
    
    # Fetch latest rates
    cache_data = get_latest_fx_rates()
    
    if cache_data:
        # Save to disk
        save_cache(cache_data)
        
        print("\n" + "=" * 60)
        print("Cache Summary")
        print("=" * 60)
        print(f"Total pairs cached: {len(cache_data)}")
        print(f"Data source: Yahoo Finance")
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n✅ Trading system can now run with real FX data!")
        print("   Even if OANDA API is unavailable, the system will")
        print("   use these cached real rates as fallback.")
        print("=" * 60)
    else:
        print("\n❌ Failed to populate cache")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
