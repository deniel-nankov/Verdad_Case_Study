# Data Sources Integration Guide

## Summary

I've successfully integrated **yfinance** (Yahoo Finance) as a free data source for your FX carry analysis. The data fetcher script has downloaded:

- **VIX** (6,289 observations) - Market volatility indicator
- **Dollar Index (DXY)** (6,316 observations) - USD strength measure
- **Commodity Index (DBC)** (4,758 observations) - Broad commodity exposure
- **Bond ETFs** (TLT, IEF, LQD, HYG) - Treasury and credit returns
- **Equity Indices** (S&P 500, Dow, FTSE, Nikkei, EM) - Global equity markets
- **Commodities** (Gold, Oil, Silver) - Individual commodity prices

All files are saved in CSV format and ready to use!

---

## Files Created

### Market Volatility
- `yf_vix.csv` - CBOE Volatility Index (2000-2025)

### Dollar & FX
- `yf_dollar_index.csv` - US Dollar Index (DXY)

### Commodities
- `yf_commodity_index.csv` - Broad commodity index (DBC ETF)
- `yf_gold_futures.csv` - Gold futures prices
- `yf_crude_oil_wti.csv` - WTI crude oil prices
- `yf_silver_futures.csv` - Silver futures prices

### Bonds & Fixed Income
- `yf_tlt.csv` - 20-Year Treasury Bond ETF
- `yf_ief.csv` - 7-10 Year Treasury Bond ETF
- `yf_lqd.csv` - Investment Grade Corporate Bonds
- `yf_hyg.csv` - High Yield Corporate Bonds

### Global Equities
- `yf_gspc.csv` - S&P 500 Index
- `yf_dji.csv` - Dow Jones Industrial Average
- `yf_ftse.csv` - FTSE 100 (UK)
- `yf_n225.csv` - Nikkei 225 (Japan)
- `yf_eem.csv` - Emerging Markets ETF

---

## How to Use in Your Analysis

### Option 1: Load in Phase 2 Notebook

```python
import pandas as pd

# Load VIX data
vix = pd.read_csv('yf_vix.csv', index_col=0, parse_dates=True)

# Load Dollar Index
dxy = pd.read_csv('yf_dollar_index.csv', index_col=0, parse_dates=True)

# Load commodity index
commodities = pd.read_csv('yf_commodity_index.csv', index_col=0, parse_dates=True)

# Merge with your existing macro_data
macro_data['VIX_yf'] = vix
macro_data['DXY_yf'] = dxy
macro_data['Commodities_yf'] = commodities
```

### Option 2: Add to Factor Analysis

```python
# Calculate returns for factor analysis
bond_returns = pd.read_csv('yf_tlt.csv', index_col=0, parse_dates=True)['Close'].pct_change()
commodity_returns = pd.read_csv('yf_commodity_index.csv', index_col=0, parse_dates=True)['Close'].pct_change()

# Add to your factor DataFrame
factors['Bonds_yf'] = bond_returns
factors['Commodities_yf'] = commodity_returns
```

### Option 3: Enhance Multi-Signal Framework

```python
# Load all signal data
signals = pd.DataFrame({
    'VIX': pd.read_csv('yf_vix.csv', index_col=0, parse_dates=True)['Close'],
    'DXY': pd.read_csv('yf_dollar_index.csv', index_col=0, parse_dates=True)['Close'],
    'Gold': pd.read_csv('yf_gold_futures.csv', index_col=0, parse_dates=True)['Close'],
    'Oil': pd.read_csv('yf_crude_oil_wti.csv', index_col=0, parse_dates=True)['Close'],
    'IG_Bonds': pd.read_csv('yf_lqd.csv', index_col=0, parse_dates=True)['Close'],
    'HY_Bonds': pd.read_csv('yf_hyg.csv', index_col=0, parse_dates=True)['Close']
})

# Calculate z-scores for signals
signals_z = (signals - signals.rolling(252).mean()) / signals.rolling(252).std()
```

---

## Alternative Data Sources (If You Want More)

### 1. **Alpha Vantage** (FREE API Key)
- **Best for:** FX data, technical indicators, fundamental data
- **Signup:** https://www.alphavantage.co/support/#api-key
- **Limits:** 5 calls/minute, 500 calls/day (free tier)
- **Install:** `pip install alpha_vantage`

```python
from alpha_vantage.foreignexchange import ForeignExchange
fx = ForeignExchange(key='YOUR_API_KEY', output_format='pandas')
data, meta = fx.get_currency_exchange_daily('EUR', 'USD', outputsize='full')
```

### 2. **Quandl** (FREE API Key)
- **Best for:** Economic data, FRED integration, alternative data
- **Signup:** https://www.quandl.com/sign-up
- **Limits:** 50 calls/day (free tier)
- **Install:** `pip install quandl`

```python
import quandl
quandl.ApiConfig.api_key = 'YOUR_API_KEY'
vix = quandl.get("FRED/VIXCLS", start_date="2000-01-01")
treasury_10y = quandl.get("FRED/DGS10", start_date="2000-01-01")
credit_spread = quandl.get("FRED/BAMLC0A0CM", start_date="2000-01-01")
```

### 3. **Tiingo** (FREE API Key)
- **Best for:** High-quality stock/ETF data, better than Yahoo
- **Signup:** https://www.tiingo.com/
- **Limits:** 500 calls/hour, 50,000 calls/month
- **Install:** `pip install tiingo`

```python
from tiingo import TiingoClient
config = {'api_key': 'YOUR_API_KEY'}
client = TiingoClient(config)
df = client.get_dataframe('SPY', startDate='2000-01-01')
```

### 4. **FRED (Federal Reserve)** (FREE API Key)
- **Best for:** US economic data (interest rates, inflation, etc.)
- **Signup:** https://fred.stlouisfed.org/docs/api/api_key.html
- **Limits:** No strict limits
- **Install:** `pip install fredapi`

```python
from fredapi import Fred
fred = Fred(api_key='YOUR_API_KEY')
vix = fred.get_series('VIXCLS', observation_start='2000-01-01')
treasury_10y = fred.get_series('DGS10', observation_start='2000-01-01')
treasury_2y = fred.get_series('DGS2', observation_start='2000-01-01')
credit_spread_ig = fred.get_series('BAMLC0A0CM', observation_start='2000-01-01')
term_spread = fred.get_series('T10Y2Y', observation_start='2000-01-01')
```

---

## Comparison Table

| Source | Cost | API Key? | Best For | Limits (Free) |
|--------|------|----------|----------|---------------|
| **yfinance** | FREE | ❌ No | Quick data, ETFs, indices | Reasonable use |
| **Alpha Vantage** | FREE | ✅ Yes | FX data, indicators | 500/day |
| **Quandl** | FREE | ✅ Yes | Economic data, FRED | 50/day |
| **Tiingo** | FREE | ✅ Yes | Quality stock data | 50K/month |
| **FRED** | FREE | ✅ Yes | US economic indicators | Unlimited |
| Quantopian | ❌ SHUT DOWN | - | - | - |
| Zipline | Open Source | No | Backtesting engine | - |
| LOBSTER | $$$ Paid | Yes | Order book data | Expensive |
| IEX Cloud | Freemium | Yes | Real-time quotes | Limited |
| Barchart | Freemium | Yes | Futures/options | Limited |
| Trading Economics | Freemium | Yes | Global economics | Limited |

---

## Recommendation for Phase 3

For your FX carry analysis, I recommend:

1. **Primary:** Use **yfinance** (already working!) for ETFs, indices, and commodities
2. **Secondary:** Add **FRED API** for US economic data (VIX, yields, credit spreads, term spreads)
3. **Optional:** Add **Alpha Vantage** if you need more FX data or want to verify your existing FX data

### Why This Combination?

- **yfinance:** Already fetched data, no setup needed
- **FRED:** Best source for macro indicators (free, reliable, well-documented)
- Together they cover 90% of what you need for factor analysis and multi-signal frameworks

---

## Quick Start: Add FRED Data

```bash
pip install fredapi
```

```python
from fredapi import Fred
fred = Fred(api_key='YOUR_FREE_API_KEY')  # Get from https://fred.stlouisfed.org/

# Fetch key indicators
data = pd.DataFrame({
    'VIX': fred.get_series('VIXCLS'),
    'Treasury_10Y': fred.get_series('DGS10'),
    'Treasury_2Y': fred.get_series('DGS2'),
    'Term_Spread': fred.get_series('T10Y2Y'),
    'Credit_Spread_IG': fred.get_series('BAMLC0A0CM'),
    'Credit_Spread_HY': fred.get_series('BAMLH0A0HYM2')
})

# Merge with your existing analysis
macro_data_enhanced = macro_data.join(data, how='left')
```

---

## Next Steps

1. ✅ **Data fetched** - yfinance data is ready to use
2. 📝 **Optional:** Sign up for FRED API key (2 minutes, instant approval)
3. 📊 **Integrate:** Add these data sources to your Phase 2 notebook
4. 🚀 **Analyze:** Use them in factor decomposition and signal testing

Let me know if you'd like help integrating any specific data source into your notebooks!
