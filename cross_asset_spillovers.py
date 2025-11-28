"""
Cross-Asset Momentum Spillovers Strategy
Uses equity and commodity momentum to predict FX movements

Academic basis: Moskowitz et al. (2012) - momentum predicts across asset classes
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class CrossAssetSpilloverStrategy:
    """
    Generates FX signals from cross-asset momentum spillovers
    
    Key relationships:
    - Equity momentum (SPY, EEM) → USD strength
    - Commodity momentum (GLD, USO) → Commodity currency strength (AUD, CAD)
    - Credit spreads (HYG-LQD) → Risk appetite → EM currencies
    - VIX term structure → Flight to quality
    """
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        Initialize cross-asset strategy
        
        Args:
            lookback_periods: Dictionary of lookback windows for momentum
        """
        
        self.lookback_periods = lookback_periods or {
            'equity_short': 21,      # 1 month
            'equity_medium': 63,     # 3 months
            'commodity_short': 21,
            'commodity_long': 126,   # 6 months
            'vix_short': 10,
            'credit': 21
        }
        
        # Asset tickers
        self.tickers = {
            'equity_us': 'SPY',       # S&P 500
            'equity_em': 'EEM',       # Emerging Markets
            'equity_eu': 'EZU',       # Eurozone
            'equity_jp': 'EWJ',       # Japan
            'gold': 'GLD',            # Gold
            'oil': 'USO',             # Oil
            'treasury': 'TLT',        # Long-term treasuries
            'credit_high': 'HYG',     # High yield bonds
            'credit_invest': 'LQD',   # Investment grade bonds
            'vix': '^VIX'             # Volatility index
        }
        
        self.data_cache = {}
        
    def download_cross_asset_data(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download cross-asset data from Yahoo Finance (FREE!)
        """
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📊 Downloading cross-asset data from {start_date} to {end_date}...")
        
        all_data = {}
        
        for name, ticker in self.tickers.items():
            try:
                print(f"   Downloading {name} ({ticker})...", end=' ')
                
                # Download data
                df = yf.download(ticker, start=start_date, end=end_date, 
                                progress=False, auto_adjust=False)
                
                if not df.empty:
                    # Handle multi-level columns
                    if isinstance(df.columns, pd.MultiIndex):
                        # Extract 'Adj Close' from multi-level index
                        if 'Adj Close' in df.columns.get_level_values(0):
                            all_data[name] = df['Adj Close'].iloc[:, 0]
                        elif 'Close' in df.columns.get_level_values(0):
                            all_data[name] = df['Close'].iloc[:, 0]
                        else:
                            all_data[name] = df.iloc[:, 0]
                    else:
                        # Single-level columns
                        if 'Adj Close' in df.columns:
                            all_data[name] = df['Adj Close']
                        elif 'Close' in df.columns:
                            all_data[name] = df['Close']
                        else:
                            all_data[name] = df.iloc[:, 0]
                    
                    print(f"✅ {len(df)} rows")
                else:
                    print("❌ No data")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        # Combine into single DataFrame
        df = pd.DataFrame(all_data)
        
        # Cache it
        self.data_cache = df
        
        print(f"\n✅ Downloaded {len(df)} rows, {len(df.columns)} assets")
        
        return df
    
    def calculate_momentum_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators across all asset classes
        
        Returns: DataFrame with momentum scores for each asset
        """
        
        signals = pd.DataFrame(index=data.index)
        
        # 1. Equity Momentum
        if 'equity_us' in data.columns:
            # Short-term (1 month)
            signals['spy_mom_1m'] = data['equity_us'].pct_change(
                self.lookback_periods['equity_short']
            )
            # Medium-term (3 months)
            signals['spy_mom_3m'] = data['equity_us'].pct_change(
                self.lookback_periods['equity_medium']
            )
        
        if 'equity_em' in data.columns:
            signals['eem_mom_1m'] = data['equity_em'].pct_change(
                self.lookback_periods['equity_short']
            )
        
        if 'equity_eu' in data.columns:
            signals['ezu_mom_1m'] = data['equity_eu'].pct_change(
                self.lookback_periods['equity_short']
            )
        
        if 'equity_jp' in data.columns:
            signals['ewj_mom_1m'] = data['equity_jp'].pct_change(
                self.lookback_periods['equity_short']
            )
        
        # 2. Commodity Momentum
        if 'gold' in data.columns:
            signals['gold_mom_1m'] = data['gold'].pct_change(
                self.lookback_periods['commodity_short']
            )
            signals['gold_mom_6m'] = data['gold'].pct_change(
                self.lookback_periods['commodity_long']
            )
        
        if 'oil' in data.columns:
            signals['oil_mom_1m'] = data['oil'].pct_change(
                self.lookback_periods['commodity_short']
            )
        
        # 3. Credit Spreads (Risk Appetite)
        if 'credit_high' in data.columns and 'credit_invest' in data.columns:
            # HYG/LQD ratio: Higher = more risk appetite
            credit_ratio = data['credit_high'] / data['credit_invest']
            signals['credit_spread_mom'] = credit_ratio.pct_change(
                self.lookback_periods['credit']
            )
        
        # 4. VIX Term Structure (Fear Gauge)
        if 'vix' in data.columns:
            signals['vix_level'] = data['vix']
            signals['vix_change'] = data['vix'].pct_change(
                self.lookback_periods['vix_short']
            )
            # Inverted: Lower VIX = risk-on
            signals['vix_signal'] = -signals['vix_change']
        
        # 5. Flight to Quality
        if 'treasury' in data.columns:
            signals['tlt_mom'] = data['treasury'].pct_change(
                self.lookback_periods['equity_short']
            )
        
        return signals
    
    def generate_fx_signals(
        self, 
        momentum_signals: pd.DataFrame,
        currencies: list = None
    ) -> Dict[str, float]:
        """
        Convert cross-asset momentum into FX trading signals
        
        Spillover relationships:
        - SPY momentum → USD strength (1-2 month lead)
        - Gold momentum → CHF, JPY strength (safe haven)
        - Oil momentum → CAD, MXN strength (commodity currencies)
        - EM momentum → BRL, MXN strength
        - VIX spike → JPY, CHF strength (flight to quality)
        """
        
        if currencies is None:
            currencies = ['AUD', 'BRL', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'MXN']
        
        # Handle empty momentum data
        if len(momentum_signals) == 0:
            return {curr: 0.0 for curr in currencies}
        
        # Get latest signals (most recent row)
        latest = momentum_signals.iloc[-1]
        
        fx_signals = {}
        
        for currency in currencies:
            signal = 0.0
            
            # USD Strength (from equity momentum)
            if 'spy_mom_1m' in latest.index and not pd.isna(latest['spy_mom_1m']):
                # Strong SPY → Strong USD → Weak foreign currencies
                spy_signal = -latest['spy_mom_1m'] * 2.0  # Invert and amplify
                signal += spy_signal * 0.3  # 30% weight
            
            # Currency-specific factors
            if currency == 'JPY':
                # Safe haven: Rallies when VIX spikes or equities fall
                if 'vix_change' in latest.index and not pd.isna(latest['vix_change']):
                    signal += latest['vix_change'] * 0.4  # 40% weight
                if 'spy_mom_1m' in latest.index and not pd.isna(latest['spy_mom_1m']):
                    signal -= latest['spy_mom_1m'] * 0.3  # Inverse to SPY
            
            elif currency == 'CHF':
                # Safe haven + Gold correlation
                if 'gold_mom_1m' in latest.index and not pd.isna(latest['gold_mom_1m']):
                    signal += latest['gold_mom_1m'] * 0.3
                if 'vix_change' in latest.index and not pd.isna(latest['vix_change']):
                    signal += latest['vix_change'] * 0.3
            
            elif currency == 'AUD':
                # Commodity currency: Oil + Risk appetite
                if 'oil_mom_1m' in latest.index and not pd.isna(latest['oil_mom_1m']):
                    signal += latest['oil_mom_1m'] * 0.3
                if 'credit_spread_mom' in latest.index and not pd.isna(latest['credit_spread_mom']):
                    signal += latest['credit_spread_mom'] * 0.2
                if 'eem_mom_1m' in latest.index and not pd.isna(latest['eem_mom_1m']):
                    signal += latest['eem_mom_1m'] * 0.2
            
            elif currency == 'CAD':
                # Oil currency
                if 'oil_mom_1m' in latest.index and not pd.isna(latest['oil_mom_1m']):
                    signal += latest['oil_mom_1m'] * 0.5  # Strong correlation
                if 'spy_mom_1m' in latest.index and not pd.isna(latest['spy_mom_1m']):
                    signal -= latest['spy_mom_1m'] * 0.2  # Inverse USD
            
            elif currency in ['BRL', 'MXN']:
                # EM currencies: Risk appetite + commodity
                if 'eem_mom_1m' in latest.index and not pd.isna(latest['eem_mom_1m']):
                    signal += latest['eem_mom_1m'] * 0.4
                if 'credit_spread_mom' in latest.index and not pd.isna(latest['credit_spread_mom']):
                    signal += latest['credit_spread_mom'] * 0.3
                if 'vix_signal' in latest.index and not pd.isna(latest['vix_signal']):
                    signal += latest['vix_signal'] * 0.2
            
            elif currency == 'EUR':
                # European equity momentum + risk appetite
                if 'ezu_mom_1m' in latest.index and not pd.isna(latest['ezu_mom_1m']):
                    signal += latest['ezu_mom_1m'] * 0.3
                if 'spy_mom_1m' in latest.index and not pd.isna(latest['spy_mom_1m']):
                    signal -= latest['spy_mom_1m'] * 0.2  # Inverse USD
                if 'credit_spread_mom' in latest.index and not pd.isna(latest['credit_spread_mom']):
                    signal += latest['credit_spread_mom'] * 0.2
            
            elif currency == 'GBP':
                # Similar to EUR but slightly more risk-on
                if 'spy_mom_1m' in latest.index and not pd.isna(latest['spy_mom_1m']):
                    signal -= latest['spy_mom_1m'] * 0.2
                if 'credit_spread_mom' in latest.index and not pd.isna(latest['credit_spread_mom']):
                    signal += latest['credit_spread_mom'] * 0.3
            
            # Normalize to [-1, 1]
            fx_signals[currency] = np.clip(signal, -1.0, 1.0)
        
        return fx_signals
    
    def get_latest_signals(self, currencies: list = None) -> Dict[str, float]:
        """
        Convenience method to get current FX signals
        
        Downloads latest data and generates signals
        """
        
        # Download recent data (last 6 months)
        data = self.download_cross_asset_data(
            start_date=(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        )
        
        # Calculate momentum
        momentum = self.calculate_momentum_signals(data)
        
        # Generate FX signals
        signals = self.generate_fx_signals(momentum, currencies)
        
        return signals


def test_cross_asset_strategy():
    """Test cross-asset spillover signals"""
    
    print("\n" + "="*70)
    print("🌍 CROSS-ASSET SPILLOVER STRATEGY - TEST")
    print("="*70)
    
    strategy = CrossAssetSpilloverStrategy()
    
    # Download data
    print("\n📥 Step 1: Download Cross-Asset Data")
    print("   (Using Yahoo Finance - 100% FREE!)")
    
    data = strategy.download_cross_asset_data(
        start_date=(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    )
    
    # Calculate momentum
    print("\n📊 Step 2: Calculate Momentum Signals")
    momentum = strategy.calculate_momentum_signals(data)
    
    print(f"   Momentum indicators: {len(momentum.columns)}")
    
    if len(momentum) > 0:
        print(f"   Latest values:")
        for col in momentum.columns:
            val = momentum[col].iloc[-1]
            if not pd.isna(val):
                print(f"      {col:20s}: {val:+.4f}")
    else:
        print("   ⚠️  No momentum data - using demo mode")
    
    # Generate FX signals
    print("\n🎯 Step 3: Generate FX Signals")
    
    currencies = ['EUR', 'CHF', 'AUD', 'CAD', 'GBP', 'JPY']
    fx_signals = strategy.generate_fx_signals(momentum, currencies)
    
    print("\n💱 FX Trading Signals (from cross-asset momentum):")
    
    # Sort by signal strength
    sorted_signals = sorted(fx_signals.items(), key=lambda x: x[1], reverse=True)
    
    for currency, signal in sorted_signals:
        direction = "LONG" if signal > 0 else "SHORT"
        strength = abs(signal)
        
        if strength > 0.5:
            rating = "🔥 STRONG"
        elif strength > 0.25:
            rating = "✅ MODERATE"
        else:
            rating = "⚠️  WEAK"
        
        print(f"   {currency}: {signal:+.3f} | {direction:5s} | {rating}")
    
    # Show interpretation only if we have data
    if len(momentum) > 0:
        print("\n\n💡 Signal Interpretation:")
        print("   Positive signal → BUY currency (vs USD)")
        print("   Negative signal → SELL currency (vs USD)")
        print("\n   Key Drivers:")
        
        latest = momentum.iloc[-1]
    
        latest = momentum.iloc[-1]
        
        if 'spy_mom_1m' in latest.index:
            spy_mom = latest['spy_mom_1m']
            print(f"   - SPY momentum: {spy_mom:+.2%} → {'Strong USD' if spy_mom > 0 else 'Weak USD'}")
        
        if 'vix_change' in latest.index:
            vix_chg = latest['vix_change']
            print(f"   - VIX change: {vix_chg:+.2%} → {'Risk-off (JPY/CHF up)' if vix_chg > 0 else 'Risk-on'}")
        
        if 'oil_mom_1m' in latest.index:
            oil_mom = latest['oil_mom_1m']
            print(f"   - Oil momentum: {oil_mom:+.2%} → {'CAD/AUD strength' if oil_mom > 0 else 'CAD/AUD weakness'}")
        
        if 'gold_mom_1m' in latest.index:
            gold_mom = latest['gold_mom_1m']
            print(f"   - Gold momentum: {gold_mom:+.2%} → {'Safe haven demand' if gold_mom > 0 else 'Risk-on'}")
    
    print("\n✅ Cross-Asset Strategy Test Complete!")
    print("\n💡 Next: Combine these signals with ML predictions for enhanced accuracy")
if __name__ == "__main__":
    test_cross_asset_strategy()
