"""
VALUE FACTOR - Purchasing Power Parity (PPP) Deviation

Academic Basis:
- Rogoff (1996): "The Purchasing Power Parity Puzzle"
- Taylor & Taylor (2004): "The Purchasing Power Parity Debate"

Core Idea:
- Real exchange rates mean-revert to PPP over long horizons
- Deviation from PPP predicts future returns
- Overvalued currencies (spot > PPP) tend to depreciate
- Undervalued currencies (spot < PPP) tend to appreciate

Expected Performance:
- Sharpe: 0.20-0.30 standalone
- IC: 0.15-0.25 (moderate predictive power)
- Works best for mean-reverting currencies
- Time horizon: 3-12 months

Implementation:
- Calculate PPP ratio from price indices
- Deviation = (Spot - PPP) / PPP
- Positive deviation = overvalued → sell signal
- Negative deviation = undervalued → buy signal
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class ValueFactor:
    """
    Value factor based on PPP deviation
    """
    
    def __init__(self):
        self.fx_data = {}
        self.ppp_data = {}
        self.lookback_window = 252 * 5  # 5 years for PPP calculation
        
    def download_fx_data(self, pair, start_date='2015-01-01'):
        """
        Download spot FX rates from Yahoo Finance
        """
        ticker_map = {
            'EUR': 'EURUSD=X',
            'CHF': 'CHFUSD=X',
            'GBP': 'GBPUSD=X',
            'JPY': 'JPYUSD=X'
        }
        
        ticker = ticker_map.get(pair)
        if not ticker:
            raise ValueError(f"Unknown pair: {pair}")
        
        data = yf.download(ticker, start=start_date, progress=False)
        # Extract as Series, not DataFrame
        if isinstance(data['Close'], pd.DataFrame):
            self.fx_data[pair] = data['Close'].iloc[:, 0]
        else:
            self.fx_data[pair] = data['Close']
        
        print(f"   ✅ {ticker}: {len(data)} days")
        return self.fx_data[pair]
    
    def calculate_ppp_proxy(self, pair):
        """
        Calculate PPP proxy using inflation differential
        
        For simplicity, we'll use a historical average as baseline PPP
        and let deviations be measured from rolling mean
        
        Real implementation would use:
        - CPI data from FRED for US vs foreign country
        - PPP = CPI_US / CPI_Foreign * Spot_0
        
        Proxy approach:
        - Use 5-year rolling mean as "fair value"
        - Deviation = (Spot - Fair Value) / Fair Value
        """
        spot = self.fx_data[pair]
        
        # Rolling 5-year mean as PPP proxy
        ppp_proxy = spot.rolling(window=self.lookback_window, min_periods=252).mean()
        
        # Store for reference
        self.ppp_data[pair] = ppp_proxy
        
        return ppp_proxy
    
    def calculate_ppp_deviation(self, pair):
        """
        Calculate PPP deviation
        
        Returns:
        - DataFrame with spot, ppp, deviation, z-score
        """
        if pair not in self.fx_data:
            self.download_fx_data(pair)
        
        if pair not in self.ppp_data:
            self.calculate_ppp_proxy(pair)
        
        spot = self.fx_data[pair]
        ppp = self.ppp_data[pair]
        
        # Deviation in percentage terms
        deviation = (spot - ppp) / ppp * 100
        
        # Z-score over 2-year window
        zscore = (deviation - deviation.rolling(252*2).mean()) / deviation.rolling(252*2).std()
        
        # Create DataFrame with explicit index
        result = pd.DataFrame({
            'spot': spot.values,
            'ppp': ppp.values,
            'deviation_pct': deviation.values,
            'zscore': zscore.values
        }, index=spot.index)
        
        return result
    
    def generate_value_signal(self, pair):
        """
        Generate value signal based on PPP deviation
        
        Returns:
        - signal: -1 (overvalued, sell) to +1 (undervalued, buy)
        - strength: 0 to 1 (how extreme the deviation)
        - components: dict with details
        """
        ppp_data = self.calculate_ppp_deviation(pair)
        
        # Latest values
        latest = ppp_data.iloc[-1]
        
        # Signal based on z-score
        # Negative z-score = overvalued → sell signal (-1)
        # Positive z-score = undervalued → buy signal (+1)
        zscore = latest['zscore']
        
        # Clip z-score to [-3, +3] and normalize to [-1, +1]
        signal = np.clip(zscore / 3.0, -1, 1)
        
        # Note: We INVERT the signal because:
        # Positive deviation (overvalued) → expect depreciation → SELL
        # So signal = -zscore (negative deviation gives positive signal)
        signal = -signal
        
        # Strength based on absolute z-score
        strength = min(abs(zscore) / 2.0, 1.0)
        
        components = {
            'spot': latest['spot'],
            'ppp': latest['ppp'],
            'deviation_pct': latest['deviation_pct'],
            'zscore': zscore
        }
        
        return signal, strength, components
    
    def backtest_value(self, pair, forward_period=63):
        """
        Backtest value factor performance
        
        Args:
        - pair: Currency pair
        - forward_period: Forward return period in days (default 63 = 3 months)
        
        Returns:
        - sharpe: Sharpe ratio
        - total_return: Total return
        - win_rate: Percentage of profitable trades
        - ic: Information coefficient (correlation)
        """
        ppp_data = self.calculate_ppp_deviation(pair)
        spot = self.fx_data[pair]
        
        # Forward returns
        forward_returns = spot.pct_change(forward_period).shift(-forward_period)
        
        # Value signal = -zscore (overvalued → negative signal → expect depreciation)
        value_signal = -ppp_data['zscore']
        
        # Align
        aligned_data = pd.DataFrame({
            'value_signal': value_signal.values.flatten(),
            'forward_return': forward_returns.values.flatten()
        }, index=ppp_data.index).dropna()
        
        # Strategy returns
        aligned_data['strategy_return'] = (
            np.sign(aligned_data['value_signal']) * aligned_data['forward_return']
        )
        
        # Performance metrics
        sharpe = aligned_data['strategy_return'].mean() / aligned_data['strategy_return'].std() * np.sqrt(252/forward_period)
        total_return = (1 + aligned_data['strategy_return']).cumprod().iloc[-1] - 1
        win_rate = (aligned_data['strategy_return'] > 0).mean()
        
        # Information coefficient
        ic = aligned_data['value_signal'].corr(aligned_data['forward_return'])
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'win_rate': win_rate,
            'ic': ic,
            'num_signals': len(aligned_data)
        }


if __name__ == '__main__':
    print("="*70)
    print("🚀 VALUE FACTOR ANALYSIS (PPP Deviation)")
    print("="*70)
    print()
    
    # Initialize
    value = ValueFactor()
    
    # Download data
    print("📥 Downloading FX data...")
    pairs = ['EUR', 'CHF']
    for pair in pairs:
        value.download_fx_data(pair)
    print()
    
    # Current signals
    print("="*70)
    print("📊 CURRENT VALUE SIGNALS")
    print("="*70)
    print()
    
    for pair in pairs:
        signal, strength, components = value.generate_value_signal(pair)
        
        print(f"{pair}/USD:")
        print(f"   Signal: {signal:+.3f} (-1 to +1)")
        print(f"   Strength: {strength*100:.1f}%")
        print(f"   Z-Score: {components['zscore']:+.2f}")
        print(f"   Deviation: {components['deviation_pct']:+.2f}%")
        print(f"   Spot: {components['spot']:.4f}")
        print(f"   PPP: {components['ppp']:.4f}")
        print()
        
        # Interpretation
        if signal > 0.3:
            print(f"   💡 UNDERVALUED - Strong BUY signal")
        elif signal > 0.1:
            print(f"   💡 Slightly undervalued - Weak buy")
        elif signal < -0.3:
            print(f"   💡 OVERVALUED - Strong SELL signal")
        elif signal < -0.1:
            print(f"   💡 Slightly overvalued - Weak sell")
        else:
            print(f"   💡 FAIR VALUE - Neutral")
        print()
    
    # Backtest
    print("="*70)
    print("📈 HISTORICAL BACKTEST (3-month forward returns)")
    print("="*70)
    print()
    
    results = []
    for pair in pairs:
        perf = value.backtest_value(pair, forward_period=63)
        
        print(f"{pair}/USD:")
        print(f"   Sharpe Ratio: {perf['sharpe']:.3f}")
        print(f"   Total Return: {perf['total_return']*100:+.1f}%")
        print(f"   Win Rate: {perf['win_rate']*100:.1f}%")
        print(f"   Information Coefficient: {perf['ic']:+.3f}")
        print(f"   Number of Signals: {perf['num_signals']}")
        print()
        
        results.append({
            'Currency': pair,
            'Sharpe': perf['sharpe'],
            'Total Return': perf['total_return'],
            'Win Rate': perf['win_rate'],
            'IC': perf['ic'],
            'Trades': perf['num_signals']
        })
    
    # Summary table
    print("="*70)
    print("📊 SUMMARY TABLE")
    print("="*70)
    print()
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    
    print("="*70)
    print("✅ VALUE FACTOR ANALYSIS COMPLETE")
    print("="*70)
    print()
    
    print("💡 Key Insights:")
    print("   • PPP deviation measures fair value")
    print("   • Overvalued (spot > PPP) → expect depreciation")
    print("   • Undervalued (spot < PPP) → expect appreciation")
    print("   • Works best for mean-reverting currencies")
    print("   • Positive IC = value predicts future returns")
    print()
    
    print("📊 Next Steps:")
    print("   1. Compare with momentum factor")
    print("   2. Add dollar risk factor (DXY beta)")
    print("   3. Add VIX regime filter")
    print("   4. Run multi-factor backtest")
