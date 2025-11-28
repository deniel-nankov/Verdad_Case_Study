"""
Momentum Factor for FX Trading
Based on: "Currency Momentum Strategies" (Menkhoff et al. 2012)

Key Finding: 12-month momentum has Sharpe ratio of 0.40-0.50 in FX markets
- Winner currencies continue outperforming
- Loser currencies continue underperforming
- Effect persists up to 12 months

Signal: Buy currencies with high past returns, sell with low past returns
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MomentumFactor:
    """
    Calculate momentum signals for FX pairs
    
    Lookback periods:
    - 1 month (21 days): Short-term momentum
    - 3 months (63 days): Medium-term momentum  
    - 6 months (126 days): Intermediate momentum
    - 12 months (252 days): Long-term momentum (BEST PERFORMER)
    """
    
    def __init__(self, lookback_periods=[21, 63, 126, 252]):
        """
        Parameters:
        - lookback_periods: List of momentum lookback windows in days
        """
        self.lookback_periods = lookback_periods
        self.price_data = {}
        
    def download_fx_data(self, pairs=['EURUSD=X', 'CHFUSD=X'], start_date='2015-01-01'):
        """Download FX price history from Yahoo Finance"""
        print(f"\n📥 Downloading FX data for momentum calculation...")
        
        for pair in pairs:
            try:
                data = yf.download(pair, start=start_date, progress=False)
                if not data.empty:
                    self.price_data[pair] = data['Close']
                    print(f"   ✅ {pair}: {len(data)} days")
                else:
                    print(f"   ❌ {pair}: No data")
            except Exception as e:
                print(f"   ❌ {pair}: Error - {e}")
        
        return self.price_data
    
    def calculate_momentum(self, prices, lookback=252):
        """
        Calculate momentum signal for given lookback period
        
        Formula: (Price today / Price N days ago) - 1
        
        Returns:
        - Positive signal: Currency has appreciated (bullish momentum)
        - Negative signal: Currency has depreciated (bearish momentum)
        """
        if len(prices) < lookback:
            return pd.Series(np.nan, index=prices.index)
        
        # Calculate cumulative return over lookback period
        momentum = prices.pct_change(lookback)
        
        return momentum
    
    def calculate_all_momentum_signals(self, pair):
        """Calculate momentum for all lookback periods"""
        if pair not in self.price_data:
            raise ValueError(f"No price data for {pair}. Run download_fx_data() first.")
        
        prices = self.price_data[pair]
        signals = pd.DataFrame(index=prices.index)
        
        for period in self.lookback_periods:
            col_name = f'momentum_{period}d'
            signals[col_name] = self.calculate_momentum(prices, lookback=period)
        
        # Composite momentum (weighted average - favor 12M)
        # Academic literature shows 12M is most predictive
        weights = {
            21: 0.10,   # 10% weight on 1M
            63: 0.15,   # 15% weight on 3M
            126: 0.25,  # 25% weight on 6M
            252: 0.50   # 50% weight on 12M (MOST IMPORTANT)
        }
        
        signals['momentum_composite'] = sum(
            weights.get(period, 0) * signals[f'momentum_{period}d'] 
            for period in self.lookback_periods
        )
        
        return signals
    
    def calculate_momentum_zscore(self, momentum, window=252):
        """
        Normalize momentum to z-score for better signal interpretation
        
        Z-score > 1: Strong positive momentum (BUY)
        Z-score < -1: Strong negative momentum (SELL)
        """
        mean = momentum.rolling(window).mean()
        std = momentum.rolling(window).std()
        zscore = (momentum - mean) / std
        return zscore
    
    def generate_momentum_signal(self, pair, current_date=None):
        """
        Generate momentum trading signal for a specific date
        
        Returns:
        - signal: Normalized momentum score (-1 to +1)
        - strength: Signal confidence (0 to 1)
        - components: Individual momentum periods
        """
        signals = self.calculate_all_momentum_signals(pair)
        
        if current_date is None:
            current_date = signals.index[-1]
        
        if current_date not in signals.index:
            return None
        
        # Get composite momentum
        composite = signals.loc[current_date, 'momentum_composite']
        
        # Calculate z-score for normalization
        zscore = self.calculate_momentum_zscore(
            signals['momentum_composite'], 
            window=252
        ).loc[current_date]
        
        # Normalize to -1 to +1 range (cap at +/- 3 std devs)
        signal = np.clip(zscore / 3.0, -1, 1)
        
        # Signal strength = how many periods agree
        components = {}
        agreements = 0
        for period in self.lookback_periods:
            mom = signals.loc[current_date, f'momentum_{period}d']
            components[f'{period}d'] = mom
            
            # Does this period agree with composite signal direction?
            if np.sign(mom) == np.sign(composite):
                agreements += 1
        
        strength = agreements / len(self.lookback_periods)
        
        return {
            'signal': signal,
            'strength': strength,
            'zscore': zscore,
            'composite_return': composite,
            'components': components
        }
    
    def backtest_momentum(self, pair, start_date='2015-01-01'):
        """
        Backtest momentum strategy on historical data
        
        Returns performance metrics
        """
        signals = self.calculate_all_momentum_signals(pair)
        prices = self.price_data[pair]
        
        # Forward returns (what we're trying to predict)
        forward_returns = prices.pct_change(21).shift(-21)
        
        # Align signals and returns - flatten to 1D arrays
        aligned_data = pd.DataFrame({
            'momentum': signals['momentum_composite'].values.flatten(),
            'forward_return': forward_returns.values.flatten()
        }, index=signals.index).dropna()
        
        # Strategy returns: momentum signal * forward return
        # Buy when momentum positive, sell when negative
        aligned_data['strategy_return'] = (
            np.sign(aligned_data['momentum']) * aligned_data['forward_return']
        )
        
        # Performance metrics
        sharpe = aligned_data['strategy_return'].mean() / aligned_data['strategy_return'].std() * np.sqrt(252/21)
        total_return = (1 + aligned_data['strategy_return']).cumprod().iloc[-1] - 1
        win_rate = (aligned_data['strategy_return'] > 0).mean()
        
        # Information coefficient (IC) - correlation between signal and returns
        ic = aligned_data['momentum'].corr(aligned_data['forward_return'])
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'win_rate': win_rate,
            'information_coefficient': ic,
            'num_trades': len(aligned_data)
        }


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🚀 MOMENTUM FACTOR ANALYSIS")
    print("="*70)
    
    # Initialize momentum factor
    momentum = MomentumFactor(lookback_periods=[21, 63, 126, 252])
    
    # Download FX data
    pairs = ['EURUSD=X', 'CHFUSD=X']
    momentum.download_fx_data(pairs, start_date='2015-01-01')
    
    print("\n" + "="*70)
    print("📊 CURRENT MOMENTUM SIGNALS")
    print("="*70)
    
    # Generate current signals
    for pair in pairs:
        result = momentum.generate_momentum_signal(pair)
        
        currency = pair.replace('USD=X', '').replace('=X', '')
        
        print(f"\n{currency}/USD:")
        print(f"   Signal: {result['signal']:+.3f} (-1 to +1)")
        print(f"   Strength: {result['strength']:.1%}")
        print(f"   Z-Score: {result['zscore']:+.2f}")
        print(f"   Composite Return: {result['composite_return']:+.2%}")
        
        print(f"\n   Components:")
        for period, value in result['components'].items():
            print(f"      {period:>4s}: {value:+.2%}")
        
        # Interpretation
        if abs(result['signal']) > 0.5:
            direction = "BULLISH" if result['signal'] > 0 else "BEARISH"
            confidence = "HIGH" if result['strength'] > 0.75 else "MODERATE"
            print(f"\n   💡 {confidence} {direction} momentum")
        else:
            print(f"\n   💡 NEUTRAL momentum")
    
    print("\n" + "="*70)
    print("📈 HISTORICAL BACKTEST")
    print("="*70)
    
    # Backtest each pair
    results_table = []
    
    for pair in pairs:
        currency = pair.replace('USD=X', '').replace('=X', '')
        perf = momentum.backtest_momentum(pair)
        
        results_table.append({
            'Currency': currency,
            'Sharpe': perf['sharpe'],
            'Total Return': perf['total_return'],
            'Win Rate': perf['win_rate'],
            'IC': perf['information_coefficient'],
            'Trades': perf['num_trades']
        })
        
        print(f"\n{currency}/USD:")
        print(f"   Sharpe Ratio: {perf['sharpe']:.3f}")
        print(f"   Total Return: {perf['total_return']:+.1%}")
        print(f"   Win Rate: {perf['win_rate']:.1%}")
        print(f"   Information Coefficient: {perf['information_coefficient']:.3f}")
        print(f"   Number of Signals: {perf['num_trades']}")
    
    # Summary table
    results_df = pd.DataFrame(results_table)
    
    print("\n" + "="*70)
    print("📊 SUMMARY TABLE")
    print("="*70)
    print("\n" + results_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ MOMENTUM FACTOR ANALYSIS COMPLETE")
    print("="*70)
    
    print("\n💡 Key Insights:")
    print("   • 12-month momentum is most predictive")
    print("   • Positive IC = momentum predicts future returns")
    print("   • Sharpe > 0.30 indicates strong factor")
    print("   • Combine with carry for enhanced performance")
    
    print("\n📊 Next Steps:")
    print("   1. Integrate with existing carry strategy")
    print("   2. Add value factor (PPP deviation)")
    print("   3. Add dollar risk factor (DXY beta)")
    print("   4. Run multi-factor backtest")
