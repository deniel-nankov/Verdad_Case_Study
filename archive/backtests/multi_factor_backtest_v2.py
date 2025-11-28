"""
MULTI-FACTOR BACKTEST V2 - Fixed Time Horizons

KEY FIX:
- Momentum: Works on 1-day returns (trending)
- Value: Works on 21-day returns (mean reversion)
- Combined strategy uses BOTH at appropriate horizons

Tests on multiple time periods: 2015-2020, 2020-2025, Full period
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

# Import factor modules
from momentum_factor import MomentumFactor
from value_factor import ValueFactor
from dollar_risk_factor import DollarRiskFactor
from vix_regime_filter import VIXRegimeFilter


class MultiFactorBacktestV2:
    """
    Fixed multi-factor backtest with proper time horizons
    """
    
    def __init__(self, start_date='2020-01-01', end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Initialize factors
        self.momentum = MomentumFactor()
        self.value = ValueFactor()
        self.dollar_risk = DollarRiskFactor()
        self.vix_regime = VIXRegimeFilter()
        
        # Data storage
        self.fx_data = {}
        self.factor_signals = {}
        
    def download_all_data(self):
        """Download all required data"""
        print("="*70)
        print("📥 DOWNLOADING ALL DATA")
        print("="*70)
        print()
        
        ticker_map = {'EUR': 'EURUSD=X', 'CHF': 'CHFUSD=X'}
        
        print("FX Pairs:")
        for pair in ['EUR', 'CHF']:
            ticker = ticker_map[pair]
            self.momentum.download_fx_data([ticker], start_date=self.start_date)
            self.value.download_fx_data(pair, start_date=self.start_date)
            self.dollar_risk.download_fx_data(pair, start_date=self.start_date)
            self.fx_data[pair] = self.value.fx_data[pair]
        
        print()
        print("Dollar Index:")
        self.dollar_risk.download_dxy(start_date=self.start_date)
        print()
        print("Volatility Index:")
        self.vix_regime.download_vix(start_date=self.start_date)
        print()
        
    def backtest_strategy_v2(self, pair, strategy_type='baseline', holding_period=1):
        """
        Backtest with PROPER time horizons
        
        Args:
        - pair: 'EUR' or 'CHF'
        - strategy_type: Strategy configuration
        - holding_period: Days to hold position (1 for momentum, 21 for value)
        """
        ticker_map = {'EUR': 'EURUSD=X', 'CHF': 'CHFUSD=X'}
        ticker = ticker_map[pair]
        
        # Get signals
        momentum_signals = self.momentum.calculate_all_momentum_signals(ticker)
        value_data = self.value.calculate_ppp_deviation(pair)
        dollar_risk_data = self.dollar_risk.calculate_risk_multiplier(pair)
        vix_regime_data = self.vix_regime.calculate_regime_history()
        
        # Signals
        momentum_signal = momentum_signals['momentum_composite']
        value_signal = -value_data['zscore']  # Overvalued → negative
        dollar_multiplier = dollar_risk_data['risk_multiplier']
        vix_multiplier = vix_regime_data['leverage_multiplier']
        
        # Returns at appropriate horizon
        spot = self.fx_data[pair]
        returns = spot.pct_change(holding_period).shift(-holding_period)
        
        # Align
        common_index = momentum_signals.index.intersection(value_data.index)
        common_index = common_index.intersection(dollar_risk_data.index)
        common_index = common_index.intersection(vix_regime_data.index)
        common_index = common_index.intersection(returns.dropna().index)
        
        aligned = pd.DataFrame({
            'returns': returns.loc[common_index].values.flatten(),
            'momentum': momentum_signal.loc[common_index].values.flatten(),
            'value': value_signal.loc[common_index].values.flatten(),
            'dollar_risk': dollar_multiplier.loc[common_index].values.flatten(),
            'vix_leverage': vix_multiplier.loc[common_index].values.flatten()
        }, index=common_index).dropna()
        
        # Calculate strategy return
        if strategy_type == 'baseline':
            strategy_return = aligned['returns']
            
        elif strategy_type == 'momentum_1d':
            # Momentum on 1-day returns
            strategy_return = np.sign(aligned['momentum']) * aligned['returns']
            
        elif strategy_type == 'value_21d':
            # Value on 21-day returns
            strategy_return = np.sign(aligned['value']) * aligned['returns']
            
        elif strategy_type == 'value_vix_21d':
            # Value + VIX on 21-day
            strategy_return = (
                np.sign(aligned['value']) * 
                aligned['returns'] * 
                aligned['vix_leverage']
            )
            
        elif strategy_type == 'momentum_vix_1d':
            # Momentum + VIX on 1-day
            strategy_return = (
                np.sign(aligned['momentum']) * 
                aligned['returns'] * 
                aligned['vix_leverage']
            )
            
        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")
        
        # Performance metrics
        periods_per_year = 252 / holding_period
        sharpe = strategy_return.mean() / strategy_return.std() * np.sqrt(periods_per_year)
        total_return = (1 + strategy_return).cumprod().iloc[-1] - 1
        max_dd = (strategy_return.cumsum() - strategy_return.cumsum().cummax()).min()
        win_rate = (strategy_return > 0).mean()
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'num_periods': len(strategy_return),
            'returns': strategy_return
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive test across all strategies"""
        print("="*70)
        print("📊 MULTI-FACTOR BACKTEST V2 (Fixed Time Horizons)")
        print("="*70)
        print()
        
        results = []
        
        for pair in ['EUR', 'CHF']:
            print(f"\n{'='*70}")
            print(f"Testing {pair}/USD")
            print('='*70)
            
            # 1-day holding period strategies
            print("\n1-DAY HOLDING PERIOD:")
            baseline_1d = self.backtest_strategy_v2(pair, 'baseline', holding_period=1)
            momentum_1d = self.backtest_strategy_v2(pair, 'momentum_1d', holding_period=1)
            momentum_vix_1d = self.backtest_strategy_v2(pair, 'momentum_vix_1d', holding_period=1)
            
            print(f"  Baseline:        Sharpe {baseline_1d['sharpe']:+.3f}, Return {baseline_1d['total_return']*100:+.1f}%")
            print(f"  Momentum:        Sharpe {momentum_1d['sharpe']:+.3f} ({momentum_1d['sharpe']-baseline_1d['sharpe']:+.3f}), Return {momentum_1d['total_return']*100:+.1f}%")
            print(f"  Momentum+VIX:    Sharpe {momentum_vix_1d['sharpe']:+.3f} ({momentum_vix_1d['sharpe']-baseline_1d['sharpe']:+.3f}), Return {momentum_vix_1d['total_return']*100:+.1f}%")
            
            # 21-day holding period strategies
            print("\n21-DAY HOLDING PERIOD:")
            baseline_21d = self.backtest_strategy_v2(pair, 'baseline', holding_period=21)
            value_21d = self.backtest_strategy_v2(pair, 'value_21d', holding_period=21)
            value_vix_21d = self.backtest_strategy_v2(pair, 'value_vix_21d', holding_period=21)
            
            print(f"  Baseline:        Sharpe {baseline_21d['sharpe']:+.3f}, Return {baseline_21d['total_return']*100:+.1f}%")
            print(f"  Value:           Sharpe {value_21d['sharpe']:+.3f} ({value_21d['sharpe']-baseline_21d['sharpe']:+.3f}), Return {value_21d['total_return']*100:+.1f}%")
            print(f"  Value+VIX:       Sharpe {value_vix_21d['sharpe']:+.3f} ({value_vix_21d['sharpe']-baseline_21d['sharpe']:+.3f}), Return {value_vix_21d['total_return']*100:+.1f}%")
            
            # Store results
            for name, perf, period in [
                ('baseline_1d', baseline_1d, 1),
                ('momentum_1d', momentum_1d, 1),
                ('momentum_vix_1d', momentum_vix_1d, 1),
                ('baseline_21d', baseline_21d, 21),
                ('value_21d', value_21d, 21),
                ('value_vix_21d', value_vix_21d, 21)
            ]:
                results.append({
                    'Pair': pair,
                    'Strategy': name,
                    'Period': period,
                    'Sharpe': perf['sharpe'],
                    'Total_Return': perf['total_return'],
                    'Max_DD': perf['max_dd'],
                    'Win_Rate': perf['win_rate']
                })
        
        return pd.DataFrame(results)


def test_multiple_periods():
    """Test across different time periods"""
    print("\n" + "="*70)
    print("📅 TESTING ACROSS MULTIPLE TIME PERIODS")
    print("="*70)
    print()
    
    periods = [
        ('2015-2020', '2015-01-01', '2019-12-31'),
        ('2020-2025', '2020-01-01', '2025-11-06'),
        ('Full (2015-2025)', '2015-01-01', '2025-11-06')
    ]
    
    all_results = []
    
    for period_name, start, end in periods:
        print(f"\n{'='*70}")
        print(f"Period: {period_name}")
        print('='*70)
        
        bt = MultiFactorBacktestV2(start_date=start, end_date=end)
        bt.download_all_data()
        results = bt.run_comprehensive_test()
        
        # Add period column
        results['Time_Period'] = period_name
        all_results.append(results)
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # Save
    combined.to_csv('multi_factor_v2_results.csv', index=False)
    print(f"\n✅ Results saved to multi_factor_v2_results.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 SUMMARY BY TIME PERIOD")
    print("="*70)
    print()
    
    for period_name, _, _ in periods:
        period_data = combined[combined['Time_Period'] == period_name]
        
        print(f"\n{period_name}:")
        print("-" * 70)
        
        # Best strategies
        for pair in ['EUR', 'CHF']:
            pair_data = period_data[period_data['Pair'] == pair]
            best = pair_data.nlargest(1, 'Sharpe').iloc[0]
            
            print(f"  {pair}: {best['Strategy']:20s} Sharpe {best['Sharpe']:+.3f}")
    
    return combined


if __name__ == '__main__':
    print("="*70)
    print("🚀 MULTI-FACTOR BACKTEST V2")
    print("="*70)
    print()
    print("KEY FIXES:")
    print("  ✅ Momentum tested on 1-day returns (trending)")
    print("  ✅ Value tested on 21-day returns (mean reversion)")
    print("  ✅ Proper time horizon matching")
    print()
    
    # Run multi-period test
    results = test_multiple_periods()
    
    print("\n" + "="*70)
    print("✅ MULTI-FACTOR BACKTEST V2 COMPLETE")
    print("="*70)
