"""
DEBUG VALUE FACTOR SIGNAL

Investigate why value factor works standalone but fails in backtest
"""

import pandas as pd
import numpy as np
from value_factor import ValueFactor

print("="*70)
print("🔍 DEBUGGING VALUE FACTOR SIGNAL")
print("="*70)
print()

# Initialize
value = ValueFactor()

# Download EUR data
value.download_fx_data('EUR', start_date='2020-01-01')
print()

# Calculate PPP deviation
ppp_data = value.calculate_ppp_deviation('EUR')
spot = value.fx_data['EUR']

print("📊 PPP Data Sample (last 10 days):")
print(ppp_data[['spot', 'ppp', 'deviation_pct', 'zscore']].tail(10))
print()

# Test signal direction
print("="*70)
print("🧪 TESTING SIGNAL DIRECTIONS")
print("="*70)
print()

# Value signal as calculated
value_signal = -ppp_data['zscore']  # Current implementation

print("Current Implementation:")
print("  value_signal = -zscore")
print("  Interpretation: Negative zscore (overvalued) → expect depreciation → NEGATIVE signal")
print()

# Test with FORWARD returns (3 months)
forward_returns_3m = spot.pct_change(63).shift(-63)

aligned_3m = pd.DataFrame({
    'value_signal': value_signal.values.flatten(),
    'forward_return_3m': forward_returns_3m.values.flatten()
}, index=ppp_data.index).dropna()

ic_3m = aligned_3m['value_signal'].corr(aligned_3m['forward_return_3m'])
print(f"IC with 3-month FORWARD returns: {ic_3m:+.3f}")

# Test with CURRENT returns (1 day)
current_returns = spot.pct_change()

aligned_1d = pd.DataFrame({
    'value_signal': value_signal.values.flatten(),
    'current_return': current_returns.values.flatten()
}, index=ppp_data.index).dropna()

ic_1d = aligned_1d['value_signal'].corr(aligned_1d['current_return'])
print(f"IC with 1-day CURRENT returns:   {ic_1d:+.3f}")
print()

# Test INVERTED signal
value_signal_inv = ppp_data['zscore']  # Inverted

aligned_3m_inv = pd.DataFrame({
    'value_signal': value_signal_inv.values.flatten(),
    'forward_return_3m': forward_returns_3m.values.flatten()
}, index=ppp_data.index).dropna()

ic_3m_inv = aligned_3m_inv['value_signal'].corr(aligned_3m_inv['forward_return_3m'])
print(f"INVERTED signal IC with 3-month forward: {ic_3m_inv:+.3f}")

# Backtest with 3-month forward returns
aligned_3m['strategy_return'] = (
    np.sign(aligned_3m['value_signal']) * aligned_3m['forward_return_3m']
)

sharpe_3m = aligned_3m['strategy_return'].mean() / aligned_3m['strategy_return'].std() * np.sqrt(252/63)
print(f"Sharpe with 3-month forward:     {sharpe_3m:+.3f}")

# Backtest with 1-day current returns
aligned_1d['strategy_return'] = (
    np.sign(aligned_1d['value_signal']) * aligned_1d['current_return']
)

sharpe_1d = aligned_1d['strategy_return'].mean() / aligned_1d['strategy_return'].std() * np.sqrt(252)
print(f"Sharpe with 1-day current:       {sharpe_1d:+.3f}")
print()

print("="*70)
print("💡 DIAGNOSIS")
print("="*70)
print()

print("Issue:")
print("  1. Value factor designed for 3-MONTH forward returns")
print("  2. Multi-factor backtest uses 1-DAY current returns")
print("  3. PPP mean reversion is SLOW (months), not days")
print()

print("Solutions:")
print("  A) Use value signal with 21-day forward returns (compromise)")
print("  B) Use value signal as position sizing (not timing)")
print("  C) Combine value with momentum (value for direction, momentum for timing)")
print()

# Test solution A: 21-day forward returns
forward_returns_21d = spot.pct_change(21).shift(-21)

aligned_21d = pd.DataFrame({
    'value_signal': value_signal.values.flatten(),
    'forward_return_21d': forward_returns_21d.values.flatten()
}, index=ppp_data.index).dropna()

ic_21d = aligned_21d['value_signal'].corr(aligned_21d['forward_return_21d'])
aligned_21d['strategy_return'] = (
    np.sign(aligned_21d['value_signal']) * aligned_21d['forward_return_21d']
)
sharpe_21d = aligned_21d['strategy_return'].mean() / aligned_21d['strategy_return'].std() * np.sqrt(252/21)

print(f"✅ SOLUTION A - 21-day forward returns:")
print(f"   IC:     {ic_21d:+.3f}")
print(f"   Sharpe: {sharpe_21d:+.3f}")
print()

print("="*70)
print("✅ RECOMMENDATION")
print("="*70)
print()
print("Modify multi_factor_backtest.py to use:")
print("  - 21-day holding period for value signals")
print("  - OR use value for position sizing only")
print("  - Momentum works for 1-day (trending)")
print("  - Value works for 21-63 day (mean reversion)")
