#!/usr/bin/env python3
"""
BREAKTHROUGH ATTEMPT - Focus on High-Quality Signals
=====================================================
Hypothesis: EUR/USD 2024-2025 may have structural regime change making 
historical patterns less predictive. Let's try:

1. Longer training history (2015-2023 instead of 2018-2023)
2. Focus on proven alpha factors only
3. Simpler model with regularization
4. Transaction cost aware position sizing
5. Only trade when signal is VERY strong

This is about quality over quantity.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

print('='*90)
print('💎 BREAKTHROUGH ATTEMPT - High-Quality Signals Only')
print('='*90)
print()

# ============================================================================
# TRY 1: LONGER HISTORY
# ============================================================================

print('📊 Loading EUR/USD data (extended history: 2015-2025)...')
eur_data = yf.download('EURUSD=X', start='2015-01-01', progress=False)

# Flatten multi-index
if isinstance(eur_data.columns, pd.MultiIndex):
    eur_data.columns = eur_data.columns.get_level_values(0)

eur_data['returns'] = eur_data['Close'].pct_change()
eur_data = eur_data.dropna()
print(f'✅ Loaded {len(eur_data)} days (vs 2045 before)')
print(f'   Training: 2015-2023 ({len(eur_data[eur_data.index < "2024-01-01"])} days)')
print(f'   Testing:  2024-2025 ({len(eur_data[eur_data.index >= "2024-01-01"])} days)')
print()

# ============================================================================
# TRY 2: PROVEN ALPHA FACTORS ONLY
# ============================================================================

print('🎯 Creating high-quality features (proven alpha factors)...')

features = pd.DataFrame(index=eur_data.index)

# 1. Momentum (well-established factor)
features['mom_5'] = eur_data['Close'].pct_change(5)
features['mom_21'] = eur_data['Close'].pct_change(21)
features['mom_63'] = eur_data['Close'].pct_change(63)

# 2. Trend (moving average crossovers)
ma_5 = eur_data['Close'].rolling(5).mean()
ma_21 = eur_data['Close'].rolling(21).mean()
ma_63 = eur_data['Close'].rolling(63).mean()

features['ma_cross_5_21'] = (ma_5 - ma_21) / eur_data['Close']
features['ma_cross_21_63'] = (ma_21 - ma_63) / eur_data['Close']

# 3. Volatility (risk-adjusted returns)
vol_21 = eur_data['returns'].rolling(21).std()
vol_63 = eur_data['returns'].rolling(63).std()

features['vol_21'] = vol_21
features['vol_63'] = vol_63
features['sharpe_21'] = features['mom_21'] / (vol_21 + 1e-8)

# 4. Mean reversion (deviation from MA)
features['mean_rev_21'] = (eur_data['Close'] - ma_21) / (vol_21 * eur_data['Close'] + 1e-8)
features['mean_rev_63'] = (eur_data['Close'] - ma_63) / (vol_63 * eur_data['Close'] + 1e-8)

# 5. Volume (institutional activity)
vol_ma = eur_data['Volume'].rolling(21).mean()
features['volume_surge'] = (eur_data['Volume'] - vol_ma) / (vol_ma + 1e-8)

# 6. Price action (candlestick patterns)
features['body_size'] = (eur_data['Close'] - eur_data['Open']).abs() / eur_data['Close']
features['wick_size'] = (eur_data['High'] - eur_data['Low']) / eur_data['Close']

features = features.fillna(0)

print(f'   Created {len(features.columns)} high-quality features')
print(f'   Feature list: {", ".join(features.columns.tolist())}')
print()

# ============================================================================
# TRY 3: SIMPLE REGULARIZED MODEL
# ============================================================================

print('🤖 Training simple regularized ensemble...')

# Create target
target = eur_data['Close'].pct_change(21).shift(-21)

# Split
train_mask = eur_data.index < '2024-01-01'
X_train = features[train_mask].iloc[:-21]
y_train = target[train_mask].iloc[:-21]

# Remove NaN
valid = ~y_train.isna()
X_train = X_train[valid]
y_train = y_train[valid]

print(f'   Training samples: {len(X_train)}')

# Simple models with regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Ridge regression (linear with L2 regularization)
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)

# Conservative Random Forest (prevent overfitting)
rf = RandomForestRegressor(
    n_estimators=50,  # Fewer trees
    max_depth=5,      # Shallower trees
    min_samples_leaf=10,  # More samples per leaf
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

# Conservative XGBoost
xgb = XGBRegressor(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.01,  # Much slower learning
    reg_alpha=1.0,       # L1 regularization
    reg_lambda=1.0,      # L2 regularization
    random_state=42
)
xgb.fit(X_train_scaled, y_train)

print(f'   ✅ Trained 3 conservative models (Ridge, RF, XGB)')
print()

# ============================================================================
# TRY 4: QUALITY-BASED POSITION SIZING
# ============================================================================

print('💎 Generating high-quality signals only...')

# Test data
X_test = features[~train_mask]
y_test = target[~train_mask]

valid_test = ~y_test.isna()
X_test = X_test[valid_test]
y_test = y_test[valid_test]

# Predict with all models
X_test_scaled = scaler.transform(X_test)

pred_ridge = ridge.predict(X_test_scaled)
pred_rf = rf.predict(X_test_scaled)
pred_xgb = xgb.predict(X_test_scaled)

# Calculate agreement (confidence)
predictions_matrix = np.column_stack([pred_ridge, pred_rf, pred_xgb])
mean_pred = predictions_matrix.mean(axis=1)
std_pred = predictions_matrix.std(axis=1)

# Signal quality score (0 to 1)
# High quality = models agree AND prediction magnitude is large
magnitude = np.abs(mean_pred)
agreement = 1.0 / (1.0 + std_pred * 100)

quality = magnitude * agreement
quality_normalized = quality / (quality.max() + 1e-8)

# ONLY TRADE HIGH QUALITY SIGNALS (top 30%)
quality_threshold = np.percentile(quality_normalized, 70)

print(f'   Quality threshold: {quality_threshold:.3f} (top 30% of signals)')
print(f'   High quality signals: {(quality_normalized > quality_threshold).sum()} / {len(quality_normalized)}')

# Position sizing
base_positions = np.sign(mean_pred) * quality_normalized * 0.5  # Conservative scaling

# Only trade when quality is high
positions = np.where(quality_normalized > quality_threshold, base_positions, 0)
positions = np.clip(positions, -1, 1)

print(f'   Active positions: {(np.abs(positions) > 0.01).sum()} / {len(positions)}')
print()

# ============================================================================
# TRY 5: TRANSACTION COST AWARE
# ============================================================================

print('💰 Applying transaction cost optimization...')

# Reduce unnecessary rebalancing
position_series = pd.Series(positions, index=X_test.index)

# Only change position if signal change > threshold
transaction_threshold = 0.15  # Need 15% change to trade
filtered_positions = position_series.iloc[0]  # Start with first signal
position_list = [filtered_positions]

for i in range(1, len(position_series)):
    current_signal = position_series.iloc[i]
    prev_position = position_list[-1]
    
    change = abs(current_signal - prev_position)
    
    if change > transaction_threshold:
        position_list.append(current_signal)
    else:
        position_list.append(prev_position)  # Hold previous position

filtered_positions = pd.Series(position_list, index=position_series.index)

# Calculate turnover reduction
original_turnover = np.abs(np.diff(positions)).sum()
filtered_turnover = np.abs(np.diff(filtered_positions.values)).sum()

print(f'   Turnover reduction: {original_turnover:.2f} → {filtered_turnover:.2f} ({(1-filtered_turnover/original_turnover)*100:.1f}% less)')

# Adjust for transaction costs (1bp per trade)
returns_gross = filtered_positions * y_test
transaction_costs = np.abs(np.diff(filtered_positions.values)) * 0.0001
transaction_costs = np.insert(transaction_costs, 0, 0)  # No cost on first position

returns_net = returns_gross - transaction_costs

print(f'   Transaction costs: {transaction_costs.sum()*100:.3f}%')
print()

# ============================================================================
# RESULTS
# ============================================================================

# Metrics
total_return = returns_net.sum()
sharpe = returns_net.mean() / (returns_net.std() + 1e-8) * np.sqrt(252)

equity = (1 + returns_net).cumprod()
running_max = equity.expanding().max()
drawdown = (equity - running_max) / running_max
max_dd = drawdown.min()

winning = returns_net[returns_net > 0]
win_rate = len(winning) / len(returns_net)

trades = np.sum(np.abs(np.diff(filtered_positions.values)) > 0.01)

print('='*90)
print('📊 BREAKTHROUGH MODEL RESULTS (2024-2025)')
print('='*90)
print()
print(f'   Total Return (net): {total_return*100:+.2f}%')
print(f'   Sharpe Ratio:       {sharpe:+.3f}')
print(f'   Max Drawdown:       {max_dd*100:.2f}%')
print(f'   Win Rate:           {win_rate*100:.1f}%')
print(f'   Trades:             {trades}')
print()

# Compare all attempts
print('📈 COMPLETE COMPARISON:')
print(f'   Original ML:      +1.41% | Sharpe +0.896 | 27 trades | -0.67% DD')
print(f'   Advanced:         +0.13% | Sharpe +0.090 | 4 trades | -1.27% DD')
print(f'   Smart Kelly:     +12.00% | Sharpe +0.491 | 197 trades | -52.51% DD')
print(f'   Balanced:         -4.29% | Sharpe -1.034 | 19 trades | -15.95% DD')
print(f'   Systematic:       -4.52% | Sharpe -1.546 | 7 trades | -9.14% DD')
print(f'   Breakthrough:     {total_return*100:+.2f}% | Sharpe {sharpe:+.3f} | {trades} trades | {max_dd*100:.2f}% DD')
print()

# Final evaluation
improvement = total_return * 100 - 1.41
sharpe_improvement = sharpe - 0.896

if improvement > 0.5 and sharpe > 0.7:
    print('   ✅ BREAKTHROUGH! Significant improvement achieved!')
    success_level = 'MAJOR'
elif improvement > 0 and sharpe_improvement > 0:
    print('   ✅ SUCCESS! Both metrics improved!')
    success_level = 'SUCCESS'
elif total_return > 0.02:
    print('   ✅ SOLID RETURNS! Positive performance')
    success_level = 'GOOD'
else:
    print('   📊 The original model remains the best performer')
    success_level = 'ORIGINAL_BEST'

print()
print('='*90)
print('💡 KEY LEARNINGS')
print('='*90)
print()

if success_level in ['MAJOR', 'SUCCESS']:
    print('🎉 Winning factors:')
    print('   ✓ Extended training history (2015-2023 vs 2018-2023)')
    print('   ✓ Focus on proven alpha factors only')
    print('   ✓ Regularized models (prevent overfitting)')
    print('   ✓ Quality-based filtering (only high-confidence trades)')
    print('   ✓ Transaction cost optimization')
else:
    print('📚 Important insights:')
    print('   • EUR/USD is highly efficient - hard to beat consistently')
    print('   • Original model (+1.41% / 0.896 Sharpe) is actually excellent')
    print('   • 2024-2025 may have different regime than training period')
    print()
    print('🎯 To improve further, consider:')
    print('   1. Alternative data (sentiment, positioning, flows)')
    print('   2. Multi-asset portfolio (diversification across pairs)')
    print('   3. Different markets (EM FX, commodities, rates)')
    print('   4. Shorter horizons (intraday, weekly vs monthly)')
    print('   5. Live market microstructure (order flow, depth)')

print()
print('='*90)
