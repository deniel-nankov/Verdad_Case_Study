#!/usr/bin/env python3
"""
VERIFICATION: Check if multi-currency test used the EXACT same model as original
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print('='*90)
print('🔍 VERIFICATION: Comparing Original vs Multi-Currency Test')
print('='*90)
print()

# ============================================================================
# ORIGINAL MODEL (from backtest_all_strategies.py)
# ============================================================================

print('1️⃣  ORIGINAL MODEL (backtest_all_strategies.py features)')
print('-'*90)

eur_data = yf.download('EURUSD=X', start='2018-01-01', progress=False)
eur_data['returns'] = eur_data['Close'].pct_change()
eur_data = eur_data.dropna()

train_data = eur_data[eur_data.index < '2024-01-01'].copy()
test_data = eur_data[eur_data.index >= '2024-01-01'].copy()

def create_original_features(data):
    """EXACT features from backtest_all_strategies.py"""
    df = data.copy()
    features = pd.DataFrame(index=df.index)
    
    # Moving averages
    for window in [5, 10, 21, 63]:
        features[f'sma_{window}'] = df['Close'].rolling(window).mean() / df['Close'] - 1.0
        features[f'vol_{window}'] = df['returns'].rolling(window).std()
    
    # Momentum
    for window in [5, 10, 21]:
        features[f'momentum_{window}'] = df['Close'].pct_change(window)
    
    # Volume
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(21).mean()
    
    # Price position
    features['price_position'] = (df['Close'] - df['Low'].rolling(21).min()) / \
                                 (df['High'].rolling(21).max() - df['Low'].rolling(21).min() + 1e-8)
    
    return features.fillna(0)

# Train
train_features = create_original_features(train_data)
train_target = train_data['Close'].pct_change(21).shift(-21)
valid_idx = train_features.index.intersection(train_target.dropna().index)
X_train = train_features.loc[valid_idx].values
y_train = train_target.loc[valid_idx].values

rf_orig = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
xgb_orig = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
rf_orig.fit(X_train, y_train)
xgb_orig.fit(X_train, y_train)

# Test
test_features = create_original_features(test_data)
rf_pred = rf_orig.predict(test_features.values)
xgb_pred = xgb_orig.predict(test_features.values)
ml_pred = (rf_pred + xgb_pred) / 2.0
signals_orig = np.clip(ml_pred * 10, -1, 1)

# Calculate metrics
test_returns_series = test_data['Close'].pct_change(21).shift(-21)
valid_idx_test = test_features.index.intersection(test_returns_series.dropna().index)
positions = signals_orig[:len(valid_idx_test)]
returns = test_returns_series.loc[valid_idx_test].values
strategy_returns = positions * returns

total_return_orig = strategy_returns.sum()
sharpe_orig = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
equity = np.cumprod(1 + strategy_returns)
running_max = np.maximum.accumulate(equity)
drawdown = (equity - running_max) / running_max
max_dd_orig = drawdown.min()
trades_orig = np.sum(np.abs(np.diff(positions)) > 0.1)

print(f'Features: {len(train_features.columns)} ({", ".join(list(train_features.columns)[:5])}...)')
print(f'Training samples: {len(X_train)}')
print(f'Test samples: {len(returns)}')
print()
print('Results:')
print(f'  Return: {total_return_orig*100:+.2f}%')
print(f'  Sharpe: {sharpe_orig:+.3f}')
print(f'  Max DD: {max_dd_orig*100:.2f}%')
print(f'  Trades: {trades_orig}')
print()

# ============================================================================
# MULTI-CURRENCY TEST MODEL (from test_multi_currency.py)
# ============================================================================

print('2️⃣  MULTI-CURRENCY TEST MODEL (test_multi_currency.py features)')
print('-'*90)

def create_multi_features(df):
    """Features from test_multi_currency.py - CHECK IF DIFFERENT"""
    features = pd.DataFrame(index=df.index)
    
    # Core momentum features
    for window in [5, 10, 21, 42, 63]:
        features[f'mom_{window}'] = df['Close'].pct_change(window)
    
    # Volatility features
    for window in [5, 10, 21, 42]:
        features[f'vol_{window}'] = df['returns'].rolling(window).std()
    
    # Moving averages (normalized)
    for window in [5, 10, 21, 42, 63]:
        ma = df['Close'].rolling(window).mean()
        features[f'sma_{window}'] = (df['Close'] - ma) / df['Close']
    
    # Volume features
    features['volume_ratio_5'] = df['Volume'] / (df['Volume'].rolling(5).mean() + 1e-8)
    features['volume_ratio_21'] = df['Volume'] / (df['Volume'].rolling(21).mean() + 1e-8)
    features['volume_vol_21'] = df['Volume'].rolling(21).std() / (df['Volume'].rolling(21).mean() + 1e-8)
    
    # Trend
    features['trend_21'] = (df['Close'].rolling(21).mean() - df['Close'].rolling(63).mean()) / df['Close']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Price position
    features['price_pos_21'] = (df['Close'] - df['Close'].rolling(21).min()) / (df['Close'].rolling(21).max() - df['Close'].rolling(21).min() + 1e-8)
    
    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    features['bb_position'] = (df['Close'] - sma_20) / (2 * std_20 + 1e-8)
    features['bb_width'] = (4 * std_20) / (sma_20 + 1e-8)
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    features['macd'] = (ema_12 - ema_26) / df['Close']
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    
    # Volatility ratios
    features['vol_ratio_5_21'] = features['vol_5'] / (features['vol_21'] + 1e-8)
    features['vol_ratio_21_42'] = features['vol_21'] / (features['vol_42'] + 1e-8)
    
    # Momentum divergence
    features['mom_div_5_21'] = features['mom_5'] / (features['mom_21'] + 1e-8)
    
    return features.fillna(0)

# Train
train_features_multi = create_multi_features(train_data)
valid_idx_multi = train_features_multi.index.intersection(train_target.dropna().index)
X_train_multi = train_features_multi.loc[valid_idx_multi].values
y_train_multi = train_target.loc[valid_idx_multi].values

rf_multi = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
xgb_multi = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
rf_multi.fit(X_train_multi, y_train_multi)
xgb_multi.fit(X_train_multi, y_train_multi)

# Test
test_features_multi = create_multi_features(test_data)
rf_pred_multi = rf_multi.predict(test_features_multi.values)
xgb_pred_multi = xgb_multi.predict(test_features_multi.values)
ml_pred_multi = (rf_pred_multi + xgb_pred_multi) / 2.0
signals_multi = np.clip(ml_pred_multi * 10, -1, 1)

# Calculate metrics
test_returns_series_multi = test_data['Close'].pct_change(21).shift(-21)
valid_idx_test_multi = test_features_multi.index.intersection(test_returns_series_multi.dropna().index)
positions_multi = signals_multi[:len(valid_idx_test_multi)]
returns_multi = test_returns_series_multi.loc[valid_idx_test_multi].values
strategy_returns_multi = positions_multi * returns_multi

total_return_multi = strategy_returns_multi.sum()
sharpe_multi = strategy_returns_multi.mean() / (strategy_returns_multi.std() + 1e-8) * np.sqrt(252)
equity_multi = np.cumprod(1 + strategy_returns_multi)
running_max_multi = np.maximum.accumulate(equity_multi)
drawdown_multi = (equity_multi - running_max_multi) / running_max_multi
max_dd_multi = drawdown_multi.min()
trades_multi = np.sum(np.abs(np.diff(positions_multi)) > 0.1)

print(f'Features: {len(train_features_multi.columns)} ({", ".join(list(train_features_multi.columns)[:5])}...)')
print(f'Training samples: {len(X_train_multi)}')
print(f'Test samples: {len(returns)}')
print()
print('Results:')
print(f'  Return: {total_return_multi*100:+.2f}%')
print(f'  Sharpe: {sharpe_multi:+.3f}')
print(f'  Max DD: {max_dd_multi*100:.2f}%')
print(f'  Trades: {trades_multi}')
print()

# ============================================================================
# COMPARISON
# ============================================================================

print('='*90)
print('📊 COMPARISON')
print('='*90)
print()

print(f'{"Metric":<20} {"Original Model":<20} {"Multi-Currency Test":<20} {"Difference":<15}')
print('-'*90)
print(f'{"Features":<20} {len(train_features.columns):<20} {len(train_features_multi.columns):<20} {len(train_features_multi.columns) - len(train_features.columns):+d}')
print(f'{"Return":<20} {total_return_orig*100:+.2f}%{"":<15} {total_return_multi*100:+.2f}%{"":<15} {(total_return_multi-total_return_orig)*100:+.2f}%')
print(f'{"Sharpe":<20} {sharpe_orig:+.3f}{"":<16} {sharpe_multi:+.3f}{"":<16} {sharpe_multi-sharpe_orig:+.3f}')
print(f'{"Max DD":<20} {max_dd_orig*100:.2f}%{"":<15} {max_dd_multi*100:.2f}%{"":<15} {(max_dd_multi-max_dd_orig)*100:+.2f}%')
print(f'{"Trades":<20} {trades_orig:<20} {trades_multi:<20} {trades_multi-trades_orig:+d}')
print()

if abs(total_return_multi - total_return_orig) < 0.001:
    print('✅ MODELS ARE IDENTICAL - Same features and results')
else:
    print('⚠️  MODELS ARE DIFFERENT!')
    print()
    print('Key differences:')
    print(f'  • Original has {len(train_features.columns)} features')
    print(f'  • Multi-currency has {len(train_features_multi.columns)} features')
    print()
    print('The multi-currency test used MORE FEATURES than the original!')
    print('This explains the different (better) results.')
    print()
    print('Original features:', list(train_features.columns))
    print()
    print('Additional features in multi-currency:')
    extra_features = set(train_features_multi.columns) - set(train_features.columns)
    print(extra_features)

print()
print('='*90)
print('🎯 CONCLUSION')
print('='*90)
print()

if total_return_multi > total_return_orig * 2:
    print('The multi-currency test results (+8.41% EUR, +38.86% AUD, etc.) used')
    print('an ENHANCED model with more features than the original +1.41% baseline.')
    print()
    print('This is still valuable - it shows feature engineering improves the model!')
    print('But it\'s comparing apples to oranges with the original backtest.')
else:
    print('Models are similar enough to make valid comparisons.')
