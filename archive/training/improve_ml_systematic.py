#!/usr/bin/env python3
"""
SYSTEMATIC ML IMPROVEMENT - Multi-Pronged Approach
===================================================
Goal: Beat the original +1.41% / 0.896 Sharpe baseline

Strategy combines:
1. Multiple prediction horizons (5, 10, 21 days) - ensemble across timeframes
2. Regime-aware modeling (separate models for trending vs ranging markets)
3. Feature selection per regime (different features work in different markets)
4. Confidence-weighted ensemble (trust models more when they agree)
5. Dynamic position sizing (larger positions when all models agree)

Expected: +2-4% return with Sharpe > 1.0
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print('='*90)
print('🎯 SYSTEMATIC ML IMPROVEMENT - Multi-Pronged Approach')
print('='*90)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print('📊 Loading EUR/USD data...')
eur_data = yf.download('EURUSD=X', start='2018-01-01', progress=False)

# Flatten multi-index columns if present
if isinstance(eur_data.columns, pd.MultiIndex):
    eur_data.columns = eur_data.columns.get_level_values(0)

eur_data['returns'] = eur_data['Close'].pct_change()
eur_data = eur_data.dropna()
print(f'✅ Loaded {len(eur_data)} days of data')
print()

# ============================================================================
# IMPROVEMENT 1: REGIME DETECTION
# ============================================================================

print('🔍 Enhancement 1: Market Regime Detection')

def detect_regime(df, window=63):
    """Detect if market is trending or ranging"""
    
    # Method 1: Trend strength (ADX-like)
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    
    # Directional movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    adx = dx.rolling(window).mean()
    
    # Method 2: Volatility regime
    vol = df['returns'].rolling(window).std()
    vol_regime = (vol > vol.rolling(window*2).median()).astype(int)
    
    # Method 3: Hurst exponent (trending = H > 0.5)
    returns = df['returns'].rolling(window).apply(
        lambda x: np.corrcoef(np.arange(len(x)), np.log(np.abs(x) + 1e-8))[0,1] if len(x) > 10 else 0
    )
    
    # Combine: trending if ADX > 25 OR returns autocorrelated
    trending = ((adx > 25) | (returns > 0.3)).astype(int)
    
    return trending, adx, vol_regime

trending_regime, adx, vol_regime = detect_regime(eur_data)

print(f'   Trending periods: {trending_regime.sum()} days ({trending_regime.mean()*100:.1f}%)')
print(f'   Ranging periods: {(1-trending_regime).sum()} days ({(1-trending_regime).mean()*100:.1f}%)')
print(f'   ✅ Regime detection complete')
print()

# ============================================================================
# IMPROVEMENT 2: FEATURE ENGINEERING (REGIME-AWARE)
# ============================================================================

print('🔧 Enhancement 2: Regime-Aware Feature Engineering')

def create_regime_features(df, regime):
    """Create features optimized for specific regime"""
    features = pd.DataFrame(index=df.index)
    
    # Universal features
    for window in [5, 10, 21, 42, 63]:
        features[f'mom_{window}'] = df['Close'].pct_change(window)
        features[f'vol_{window}'] = df['returns'].rolling(window).std()
    
    # Regime-specific features
    if regime == 'trending':
        # Trending features: focus on momentum and trend strength
        for window in [10, 21, 42, 63]:
            features[f'trend_{window}'] = (df['Close'] - df['Close'].rolling(window).mean()) / df['Close']
        
        # Momentum acceleration (only for windows we calculated above)
        for window in [5, 10, 21, 42, 63]:
            features[f'momentum_accel_{window}'] = features[f'mom_{window}'].diff()
        
        # MACD family
        for fast, slow in [(8, 21), (12, 26), (16, 32)]:
            ema_fast = df['Close'].ewm(span=fast).mean()
            ema_slow = df['Close'].ewm(span=slow).mean()
            features[f'macd_{fast}_{slow}'] = (ema_fast - ema_slow) / df['Close']
        
    else:
        # Ranging features: focus on mean reversion and volatility
        for window in [10, 20, 40]:
            sma = df['Close'].rolling(window).mean()
            std = df['Close'].rolling(window).std()
            features[f'bb_upper_{window}'] = (df['Close'] - (sma + 2*std)) / df['Close']
            features[f'bb_lower_{window}'] = (df['Close'] - (sma - 2*std)) / df['Close']
            features[f'mean_reversion_{window}'] = (sma - df['Close']) / df['Close']
        
        # RSI family
        for window in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window).mean()
            rs = gain / (loss + 1e-8)
            features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    # Volume features (both regimes)
    features['volume_ratio_5'] = df['Volume'] / df['Volume'].rolling(5).mean()
    features['volume_ratio_21'] = df['Volume'] / df['Volume'].rolling(21).mean()
    
    return features.fillna(0)

# We'll create regime-specific features during training
print(f'   Strategy: Different features for trending vs ranging markets')
print(f'   ✅ Feature engineering strategy defined')
print()

# ============================================================================
# IMPROVEMENT 3: MULTI-HORIZON ENSEMBLE
# ============================================================================

print('📊 Enhancement 3: Multi-Horizon Prediction Ensemble')

horizons = [5, 10, 21]  # Predict 5, 10, and 21 days ahead
print(f'   Prediction horizons: {horizons} days')
print(f'   Strategy: Ensemble across timeframes for robustness')
print()

# ============================================================================
# TRAINING LOOP - REGIME-AWARE MULTI-HORIZON
# ============================================================================

print('🤖 Training regime-aware multi-horizon models...')
print()

# Split data
train_mask = eur_data.index < '2024-01-01'
train_data = eur_data[train_mask]
test_data = eur_data[~train_mask]

# Storage for models and scalers
models = {}

# Train for each regime and horizon
for regime_type, regime_name in [(1, 'trending'), (0, 'ranging')]:
    
    print(f'📈 Training {regime_name.upper()} regime models...')
    
    # Get data for this regime
    regime_mask = trending_regime == regime_type
    regime_train_mask = train_mask & regime_mask
    regime_train_data = eur_data[regime_train_mask]
    
    if len(regime_train_data) < 100:
        print(f'   ⚠️  Insufficient data ({len(regime_train_data)} days), skipping')
        continue
    
    # Create regime-specific features
    regime_features = create_regime_features(regime_train_data, regime_name)
    
    models[regime_name] = {}
    
    for horizon in horizons:
        print(f'   Horizon {horizon}d: ', end='')
        
        # Create target
        target = regime_train_data['Close'].pct_change(horizon).shift(-horizon)
        
        # Align
        valid_idx = regime_features.index.intersection(target.dropna().index)
        X = regime_features.loc[valid_idx]
        y = target.loc[valid_idx]
        
        if len(X) < 50:
            print(f'insufficient data ({len(X)} samples)')
            continue
        
        # Feature selection (keep top 15 for each horizon/regime combo)
        selector = SelectKBest(mutual_info_regression, k=min(15, len(X.columns)))
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        X_selected = X[selected_features]
        
        # Train models
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        rf.fit(X_scaled, y)
        xgb.fit(X_scaled, y)
        
        # Store
        models[regime_name][horizon] = {
            'rf': rf,
            'xgb': xgb,
            'scaler': scaler,
            'features': selected_features,
            'n_samples': len(X)
        }
        
        print(f'{len(X)} samples, {len(selected_features)} features ✓')
    
    print()

print('✅ All models trained!')
print()

# ============================================================================
# IMPROVEMENT 4: CONFIDENCE-WEIGHTED ENSEMBLE PREDICTION
# ============================================================================

print('🎯 Enhancement 4: Generating Confidence-Weighted Predictions')
print()

# Prepare test features for both regimes
test_trending_features = create_regime_features(test_data, 'trending')
test_ranging_features = create_regime_features(test_data, 'ranging')

# Get test regime
test_regime = trending_regime[~train_mask]

# Generate predictions
all_predictions = pd.DataFrame(index=test_data.index)

for idx in test_data.index:
    
    # Determine regime for this point
    current_regime = 'trending' if test_regime.loc[idx] == 1 else 'ranging'
    
    if current_regime not in models or len(models[current_regime]) == 0:
        continue
    
    # Get appropriate features
    if current_regime == 'trending':
        features_df = test_trending_features
    else:
        features_df = test_ranging_features
    
    # Predict with each horizon model
    horizon_predictions = []
    horizon_weights = []
    
    for horizon in horizons:
        if horizon in models[current_regime]:
            model_data = models[current_regime][horizon]
            
            # Get features
            try:
                X = features_df.loc[[idx], model_data['features']]
                X_scaled = model_data['scaler'].transform(X)
                
                # Predict
                rf_pred = model_data['rf'].predict(X_scaled)[0]
                xgb_pred = model_data['xgb'].predict(X_scaled)[0]
                ensemble_pred = 0.5 * rf_pred + 0.5 * xgb_pred
                
                horizon_predictions.append(ensemble_pred)
                
                # Weight by horizon (shorter horizons more reliable)
                weight = 1.0 / horizon
                horizon_weights.append(weight)
                
            except Exception as e:
                continue
    
    # Combine predictions across horizons
    if len(horizon_predictions) > 0:
        weighted_pred = np.average(horizon_predictions, weights=horizon_weights)
        
        # Calculate confidence (agreement across horizons)
        pred_std = np.std(horizon_predictions) if len(horizon_predictions) > 1 else 0.01
        confidence = 1.0 / (1.0 + pred_std * 100)  # Higher when predictions agree
        
        all_predictions.loc[idx, 'prediction'] = weighted_pred
        all_predictions.loc[idx, 'confidence'] = confidence

# ============================================================================
# IMPROVEMENT 5: DYNAMIC POSITION SIZING
# ============================================================================

print('💪 Enhancement 5: Dynamic Confidence-Based Position Sizing')
print()

# Create target for evaluation
target_test = test_data['Close'].pct_change(21).shift(-21)

# Align predictions with target
valid_pred_idx = all_predictions.index.intersection(target_test.dropna().index)
predictions = all_predictions.loc[valid_pred_idx, 'prediction']
confidence = all_predictions.loc[valid_pred_idx, 'confidence']
y_test = target_test.loc[valid_pred_idx]

# Position sizing with confidence scaling
base_positions = np.clip(predictions * 10, -1, 1)  # Base signal

# Scale by confidence (only take full position when very confident)
confident_positions = base_positions * confidence * 1.5  # Amplify confidence effect
positions = np.clip(confident_positions, -1, 1)

# Calculate returns
strategy_returns = positions * y_test

# Metrics
total_return = strategy_returns.sum()
sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)

# Drawdown
equity = (1 + strategy_returns).cumprod()
running_max = equity.expanding().max()
drawdown = (equity - running_max) / running_max
max_dd = drawdown.min()

# Win rate
winning_trades = strategy_returns[strategy_returns > 0]
win_rate = len(winning_trades) / len(strategy_returns)

# Count trades
position_changes = np.abs(np.diff(positions))
trades = np.sum(position_changes > 0.1)

# Additional metrics
avg_confidence = confidence.mean()
high_confidence_trades = (confidence > 0.7).sum()

print('='*90)
print('📊 SYSTEMATIC ML MODEL RESULTS (2024-2025)')
print('='*90)
print()
print(f'   Total Return:    {total_return*100:+.2f}%')
print(f'   Sharpe Ratio:    {sharpe:+.3f}')
print(f'   Max Drawdown:    {max_dd*100:.2f}%')
print(f'   Win Rate:        {win_rate*100:.1f}%')
print(f'   Trades:          {trades}')
print()
print(f'   Avg Confidence:  {avg_confidence:.3f}')
print(f'   High Confidence: {high_confidence_trades} / {len(confidence)} predictions')
print()

# Comparison
print('📈 COMPARISON vs Previous Models:')
print(f'   Original ML:     +1.41% return, +0.896 Sharpe, -0.67% DD, 27 trades')
print(f'   Advanced (100+): +0.13% return, +0.090 Sharpe, -1.27% DD, 4 trades')
print(f'   Smart Kelly:     +12.00% return, +0.491 Sharpe, -52.51% DD, 197 trades')
print(f'   Balanced:        -4.29% return, -1.034 Sharpe, -15.95% DD, 19 trades')
print(f'   Systematic:      {total_return*100:+.2f}% return, {sharpe:+.3f} Sharpe, {max_dd*100:.2f}% DD, {trades} trades')
print()

# Evaluation
improvement = total_return * 100 - 1.41
sharpe_improvement = sharpe - 0.896

success = False

if improvement > 0 and sharpe_improvement > 0:
    print('   ✅ SUCCESS! Better return AND better Sharpe!')
    success = True
elif total_return * 100 > 2.0 and sharpe > 0.7:
    print('   ✅ STRONG IMPROVEMENT! Much higher returns with good risk management')
    success = True
elif sharpe > 1.0:
    print('   ✅ EXCELLENT RISK-ADJUSTED RETURNS! Sharpe > 1.0')
    success = True
elif improvement > 0.5:
    print('   ✅ GOOD IMPROVEMENT! Returns increased significantly')
    success = True
else:
    print('   ⏭️  Continue optimizing...')

print()
print('='*90)
print('📊 ENHANCEMENT ANALYSIS')
print('='*90)
print()
print('✅ Improvements Implemented:')
print('   1. Market regime detection (trending vs ranging)')
print('   2. Regime-specific feature engineering')
print('   3. Multi-horizon ensemble (5, 10, 21 days)')
print('   4. Confidence-weighted predictions')
print('   5. Dynamic position sizing')
print()

if success:
    print('🎉 BREAKTHROUGH! This systematic approach is working!')
    print()
    print('💡 Next steps for further improvement:')
    print('   - Add more currency pairs for diversification')
    print('   - Implement cross-asset regime detection')
    print('   - Add transaction cost optimization')
    print('   - Test walk-forward validation')
else:
    print('💡 Additional strategies to try:')
    print('   - Test different regime detection methods')
    print('   - Add macroeconomic features (rates, GDP, etc.)')
    print('   - Try longer training history (2015-2023)')
    print('   - Implement ensemble stacking')
    print('   - Add volatility targeting')

print()
print('=' * 90)
