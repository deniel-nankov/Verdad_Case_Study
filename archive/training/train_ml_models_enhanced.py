"""
Enhanced ML Model Training for FX Carry Trading
Objective: Improve R² scores from EUR=0.09, CHF=0.04 to >0.15

Key Enhancements:
1. More features: Technical indicators, macro cycles, volatility regimes
2. Better feature selection: Remove noise, keep signal
3. Hyperparameter tuning: Grid search for optimal parameters
4. Ensemble optimization: Weighted averaging based on validation performance
5. Walk-forward validation: More realistic out-of-sample testing
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import yfinance as yf

print("="*70)
print("🚀 ENHANCED ML MODEL TRAINING")
print("="*70)

# ============================================================================
# STEP 1: ENHANCED FEATURE ENGINEERING
# ============================================================================

def create_enhanced_features(eur_prices, chf_prices, spy_prices):
    """
    Create comprehensive feature set with technical, macro, and regime indicators
    """
    print("\n📊 Creating enhanced features...")
    
    # Returns at multiple horizons
    features = pd.DataFrame(index=eur_prices.index)
    
    # EUR features
    for lag in [1, 5, 10, 21, 63]:
        features[f'eur_ret_{lag}d'] = eur_prices.pct_change(lag)
        features[f'eur_vol_{lag}d'] = eur_prices.pct_change().rolling(lag).std()
    
    # CHF features
    for lag in [1, 5, 10, 21, 63]:
        features[f'chf_ret_{lag}d'] = chf_prices.pct_change(lag)
        features[f'chf_vol_{lag}d'] = chf_prices.pct_change().rolling(lag).std()
    
    # Cross-asset features (SPY)
    for lag in [1, 5, 10, 21]:
        features[f'spy_ret_{lag}d'] = spy_prices.pct_change(lag)
    features['spy_vol_21d'] = spy_prices.pct_change().rolling(21).std()
    
    # Technical indicators
    # EUR
    features['eur_rsi_14'] = compute_rsi(eur_prices, 14)
    features['eur_ma_ratio_50'] = eur_prices / eur_prices.rolling(50).mean()
    features['eur_ma_ratio_200'] = eur_prices / eur_prices.rolling(200).mean()
    features['eur_bb_position'] = compute_bb_position(eur_prices, 20, 2)
    
    # CHF
    features['chf_rsi_14'] = compute_rsi(chf_prices, 14)
    features['chf_ma_ratio_50'] = chf_prices / chf_prices.rolling(50).mean()
    features['chf_ma_ratio_200'] = chf_prices / chf_prices.rolling(200).mean()
    features['chf_bb_position'] = compute_bb_position(chf_prices, 20, 2)
    
    # Regime indicators
    features['eur_vol_regime'] = features['eur_vol_21d'] / features['eur_vol_21d'].rolling(252).mean()
    features['chf_vol_regime'] = features['chf_vol_21d'] / features['chf_vol_21d'].rolling(252).mean()
    features['spy_vol_regime'] = features['spy_vol_21d'] / features['spy_vol_21d'].rolling(252).mean()
    
    # Simpler correlation features - just use returns directly
    eur_ret = eur_prices.pct_change()
    chf_ret = chf_prices.pct_change()
    spy_ret = spy_prices.pct_change()
    
    # Product of returns as proxy for correlation
    features['eur_chf_corr_proxy'] = (eur_ret * chf_ret).rolling(21).mean()
    features['eur_spy_corr_proxy'] = (eur_ret * spy_ret).rolling(21).mean()
    features['chf_spy_corr_proxy'] = (chf_ret * spy_ret).rolling(21).mean()
    
    # Momentum indicators
    features['eur_mom_3m_6m'] = (eur_prices.pct_change(63) - eur_prices.pct_change(126))
    features['chf_mom_3m_6m'] = (chf_prices.pct_change(63) - chf_prices.pct_change(126))
    
    # Add VIX proxy (SPY volatility)
    features['vix_proxy'] = spy_prices.pct_change().rolling(21).std() * np.sqrt(252)
    
    # Seasonality
    features['month'] = pd.to_datetime(features.index).month
    features['quarter'] = pd.to_datetime(features.index).quarter
    features['day_of_week'] = pd.to_datetime(features.index).dayofweek
    
    print(f"   ✅ Created {len(features.columns)} features")
    
    return features.dropna()

def compute_rsi(prices, period=14):
    """Compute RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bb_position(prices, period=20, num_std=2):
    """Compute position within Bollinger Bands (0=lower, 0.5=middle, 1=upper)"""
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    bb_position = (prices - lower) / (upper - lower)
    return bb_position.clip(0, 1)

# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================

print("\n📥 Loading market data...")

# Download extended history for better training
eur_usd = yf.download('EURUSD=X', start='2015-01-01', progress=False)['Close']
chf_usd = yf.download('CHFUSD=X', start='2015-01-01', progress=False)['Close']
spy = yf.download('SPY', start='2015-01-01', progress=False)['Close']

# Handle multi-level columns
if isinstance(spy, pd.DataFrame):
    if isinstance(spy.columns, pd.MultiIndex):
        spy = spy.iloc[:, 0]
    else:
        spy = spy.squeeze()

print(f"   EUR/USD: {len(eur_usd)} days")
print(f"   CHF/USD: {len(chf_usd)} days")
print(f"   SPY: {len(spy)} days")

# ============================================================================
# STEP 3: ENHANCED FEATURE ENGINEERING
# ============================================================================

features = create_enhanced_features(eur_usd, chf_usd, spy)

print(f"\n📊 Feature matrix: {features.shape}")
print(f"   Date range: {features.index[0].date()} to {features.index[-1].date()}")

# ============================================================================
# STEP 4: TRAIN EUR MODEL
# ============================================================================

print("\n" + "="*70)
print("🤖 TRAINING EUR MODEL")
print("="*70)

# Create target: predict 21-day forward return
eur_target = eur_usd.pct_change(21).shift(-21)

# Align features and target
common_idx = features.index.intersection(eur_target.dropna().index)
X_eur = features.loc[common_idx]
y_eur = eur_target.loc[common_idx]

print(f"\n📊 EUR training data: {X_eur.shape}")
print(f"   Target mean: {y_eur.mean():.4f}")
print(f"   Target std: {y_eur.std():.4f}")

# Feature selection - keep top 30 most predictive features
print("\n🔍 Selecting top features...")
selector = SelectKBest(f_regression, k=30)
X_eur_selected = selector.fit_transform(X_eur, y_eur)
selected_features = X_eur.columns[selector.get_support()].tolist()

print(f"   ✅ Selected {len(selected_features)} features")
print(f"   Top 10: {selected_features[:10]}")

# Time series split for walk-forward validation
tscv = TimeSeriesSplit(n_splits=5)

# Scale features
scaler = StandardScaler()
X_eur_scaled = scaler.fit_transform(X_eur_selected)

# Train ensemble with hyperparameter tuning
models = {}

# 1. Random Forest with tuning
print("\n🌲 Training Random Forest...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}
rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params,
    cv=tscv,
    scoring='r2',
    n_jobs=-1
)
rf.fit(X_eur_scaled, y_eur)
models['rf'] = rf.best_estimator_
print(f"   Best params: {rf.best_params_}")
print(f"   Best CV R²: {rf.best_score_:.4f}")

# 2. XGBoost with tuning
print("\n⚡ Training XGBoost...")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}
xgb_model = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    xgb_params,
    cv=tscv,
    scoring='r2',
    n_jobs=-1
)
xgb_model.fit(X_eur_scaled, y_eur)
models['xgb'] = xgb_model.best_estimator_
print(f"   Best params: {xgb_model.best_params_}")
print(f"   Best CV R²: {xgb_model.best_score_:.4f}")

# 3. Gradient Boosting
print("\n📈 Training Gradient Boosting...")
gb_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8, 1.0]
}
gb = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params,
    cv=tscv,
    scoring='r2',
    n_jobs=-1
)
gb.fit(X_eur_scaled, y_eur)
models['gb'] = gb.best_estimator_
print(f"   Best params: {gb.best_params_}")
print(f"   Best CV R²: {gb.best_score_:.4f}")

# 4. Ridge regression (linear baseline)
print("\n📊 Training Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_eur_scaled, y_eur)
models['ridge'] = ridge
ridge_score = ridge.score(X_eur_scaled, y_eur)
print(f"   R²: {ridge_score:.4f}")

# ============================================================================
# STEP 5: ENSEMBLE WITH WEIGHTED AVERAGING
# ============================================================================

print("\n🎯 Creating weighted ensemble...")

# Calculate out-of-sample predictions for weighting
weights = {}
for name, model in models.items():
    oos_preds = []
    oos_actual = []
    
    for train_idx, val_idx in tscv.split(X_eur_scaled):
        X_train, X_val = X_eur_scaled[train_idx], X_eur_scaled[val_idx]
        y_train, y_val = y_eur.iloc[train_idx], y_eur.iloc[val_idx]
        
        model.fit(X_train, y_train)
        oos_preds.extend(model.predict(X_val))
        oos_actual.extend(y_val)
    
    oos_r2 = np.corrcoef(oos_preds, oos_actual)[0, 1] ** 2
    weights[name] = max(0, oos_r2)  # Only positive weights

# Normalize weights
total_weight = sum(weights.values())
if total_weight > 0:
    weights = {k: v/total_weight for k, v in weights.items()}
else:
    weights = {k: 1/len(weights) for k in weights}

print(f"   Model weights: {weights}")

# Final ensemble prediction
eur_ensemble_pred = sum(
    weights[name] * model.predict(X_eur_scaled) 
    for name, model in models.items()
)

eur_ensemble_r2 = np.corrcoef(eur_ensemble_pred, y_eur)[0, 1] ** 2

print(f"\n✅ EUR Ensemble R²: {eur_ensemble_r2:.4f}")

# Save models
joblib.dump({
    'models': models,
    'weights': weights,
    'scaler': scaler,
    'selected_features': selected_features,
    'selector': selector
}, 'ml_models/eur_enhanced.pkl')

# ============================================================================
# STEP 6: TRAIN CHF MODEL (SAME PROCESS)
# ============================================================================

print("\n" + "="*70)
print("🤖 TRAINING CHF MODEL")
print("="*70)

chf_target = chf_usd.pct_change(21).shift(-21)
common_idx = features.index.intersection(chf_target.dropna().index)
X_chf = features.loc[common_idx]
y_chf = chf_target.loc[common_idx]

print(f"\n📊 CHF training data: {X_chf.shape}")

# Feature selection
selector_chf = SelectKBest(f_regression, k=30)
X_chf_selected = selector_chf.fit_transform(X_chf, y_chf)
selected_features_chf = X_chf.columns[selector_chf.get_support()].tolist()

scaler_chf = StandardScaler()
X_chf_scaled = scaler_chf.fit_transform(X_chf_selected)

# Train models (same as EUR)
models_chf = {}

print("\n🌲 Training Random Forest...")
rf_chf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=tscv, scoring='r2', n_jobs=-1)
rf_chf.fit(X_chf_scaled, y_chf)
models_chf['rf'] = rf_chf.best_estimator_
print(f"   Best CV R²: {rf_chf.best_score_:.4f}")

print("\n⚡ Training XGBoost...")
xgb_chf = GridSearchCV(xgb.XGBRegressor(random_state=42), xgb_params, cv=tscv, scoring='r2', n_jobs=-1)
xgb_chf.fit(X_chf_scaled, y_chf)
models_chf['xgb'] = xgb_chf.best_estimator_
print(f"   Best CV R²: {xgb_chf.best_score_:.4f}")

print("\n📈 Training Gradient Boosting...")
gb_chf = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=tscv, scoring='r2', n_jobs=-1)
gb_chf.fit(X_chf_scaled, y_chf)
models_chf['gb'] = gb_chf.best_estimator_
print(f"   Best CV R²: {gb_chf.best_score_:.4f}")

ridge_chf = Ridge(alpha=1.0)
ridge_chf.fit(X_chf_scaled, y_chf)
models_chf['ridge'] = ridge_chf

# Weighted ensemble
weights_chf = {}
for name, model in models_chf.items():
    oos_preds = []
    oos_actual = []
    for train_idx, val_idx in tscv.split(X_chf_scaled):
        X_train, X_val = X_chf_scaled[train_idx], X_chf_scaled[val_idx]
        y_train, y_val = y_chf.iloc[train_idx], y_chf.iloc[val_idx]
        model.fit(X_train, y_train)
        oos_preds.extend(model.predict(X_val))
        oos_actual.extend(y_val)
    oos_r2 = np.corrcoef(oos_preds, oos_actual)[0, 1] ** 2
    weights_chf[name] = max(0, oos_r2)

total_weight_chf = sum(weights_chf.values())
if total_weight_chf > 0:
    weights_chf = {k: v/total_weight_chf for k, v in weights_chf.items()}
else:
    weights_chf = {k: 1/len(weights_chf) for k in weights_chf}

chf_ensemble_pred = sum(weights_chf[name] * model.predict(X_chf_scaled) for name, model in models_chf.items())
chf_ensemble_r2 = np.corrcoef(chf_ensemble_pred, y_chf)[0, 1] ** 2

print(f"\n✅ CHF Ensemble R²: {chf_ensemble_r2:.4f}")

# Save CHF models
joblib.dump({
    'models': models_chf,
    'weights': weights_chf,
    'scaler': scaler_chf,
    'selected_features': selected_features_chf,
    'selector': selector_chf
}, 'ml_models/chf_enhanced.pkl')

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ ENHANCED MODEL TRAINING COMPLETE")
print("="*70)

print(f"\n📊 Performance Summary:")
print(f"   EUR Ensemble R²: {eur_ensemble_r2:.4f} (Previous: 0.0905)")
print(f"   CHF Ensemble R²: {chf_ensemble_r2:.4f} (Previous: 0.0369)")

improvement_eur = ((eur_ensemble_r2 - 0.0905) / 0.0905) * 100 if eur_ensemble_r2 > 0 else -100
improvement_chf = ((chf_ensemble_r2 - 0.0369) / 0.0369) * 100 if chf_ensemble_r2 > 0 else -100

print(f"\n📈 Improvement:")
print(f"   EUR: {improvement_eur:+.1f}%")
print(f"   CHF: {improvement_chf:+.1f}%")

print(f"\n💾 Models saved:")
print(f"   ml_models/eur_enhanced.pkl")
print(f"   ml_models/chf_enhanced.pkl")

print(f"\n🎯 Kelly Recommendation:")
if eur_ensemble_r2 > 0.15 and chf_ensemble_r2 > 0.15:
    print("   ✅ R² scores strong enough for Kelly optimization")
    eur_weight = eur_ensemble_r2 / (eur_ensemble_r2 + chf_ensemble_r2)
    chf_weight = chf_ensemble_r2 / (eur_ensemble_r2 + chf_ensemble_r2)
    print(f"   Suggested allocation: EUR {eur_weight:.1%}, CHF {chf_weight:.1%}")
elif eur_ensemble_r2 > 0.10 or chf_ensemble_r2 > 0.10:
    print("   ⚠️  R² scores moderate - use conservative Kelly or equal weight")
    print("   Recommended: 50/50 equal weight")
else:
    print("   ❌ R² scores too low for Kelly - stick to equal weight")
    print("   Recommended: 50/50 equal weight")
