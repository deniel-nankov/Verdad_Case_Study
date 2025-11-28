#!/usr/bin/env python3
"""
Complete ML Trading System - Train All 8 Currencies
1. Train RF + XGB for all currencies (fast, no LSTM)
2. Generate feature importance charts
3. Generate live trading signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('./ml_fx')

from ml_fx.data_loader import MLDataLoader
from ml_fx.feature_engineer import FeatureEngineer
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from datetime import datetime

load_dotenv()

# Setup
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

print('='*80)
print('🚀 COMPLETE ML TRADING SYSTEM - TRAINING ALL 8 CURRENCIES')
print('='*80)
print()

# Step 1: Load Data
print('📊 STEP 1: Loading Data...')
print('-'*80)
fred_key = os.getenv('FRED_API_KEY', 'b4a18aac3a462b6951ee89d9fef027cb')
loader = MLDataLoader(fred_api_key=fred_key)
data = loader.load_all_data(start_date='2020-01-01')

print(f'✅ Loaded {len(data)} rows from {data.index[0].date()} to {data.index[-1].date()}')
print()

# Step 2: Feature Engineering
print('🔧 STEP 2: Engineering Features...')
print('-'*80)
currencies = ['AUD', 'BRL', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'MXN']
fe = FeatureEngineer(currencies=currencies)
features = fe.create_all_features(data)

print(f'✅ Created {len(features.columns)} features from {len(features)} samples')
print()

# Step 3: Train All Currencies
print('🤖 STEP 3: Training Models for All Currencies...')
print('-'*80)
print('⚡ Using RF + XGB (skipping slow LSTM for speed)')
print()

# Storage for results
all_results = {}
all_models = {}
all_feature_importance = {}

# Create ml_models directory if it doesn't exist
os.makedirs('./ml_models', exist_ok=True)

for i, currency in enumerate(currencies, 1):
    print(f'[{i}/{len(currencies)}] Training {currency}... ', end='', flush=True)
    
    fx_col = f'{currency}_USD'
    
    if fx_col not in data.columns:
        print(f'❌ No data')
        continue
    
    # Create target: predict next 21 days return
    target = data[fx_col].pct_change(21).shift(-21)
    
    # Align features and target
    valid_idx = features.index.intersection(target.dropna().index)
    X = features.loc[valid_idx]
    y = target.loc[valid_idx]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    
    # Train XGBoost
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_pred)
    
    # Ensemble
    ensemble_pred = (rf_pred + xgb_pred) / 2
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    # Store results
    all_results[currency] = {
        'rf_r2': rf_r2,
        'xgb_r2': xgb_r2,
        'ensemble_r2': ensemble_r2,
        'test_samples': len(y_test)
    }
    
    # Store models
    all_models[currency] = {
        'rf': rf,
        'xgb': xgb
    }
    
    # Feature importance
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf.feature_importances_
    })
    
    xgb_importance = pd.DataFrame({
        'feature': X.columns,
        'xgb_importance': xgb.feature_importances_
    })
    
    importance = rf_importance.merge(xgb_importance, on='feature')
    importance['avg_importance'] = (importance['rf_importance'] + importance['xgb_importance']) / 2
    importance = importance.sort_values('avg_importance', ascending=False)
    
    all_feature_importance[currency] = importance
    
    # Save models
    with open(f'./ml_models/{currency}_rf.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open(f'./ml_models/{currency}_xgb.pkl', 'wb') as f:
        pickle.dump(xgb, f)
    
    print(f'✅ R²: {ensemble_r2:.3f} (RF: {rf_r2:.3f}, XGB: {xgb_r2:.3f})')

print()
print('='*80)
print('📊 TRAINING COMPLETE - PERFORMANCE SUMMARY')
print('='*80)
print()

# Create performance dataframe
perf_df = pd.DataFrame([
    {
        'Currency': curr,
        'RF_R2': res['rf_r2'],
        'XGB_R2': res['xgb_r2'],
        'Ensemble_R2': res['ensemble_r2'],
        'Test_Samples': res['test_samples']
    }
    for curr, res in all_results.items()
])

print(perf_df.to_string(index=False))
print()
print(f'Average Ensemble R²: {perf_df["Ensemble_R2"].mean():.4f}')
print()

# Save performance summary
perf_df.to_csv('ml_performance_summary.csv', index=False)
print('✅ Saved: ml_performance_summary.csv')
print()

# TASK 1: Performance Chart
print('='*80)
print('📈 TASK 1: CREATING PERFORMANCE CHARTS')
print('='*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: Model comparison
x = np.arange(len(perf_df))
width = 0.25

ax1.bar(x - width, perf_df['RF_R2'], width, label='Random Forest', alpha=0.8, color='forestgreen')
ax1.bar(x, perf_df['XGB_R2'], width, label='XGBoost', alpha=0.8, color='steelblue')
ax1.bar(x + width, perf_df['Ensemble_R2'], width, label='Ensemble', alpha=0.8, color='gold')

ax1.set_xlabel('Currency', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison (R² Score)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(perf_df['Currency'])
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1.axhline(y=0.1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good threshold')

# Chart 2: Sorted by performance
sorted_perf = perf_df.sort_values('Ensemble_R2', ascending=True)
colors = ['red' if x < 0 else 'green' if x > 0.1 else 'orange' for x in sorted_perf['Ensemble_R2']]

ax2.barh(sorted_perf['Currency'], sorted_perf['Ensemble_R2'], color=colors, alpha=0.7)
ax2.set_xlabel('Ensemble R² Score', fontsize=12, fontweight='bold')
ax2.set_title('Currencies Ranked by Predictive Power', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=0.1, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('ml_model_performance.png', dpi=150, bbox_inches='tight')
print('✅ Saved: ml_model_performance.png')
plt.close()
print()

# TASK 2: Feature Importance Charts
print('='*80)
print('🎯 TASK 2: CREATING FEATURE IMPORTANCE CHARTS')
print('='*80)

# Top features across all currencies
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, currency in enumerate(currencies):
    if currency in all_feature_importance:
        top_features = all_feature_importance[currency].head(10)
        
        axes[idx].barh(range(len(top_features)), top_features['avg_importance'][::-1], color='steelblue', alpha=0.7)
        axes[idx].set_yticks(range(len(top_features)))
        axes[idx].set_yticklabels(top_features['feature'].iloc[::-1], fontsize=8)
        axes[idx].set_xlabel('Importance', fontsize=9)
        axes[idx].set_title(f'{currency} - Top 10 Features', fontsize=11, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('ml_feature_importance_all.png', dpi=150, bbox_inches='tight')
print('✅ Saved: ml_feature_importance_all.png')
plt.close()

# Print top features for each currency
print()
for currency in currencies:
    if currency in all_feature_importance:
        print(f'{currency} - Top 5 Features:')
        top_5 = all_feature_importance[currency].head(5)
        for i, row in top_5.iterrows():
            print(f'  {i+1}. {row["feature"]:45s} {row["avg_importance"]:.4f}')
        print()

# TASK 3: Generate Live Trading Signals
print('='*80)
print('🔮 TASK 3: GENERATING LIVE TRADING SIGNALS')
print('='*80)
print()

# Get latest features
latest_features = features.iloc[-1:]

signals = {}
predictions = {}

for currency in currencies:
    if currency in all_models:
        # Get models
        rf = all_models[currency]['rf']
        xgb = all_models[currency]['xgb']
        
        # Make predictions
        rf_pred = rf.predict(latest_features)[0]
        xgb_pred = xgb.predict(latest_features)[0]
        ensemble_pred = (rf_pred + xgb_pred) / 2
        
        # Convert to signal (-1 to +1)
        # Normalize by typical return volatility (assume ~2% for 21-day)
        signal = np.tanh(ensemble_pred / 0.02)
        
        signals[currency] = signal
        predictions[currency] = ensemble_pred

# Create signals dataframe
signal_df = pd.DataFrame({
    'Currency': list(signals.keys()),
    'Signal': list(signals.values()),
    'Predicted_Return': [predictions[c] * 100 for c in signals.keys()],
    'R2_Score': [all_results[c]['ensemble_r2'] for c in signals.keys()]
})

signal_df = signal_df.sort_values('Signal', ascending=False)

print('📊 Live Trading Signals:')
print('-'*80)
print(signal_df.to_string(index=False))
print()

# Generate position recommendations
capital = 100000
max_position_size = 0.25  # 25% max per position
risk_scale = 1.0

positions = {}
total_exposure = 0

# Scale positions by signal strength and R2 confidence
for _, row in signal_df.iterrows():
    currency = row['Currency']
    signal = row['Signal']
    r2 = row['R2_Score']
    
    # Weight by both signal strength and model confidence
    confidence = max(0, r2)  # Only use positive R2
    position_weight = signal * confidence * risk_scale
    
    # Cap at max position size
    position_value = np.clip(position_weight * capital, -max_position_size * capital, max_position_size * capital)
    
    positions[currency] = position_value
    total_exposure += abs(position_value)

# Create position dataframe
position_df = pd.DataFrame({
    'Currency': list(positions.keys()),
    'Position_USD': list(positions.values()),
    'Position_Pct': [p / capital * 100 for p in positions.values()],
    'Signal': [signals[c] for c in positions.keys()],
    'Expected_Return': [predictions[c] * 100 for c in positions.keys()]
})

position_df = position_df.sort_values('Position_USD', ascending=False)

print('💰 Recommended Positions (Capital: $100k):')
print('-'*80)
print(position_df.to_string(index=False))
print()
print(f'Total Exposure: ${total_exposure:,.0f} ({total_exposure/capital*100:.1f}%)')
print()

# Save signals
signal_df.to_csv('ml_live_signals.csv', index=False)
position_df.to_csv('ml_recommended_positions.csv', index=False)
print('✅ Saved: ml_live_signals.csv')
print('✅ Saved: ml_recommended_positions.csv')
print()

# TASK 3 (cont): Visualize Signals
print('📊 Creating signal visualization...')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: Signals
colors = ['green' if x > 0 else 'red' for x in signal_df['Signal']]
ax1.barh(signal_df['Currency'], signal_df['Signal'], color=colors, alpha=0.7)
ax1.set_xlabel('Signal Strength (-1 to +1)', fontsize=12, fontweight='bold')
ax1.set_title('ML Trading Signals (Current)', fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# Add predicted returns as text
for i, row in enumerate(signal_df.itertuples()):
    x_pos = row.Signal * 1.1
    ax1.text(x_pos, i, f'{row.Predicted_Return:+.1f}%', 
             ha='left' if row.Signal > 0 else 'right', va='center', fontsize=9)

# Chart 2: Positions
colors = ['green' if x > 0 else 'red' for x in position_df['Position_USD']]
ax2.barh(position_df['Currency'], position_df['Position_USD'], color=colors, alpha=0.7)
ax2.set_xlabel('Position Size (USD)', fontsize=12, fontweight='bold')
ax2.set_title('Recommended Positions ($100k Capital)', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

# Add percentages as text
for i, row in enumerate(position_df.itertuples()):
    x_pos = row.Position_USD * 1.1
    ax2.text(x_pos, i, f'{row.Position_Pct:+.1f}%', 
             ha='left' if row.Position_USD > 0 else 'right', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('ml_live_signals.png', dpi=150, bbox_inches='tight')
print('✅ Saved: ml_live_signals.png')
plt.close()
print()

# Final Summary
print('='*80)
print('✅ COMPLETE ML TRADING SYSTEM READY!')
print('='*80)
print()
print(f'📊 Summary:')
print(f'   • Currencies trained: {len(all_results)}/8')
print(f'   • Average R² score: {perf_df["Ensemble_R2"].mean():.4f}')
print(f'   • Best performer: {perf_df.loc[perf_df["Ensemble_R2"].idxmax(), "Currency"]} (R²: {perf_df["Ensemble_R2"].max():.4f})')
print(f'   • Training data: {data.index[0].date()} to {data.index[-1].date()}')
print(f'   • Features: {len(features.columns)}')
print()
print(f'📁 Files created:')
print(f'   • ml_performance_summary.csv - Model performance metrics')
print(f'   • ml_model_performance.png - Performance comparison chart')
print(f'   • ml_feature_importance_all.png - Feature importance for all currencies')
print(f'   • ml_live_signals.csv - Current trading signals')
print(f'   • ml_recommended_positions.csv - Position recommendations')
print(f'   • ml_live_signals.png - Signal visualization')
print(f'   • ./ml_models/ - Trained models for all currencies')
print()
print(f'🚀 Next Steps:')
print(f'   1. Review signals and positions above')
print(f'   2. Backtest these signals on historical data')
print(f'   3. Integrate with live trading system')
print(f'   4. Start paper trading to validate')
print()
print(f'🎊 You now have a complete ML trading system!')
print('='*80)
