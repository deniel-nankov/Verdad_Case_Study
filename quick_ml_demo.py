#!/usr/bin/env python3
"""
Quick ML Demo - First 3 Steps (Fast Version)
Skips heavy LSTM training for speed
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('./ml_fx')

from ml_fx.data_loader import MLDataLoader
from ml_fx.feature_engineer import FeatureEngineer
from ml_fx.ml_models import MLEnsemble
from dotenv import load_dotenv
import os

load_dotenv()
print('✅ Setup complete!')
print()

# Step 1: Load Data
print('='*70)
print('STEP 1: LOAD & EXPLORE DATA')
print('='*70)
fred_key = os.getenv('FRED_API_KEY', 'b4a18aac3a462b6951ee89d9fef027cb')
loader = MLDataLoader(fred_api_key=fred_key)

# Use recent data only for speed (2020+)
data = loader.load_all_data(start_date='2020-01-01')

print(f'\n📊 Dataset Summary:')
print(f'   Shape: {data.shape}')
print(f'   Date range: {data.index[0]} to {data.index[-1]}')
print(f'   Columns: {len(data.columns)}')
print(f'\n   Sample (last 5 rows):')
print(data.tail())
print()

# Step 2: Feature Engineering
print('='*70)
print('STEP 2: FEATURE ENGINEERING')
print('='*70)
currencies = ['AUD', 'BRL', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'MXN']
fe = FeatureEngineer(currencies=currencies)
features = fe.create_all_features(data)

print(f'\n🔧 Feature Engineering Results:')
print(f'   Total features: {len(features.columns)}')
print(f'   Sample size: {len(features)} rows')

feature_groups = fe.get_feature_groups(features)
print(f'\n   Feature breakdown:')
for name, cols in feature_groups.items():
    print(f'     {name:20s}: {len(cols):3d} features')
print()

# Step 3: Train EUR Demo (FAST - RF and XGB only)
print('='*70)
print('STEP 3: TRAIN ML MODELS FOR EUR (FAST DEMO)')
print('='*70)
print('⚡ Using RF + XGB only (skipping slow LSTM)')
print()

ensemble = MLEnsemble(model_dir='./ml_models')

currency = 'EUR'
fx_col = f'{currency}_USD'

if fx_col in data.columns:
    # Create target: predict next 21 days return
    target = data[fx_col].pct_change(21).shift(-21)
    
    # Align features and target
    valid_idx = features.index.intersection(target.dropna().index)
    X = features.loc[valid_idx]
    y = target.loc[valid_idx]
    
    print(f'📊 Training data for {currency}:')
    print(f'   Features shape: {X.shape}')
    print(f'   Target shape: {y.shape}')
    print(f'   Date range: {X.index[0]} to {X.index[-1]}')
    print()
    
    print('🚀 Starting fast training... (1-2 minutes)')
    print()
    
    # Train only RF and XGB (skip LSTM)
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print('Training Random Forest...')
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
    print(f'✅ Random Forest R²: {rf_r2:.4f}')
    
    print('\nTraining XGBoost...')
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
    print(f'✅ XGBoost R²: {xgb_r2:.4f}')
    
    # Simple ensemble (average)
    ensemble_pred = (rf_pred + xgb_pred) / 2
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    print()
    print('='*70)
    print('📊 PERFORMANCE SUMMARY')
    print('='*70)
    print(f'   Random Forest R²:  {rf_r2:7.4f}')
    print(f'   XGBoost R²:        {xgb_r2:7.4f}')
    print(f'   Ensemble R²:       {ensemble_r2:7.4f}')
    print()
    
    # Feature importance
    print('🎯 Top 15 Most Important Features:')
    print()
    
    # Combine feature importances from RF and XGB
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
    importance = importance.sort_values('avg_importance', ascending=False).head(15)
    
    for idx, row in importance.iterrows():
        print(f'   {row.name+1:2d}. {row["feature"]:45s} {row["avg_importance"]:6.4f}')
    
    print()
    print('='*70)
    print('✅ FAST DEMO COMPLETE!')
    print('='*70)
    print()
    print('📝 Notes:')
    print('   - Used 2020+ data for speed')
    print('   - Skipped LSTM (too slow)')
    print('   - RF + XGB ensemble ready')
    print()
    print('💡 Next steps:')
    print('   - Full training: use train_ml_models_enhanced.py')
    print('   - Include LSTM: set skip_lstm=False')
    print('   - More data: change start_date to 2015-01-01')
    
else:
    print(f'❌ No data for {currency}')
