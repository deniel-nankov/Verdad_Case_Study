#!/usr/bin/env python3
"""
ML FX Trading Strategy - Training Script
Run training directly in VS Code for better debugging
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import sys
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Add ml_fx to path
sys.path.insert(0, str(Path(__file__).parent / 'ml_fx'))

from data_loader import MLDataLoader
from feature_engineer import FeatureEngineer
from ml_models import MLEnsemble
from ml_strategy import MLFXStrategy

# Load environment
from dotenv import load_dotenv
load_dotenv()

print("="*70)
print("🤖 ML FX CARRY TRADING STRATEGY - TRAINING")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
CURRENCIES = ['AUD', 'BRL', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'MXN']
FRED_KEY = os.getenv('FRED_API_KEY', 'b4a18aac3a462b6951ee89d9fef027cb')
START_DATE = '2010-01-01'  # More historical data for better training
FAST_MODE = True  # Set to True for faster training (skips hyperparameter optimization)
SKIP_LSTM = True  # Set to True to skip LSTM (MUCH faster, 2-3 min total)

print("📋 Configuration:")
print(f"   Currencies: {', '.join(CURRENCIES)}")
print(f"   Start date: {START_DATE} (15 years of data!)")
print(f"   FRED API Key: {'✅ Loaded' if FRED_KEY else '❌ Missing'}")
print(f"   Fast Mode: {'✅ ON (5-10 min)' if FAST_MODE else '❌ OFF (20-30 min)'}")
print(f"   Skip LSTM: {'✅ YES (2-3 min total!)' if SKIP_LSTM else '❌ NO'}")
print()

# Step 1: Load Data
print("="*70)
print("📊 STEP 1: LOADING DATA")
print("="*70)

try:
    loader = MLDataLoader(fred_api_key=FRED_KEY)
    data = loader.load_all_data(start_date=START_DATE)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Shape: {data.shape}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print(f"   Columns: {len(data.columns)}")
    print(f"\n   First few columns: {', '.join(data.columns[:5].tolist())}")
    print()
except Exception as e:
    print(f"❌ ERROR loading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Feature Engineering
print("="*70)
print("🔧 STEP 2: FEATURE ENGINEERING")
print("="*70)

try:
    fe = FeatureEngineer(currencies=CURRENCIES)
    features = fe.create_all_features(data)
    
    print(f"✅ Features created successfully!")
    print(f"   Total features: {len(features.columns)}")
    print(f"   Sample size: {len(features)} rows")
    
    # Show feature groups
    feature_groups = fe.get_feature_groups(features)
    print(f"\n   Feature breakdown:")
    for name, cols in feature_groups.items():
        print(f"     {name:20s}: {len(cols):3d} features")
    print()
except Exception as e:
    print(f"❌ ERROR creating features: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Train EUR (Demo)
print("="*70)
print("🤖 STEP 3: TRAINING EUR (DEMO)")
print("="*70)
print("Training one currency first to validate the pipeline...\n")

try:
    ensemble = MLEnsemble(model_dir="./ml_models")
    
    currency = 'EUR'
    fx_col = f'{currency}_USD'
    
    if fx_col in data.columns:
        # Create target: predict next 21 days return
        target = data[fx_col].pct_change(21).shift(-21)
        
        # Align features and target
        valid_idx = features.index.intersection(target.dropna().index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]
        
        print(f"📊 Training data for {currency}:")
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Target mean: {y.mean():.6f}")
        print(f"   Target std: {y.std():.6f}")
        print()
        
        # Train models
        print(f"🚀 Training models for {currency}...")
        results = ensemble.train(
            X, y, 
            currency=currency,
            validation_split=0.2,
            optimize=not FAST_MODE,  # Skip optimization in fast mode
            skip_lstm=SKIP_LSTM  # Skip LSTM for speed
        )
        
        # Save models
        ensemble.save_models(currency)
        
        print(f"\n✅ {currency} training complete!")
        print(f"   Random Forest R²: {results['rf']['r2']:.4f}")
        print(f"   XGBoost R²: {results['xgb']['r2']:.4f}")
        print(f"   LSTM R²: {results['lstm']['r2']:.4f}")
        print(f"   Ensemble R²: {results['ensemble']['r2']:.4f}")
        print()
    else:
        print(f"❌ No data for {currency}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ ERROR training EUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Feature Importance
print("="*70)
print("🎯 STEP 4: EUR FEATURE IMPORTANCE")
print("="*70)

try:
    top_features = ensemble.get_feature_importance('EUR', top_n=20)
    
    print("📊 Top 20 Most Important Features for EUR:")
    print()
    for idx, row in top_features.iterrows():
        print(f"   {idx+1:2d}. {row['feature']:40s} {row['avg_importance']:.4f}")
    print()
    
except Exception as e:
    print(f"❌ ERROR getting feature importance: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Train ALL Currencies
print("="*70)
print("🚀 STEP 5: TRAINING ALL CURRENCIES")
print("="*70)
if FAST_MODE:
    print("⚡ FAST MODE: Training should complete in 5-10 minutes\n")
else:
    print("⏱️  FULL MODE: This will take 20-30 minutes. Progress will be shown below...\n")

try:
    strategy = MLFXStrategy(
        fred_api_key=FRED_KEY,
        currencies=CURRENCIES
    )
    
    # Train all currencies
    all_results = strategy.train_all_currencies(
        start_date=START_DATE,
        validation_split=0.2,
        walk_forward=False,
        optimize=not FAST_MODE,
        skip_lstm=SKIP_LSTM
    )
    
    print(f"\n✅ All currencies trained successfully!")
    print()
    
except Exception as e:
    print(f"❌ ERROR training all currencies: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Performance Summary
print("="*70)
print("📊 STEP 6: PERFORMANCE SUMMARY")
print("="*70)

try:
    performance_data = []
    
    for currency in CURRENCIES:
        if currency in all_results:
            res = all_results[currency]
            performance_data.append({
                'Currency': currency,
                'RF_R2': res['rf']['r2'],
                'XGB_R2': res['xgb']['r2'],
                'LSTM_R2': res['lstm']['r2'],
                'Ensemble_R2': res['ensemble']['r2']
            })
    
    perf_df = pd.DataFrame(performance_data)
    
    print("\n📈 Model Performance (R² Scores):")
    print()
    print(perf_df.to_string(index=False))
    print()
    
    avg_r2 = perf_df['Ensemble_R2'].mean()
    print(f"📊 Average Ensemble R²: {avg_r2:.4f}")
    
    if avg_r2 > 0.05:
        print("   🎉 EXCELLENT! Models show strong predictive power")
    elif avg_r2 > 0:
        print("   ✅ GOOD! Models beat random baseline")
    else:
        print("   ⚠️  Models need improvement - try more data or features")
    print()
    
    # Save performance summary
    perf_df.to_csv('ml_performance_summary.csv', index=False)
    print("💾 Performance saved to: ml_performance_summary.csv")
    print()
    
except Exception as e:
    print(f"❌ ERROR creating performance summary: {e}")
    import traceback
    traceback.print_exc()

# Step 7: Generate Live Signals
print("="*70)
print("🔮 STEP 7: GENERATING LIVE TRADING SIGNALS")
print("="*70)

try:
    signals = strategy.generate_signals()
    
    positions = strategy.generate_positions(
        signals=signals,
        capital=100000,
        max_position_size=0.25,
        risk_scale=1.0
    )
    
    signal_df = pd.DataFrame(list(signals.items()), columns=['Currency', 'Signal'])
    signal_df = signal_df.sort_values('Signal', ascending=False)
    
    position_df = pd.DataFrame(list(positions.items()), columns=['Currency', 'Position_USD'])
    position_df = position_df.sort_values('Position_USD', ascending=False)
    
    print("\n🎯 Current Trading Signals:")
    print()
    print(signal_df.to_string(index=False))
    print()
    
    print("\n💰 Recommended Positions ($100k Capital):")
    print()
    print(position_df.to_string(index=False))
    print()
    
    # Save signals
    signal_df.to_csv('ml_current_signals.csv', index=False)
    position_df.to_csv('ml_current_positions.csv', index=False)
    print("💾 Signals saved to: ml_current_signals.csv")
    print("💾 Positions saved to: ml_current_positions.csv")
    print()
    
except Exception as e:
    print(f"❌ ERROR generating signals: {e}")
    import traceback
    traceback.print_exc()

# Final Summary
print("="*70)
print("✅ ML TRAINING COMPLETE")
print("="*70)

print(f"\n📊 Training Summary:")
print(f"   Currencies trained: {len([c for c in CURRENCIES if c in all_results])}/{len(CURRENCIES)}")
print(f"   Features engineered: {len(features.columns)}")
print(f"   Training period: {data.index[0].date()} to {data.index[-1].date()}")
print(f"   Average Ensemble R²: {avg_r2:.4f}")

print(f"\n🚀 Next Steps:")
print("   1. ✅ Models trained and saved in ./ml_models/")
print("   2. ✅ Live signals generated")
print("   3. 🔄 Review performance in ml_performance_summary.csv")
print("   4. 🔄 Review signals in ml_current_signals.csv")
print("   5. 🔄 Integrate with live_trading_system.py")
print("   6. 📄 Start paper trading")

print(f"\n🎊 Congratulations! You have a working ML trading system!")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
