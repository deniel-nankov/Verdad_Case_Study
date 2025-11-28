#!/usr/bin/env python3
"""
Backtest Script for Enhanced ML Strategy
========================================

Runs a historical backtest of the Enhanced ML Strategy (ML + Cross-Asset + Intraday + Kelly)
from 2020 to present.

Components:
1. ML Ensemble (RF, XGB, LSTM)
2. Cross-Asset Spillovers (Equity/Commodity Momentum)
3. Intraday Microstructure (Session Timing)
4. Adaptive Leverage (Kelly Criterion)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml_fx'))

from ml_fx.ml_strategy import MLFXStrategy
from ml_fx.data_loader import MLDataLoader
from cross_asset_spillovers import CrossAssetSpilloverStrategy
from intraday_microstructure import IntradayMicrostructureStrategy
from adaptive_leverage import AdaptiveLeverageOptimizer

# Configuration
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
INITIAL_CAPITAL = 100000
CURRENCIES = ['AUD', 'BRL', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'MXN']
RESULTS_DIR = Path('results/backtests')
CHARTS_DIR = Path('results/charts')

def run_backtest():
    print("="*70)
    print("🚀 ENHANCED ML STRATEGY BACKTEST")
    print("="*70)
    
    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Components
    print("\n📦 Initializing components...")
    
    from dotenv import load_dotenv
    load_dotenv()
    fred_key = os.getenv('FRED_API_KEY', 'b4a18aac3a462b6951ee89d9fef027cb')
    
    # ML Strategy
    ml_strategy = MLFXStrategy(
        fred_api_key=fred_key,
        currencies=CURRENCIES
    )
    try:
        ml_strategy.load_trained_models()
    except Exception as e:
        print(f"⚠️  Warning: Could not load some models: {e}")
    
    # Other components
    cross_asset = CrossAssetSpilloverStrategy()
    intraday = IntradayMicrostructureStrategy()
    kelly = AdaptiveLeverageOptimizer()
    
    # 2. Load Data
    print("\n📥 Loading historical data...")
    
    # Load ML data (features)
    # We use the data loader to get the full dataset with features
    full_data = ml_strategy.data_loader.load_all_data(START_DATE, END_DATE)
    features = ml_strategy.feature_engineer.create_all_features(full_data)
    
    # Load Cross-Asset data
    ca_data = cross_asset.download_cross_asset_data(START_DATE, END_DATE)
    ca_momentum = cross_asset.calculate_momentum_signals(ca_data)
    
    # Align dates
    common_dates = features.index.intersection(ca_momentum.index)
    features = features.loc[common_dates]
    ca_momentum = ca_momentum.loc[common_dates]
    
    print(f"✅ Data aligned: {len(common_dates)} trading days")
    
    # 3. Backtest Loop
    print("\n🔄 Running backtest loop...")
    
    equity = [INITIAL_CAPITAL]
    dates = [common_dates[0]]
    positions = {curr: 0.0 for curr in CURRENCIES}
    
    history = []
    
    for i in range(1, len(common_dates)):
        current_date = common_dates[i]
        prev_date = common_dates[i-1]
        
        # Calculate PnL from previous positions
        daily_pnl = 0
        for curr, pos in positions.items():
            # Get FX return for this currency
            # Note: features dataframe should have returns or prices
            # We'll use the target column if available, or calculate from raw data
            # Assuming 'EUR_USD' column exists in full_data
            
            fx_col = f'{curr}_USD'
            if fx_col in full_data.columns:
                try:
                    # Calculate return: (Price_t - Price_t-1) / Price_t-1
                    curr_price = full_data.loc[current_date, fx_col]
                    prev_price = full_data.loc[prev_date, fx_col]
                    
                    if not pd.isna(curr_price) and not pd.isna(prev_price) and prev_price != 0:
                        fx_ret = (curr_price - prev_price) / prev_price
                        
                        # Add carry (approximate)
                        # Interest rate diff / 252
                        rate_col = f'{curr}_rate'
                        usd_rate_col = 'USD_rate'
                        
                        if rate_col in full_data.columns and usd_rate_col in full_data.columns:
                            foreign_rate = full_data.loc[prev_date, rate_col]
                            usd_rate = full_data.loc[prev_date, usd_rate_col]
                            carry = (foreign_rate - usd_rate) / 100 / 252
                        else:
                            carry = 0
                            
                        total_ret = fx_ret + carry
                        daily_pnl += pos * total_ret
                except KeyError:
                    pass
        
        # Update equity
        new_equity = equity[-1] + daily_pnl
        equity.append(new_equity)
        dates.append(current_date)
        
        # Generate NEW signals for next day
        # -------------------------------
        
        # 1. ML Signals
        ml_signals = {}
        current_feats = features.loc[[current_date]]
        
        for curr in CURRENCIES:
            try:
                pred = ml_strategy.ensemble.predict(curr, current_feats)
                ml_signals[curr] = np.tanh(pred[0] * 10)
            except:
                ml_signals[curr] = 0.0
        
        # 2. Cross-Asset Signals
        # We need to pass a dataframe with just the current row to match the API
        # But generate_fx_signals expects the full momentum df or at least the columns
        # We can just pass the current row as a Series/DataFrame
        current_mom = ca_momentum.loc[[current_date]]
        ca_signals = cross_asset.generate_fx_signals(current_mom, CURRENCIES)
        
        # 3. Intraday Timing (Approximation for daily backtest)
        # Since we are trading daily, we can't fully capture intraday alpha
        # But we can apply a "timing quality" filter based on the day's volatility/volume
        # For this backtest, we'll assume neutral timing or use a simplified logic
        # e.g. if VIX is high, timing is harder -> reduce confidence
        
        timing_adjustments = {}
        for curr in CURRENCIES:
            # Simplified: Use VIX as proxy for market stability
            # If VIX is high, reduce signal confidence
            vix = full_data.loc[current_date, 'VIX'] if 'VIX' in full_data.columns else 20.0
            confidence = 1.0 if vix < 25 else 0.5
            timing_adjustments[curr] = confidence
            
        # 4. Combine Signals
        combined_signals = {}
        for curr in CURRENCIES:
            ml = ml_signals.get(curr, 0.0)
            ca = ca_signals.get(curr, 0.0)
            conf = timing_adjustments.get(curr, 1.0)
            
            # Weighted average: 70% ML, 30% Cross-Asset
            raw_combined = ml * 0.7 + ca * 0.3
            
            # Apply confidence scaling
            combined_signals[curr] = np.clip(raw_combined * conf, -1.0, 1.0)
            
        # 5. Kelly Position Sizing
        # We need to pass the signals to the optimizer
        # The optimizer expects a dictionary of signals
        
        # Note: optimize_positions prints output, we might want to suppress it or just use the logic directly
        # But let's use the method to be consistent
        
        # Redirect stdout to suppress print spam
        import io
        from contextlib import redirect_stdout
        
        with redirect_stdout(io.StringIO()):
            new_positions = kelly.optimize_positions(
                signals=combined_signals,
                capital=new_equity,
                currencies=CURRENCIES,
                max_position_pct=0.30,
                safety_factor=0.5
            )
            
        positions = new_positions
        
        # Record history
        history.append({
            'Date': current_date,
            'Equity': new_equity,
            'PnL': daily_pnl
        })
        
        if i % 100 == 0:
            print(f"   📅 {current_date.date()}: Equity ${new_equity:,.0f}")

    # 4. Analysis
    print("\n📊 Analyzing results...")
    
    equity_df = pd.DataFrame({'Date': dates, 'Equity': equity}).set_index('Date')
    
    # Calculate metrics
    returns = equity_df['Equity'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    annual_ret = (equity_df['Equity'].iloc[-1] / INITIAL_CAPITAL) ** (252 / len(returns)) - 1
    max_dd = ((equity_df['Equity'] / equity_df['Equity'].cummax()) - 1).min()
    
    print(f"\n📈 Performance Summary:")
    print(f"   Sharpe Ratio:    {sharpe:.3f}")
    print(f"   Annual Return:   {annual_ret*100:.2f}%")
    print(f"   Max Drawdown:    {max_dd*100:.2f}%")
    print(f"   Final Equity:    ${equity_df['Equity'].iloc[-1]:,.0f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'enhanced_strategy_results_{timestamp}.csv'
    equity_df.to_csv(results_file)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(equity_df.index, equity_df['Equity'], label='Enhanced Strategy')
    plt.title('Enhanced ML Strategy Performance (2020-Present)', fontsize=16)
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add stats box
    stats_text = (
        f"Sharpe: {sharpe:.2f}\n"
        f"Ann. Ret: {annual_ret:.1%}\n"
        f"Max DD: {max_dd:.1%}"
    )
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8), fontsize=12, verticalalignment='top')
    
    chart_file = CHARTS_DIR / f'enhanced_strategy_performance_{timestamp}.png'
    plt.savefig(chart_file)
    print(f"💾 Chart saved to: {chart_file}")
    
    print("\n✅ Backtest Complete!")

if __name__ == "__main__":
    run_backtest()
