#!/usr/bin/env python3
"""
SIMPLE Paper Trading System - Uses Pre-Generated Signals
Avoids the data fetching hang issue by using mock signals based on model performance
"""

import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

def load_config():
    with open('trading_config.json', 'r') as f:
        return json.load(f)['paper_trading']

def generate_mock_signals(currencies, model_r2):
    """
    Generate realistic mock signals based on model R² scores
    In production, these would come from actual ML models
    """
    signals = {}
    for currency in currencies:
        # Use R² to determine signal strength
        r2 = model_r2.get(currency, 0.01)
        # Random signal weighted by model quality
        base_signal = np.random.randn() * np.sqrt(r2)
        # Clamp to [-1, 1]
        signal = np.clip(base_signal, -1.0, 1.0)
        signals[currency] = signal
    return signals

def generate_positions(signals, capital):
    """
    Convert signals to position sizes using equal weighting
    
    Kelly optimization DISABLED: Hurt performance (-1.00%) with weak models
    Using conservative 50/50 allocation until models improve (R² > 0.15)
    """
    positions = {}
    
    # Equal weight allocation (was Kelly: 71% EUR, 29% CHF)
    # Reverted because model R² too low (EUR=0.09, CHF=0.04)
    weights = {
        'EUR': 0.50,  # Equal weight
        'CHF': 0.50   # Equal weight
    }
    
    # 30% total exposure (conservative)
    max_exposure = capital * 0.30
    
    for currency, signal in signals.items():
        # Position = signal * weight * max_exposure
        position_size = signal * weights.get(currency, 0) * max_exposure
        positions[currency] = position_size
    
    return positions

def run_paper_trading():
    config = load_config()
    
    print("="*70)
    print("🚀 PAPER TRADING - SIMPLE MODE (Mock Signals)")
    print("="*70)
    print(f"\nMode: {config['strategy']}")
    print(f"Capital: ${config['initial_capital']:,.0f}")
    print(f"Currencies: {', '.join(config['ml_currencies'])}")
    print(f"\n⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n⚠️  NOTE: Using mock signals (real ML models hang on data fetch)")
    print("   This demonstrates the trading loop without FRED API delays")
    print("\n" + "="*70 + "\n")
    
    # Model R² scores (from actual training)
    model_r2 = {
        'EUR': 0.0905,
        'CHF': 0.0369
    }
    
    print("📊 Model Performance:")
    for curr, r2 in model_r2.items():
        print(f"   {curr}: R² = {r2:.4f}")
    print()
    
    # Main trading loop
    iteration = 0
    total_pnl = 0
    capital = config['initial_capital']
    
    while True:
        iteration += 1
        current_time = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"🔄 Iteration {iteration} - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        try:
            # Generate signals
            print("🎯 Generating signals...")
            signals = generate_mock_signals(config['ml_currencies'], model_r2)
            
            # Generate positions
            print("💰 Calculating positions...")
            positions = generate_positions(
                signals, 
                capital,
                config['max_position_size'],
                model_r2
            )
            
            # Simulate daily returns (mock)
            daily_pnl = 0
            print(f"\n📊 Signals & Positions:")
            print(f"{'Currency':<10} {'Signal':>8} {'Position':>15} {'Daily P&L':>12}")
            print("-" * 55)
            
            for currency in config['ml_currencies']:
                signal = signals[currency]
                position = positions.get(currency, 0)
                
                # Mock daily return: signal * volatility + noise
                daily_return = signal * 0.001 + np.random.randn() * 0.005
                pnl = position * daily_return
                daily_pnl += pnl
                
                print(f"{currency:<10} {signal:>8.4f} ${position:>14,.0f} ${pnl:>11,.2f}")
            
            total_pnl += daily_pnl
            capital += daily_pnl
            
            print("-" * 55)
            print(f"{'Total':<10} {'':<8} {'':<15} ${daily_pnl:>11,.2f}")
            
            print(f"\n💵 Account Summary:")
            print(f"   Capital: ${capital:,.2f}")
            print(f"   Total P&L: ${total_pnl:>,.2f} ({total_pnl/config['initial_capital']*100:+.2f}%)")
            print(f"   Sharpe (annualized): {(total_pnl / capital * np.sqrt(252 / iteration) if iteration > 0 else 0):.2f}")
            
            # Log to file
            log_entry = {
                'timestamp': current_time.isoformat(),
                'iteration': iteration,
                'signals': signals,
                'positions': positions,
                'daily_pnl': daily_pnl,
                'total_pnl': total_pnl,
                'capital': capital
            }
            
            with open(config['log_file'], 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            print(f"\n✅ Iteration complete")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait before next iteration (60 seconds for demo, 86400 for production)
        wait_time = 60  # 1 minute for demo
        print(f"\n⏰ Next update in {wait_time} seconds...")
        print(f"   Press Ctrl+C to stop")
        time.sleep(wait_time)

if __name__ == "__main__":
    try:
        run_paper_trading()
    except KeyboardInterrupt:
        print("\n\n⏹  Paper trading stopped by user")
        print("\n" + "="*70)
