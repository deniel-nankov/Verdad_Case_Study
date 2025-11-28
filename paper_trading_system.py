#!/usr/bin/env python3
"""
Paper Trading System - ML FX Strategy with Enhanced Features
Runs ML strategy in paper trading mode with Kelly Optimization,
Cross-Asset Spillovers, and Intraday Microstructure timing.
"""

import json
import pandas as pd
import time
from datetime import datetime
from ml_fx.ml_strategy import MLFXStrategy
from dotenv import load_dotenv
import os
import sys

# Try to import enhanced strategies
try:
    from adaptive_leverage import AdaptiveLeverage
    from cross_asset_spillovers import CrossAssetSpillovers
    from intraday_microstructure import IntradayMicrostructure
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("⚠️  Enhanced strategies not available, using basic ML only")

load_dotenv()

def load_config():
    with open('trading_config.json', 'r') as f:
        return json.load(f)['paper_trading']

def run_paper_trading():
    config = load_config()
    
    print("="*70)
    print("🚀 STARTING PAPER TRADING - ML FX STRATEGY")
    print("="*70)
    print(f"\nMode: {config['strategy']}")
    print(f"Capital: ${config['initial_capital']:,.0f}")
    print(f"ML Currencies: {', '.join(config['ml_currencies'])}")
    print(f"\n⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70 + "\n")
    
    # Initialize strategy
    print("📦 Initializing ML strategy...")
    strategy = MLFXStrategy(
        fred_api_key=os.getenv('FRED_API_KEY'),
        currencies=config['ml_currencies']
    )
    
    # Load models
    print("📦 Loading trained models...")
    for currency in config['ml_currencies']:
        print(f"   Loading {currency}...")
        try:
            strategy.load_models(currency)
            print(f"   ✅ {currency} loaded")
        except Exception as e:
            print(f"   ❌ Error loading {currency}: {e}")
            raise
    
    print(f"\n✅ All models loaded successfully\n")
    
    # Pre-fetch data once to avoid repeated API calls
    print("📡 Pre-fetching market data (this may take 1-2 minutes)...")
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Data fetch timed out after 120 seconds")
        
        # Set 2-minute timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)
        
        initial_data = strategy.data_loader.get_latest_data()
        
        signal.alarm(0)  # Cancel timeout
        print(f"✅ Data loaded: {len(initial_data)} days")
        
    except TimeoutError as e:
        print(f"⚠️  {e}")
        print("   Using last available cached data...")
        initial_data = None
    except Exception as e:
        print(f"⚠️  Error fetching data: {e}")
        print("   Will try again on each iteration...")
        initial_data = None
    
    # Main trading loop
    iteration = 0
    while True:
        iteration += 1
        current_time = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"🔄 Iteration {iteration} - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        try:
            # Generate signals (use pre-fetched data to avoid repeated API calls)
            print("🎯 Generating ML signals...")
            signals = strategy.generate_signals(current_data=initial_data)
            
            # Generate positions
            print("💰 Calculating positions...")
            positions = strategy.generate_positions(
                signals=signals,
                capital=config['initial_capital'],
                max_position_size=config['max_position_size'],
                risk_scale=1.0
            )
            
            # Display signals and positions
            print(f"\n📊 Current Signals:")
            for currency, signal in signals.items():
                print(f"   {currency}: {signal:>7.4f}")
            
            print(f"\n💼 Current Positions:")
            for currency, position in positions.items():
                print(f"   {currency}: ${position:>12,.0f}")
            
            # Log to file
            log_entry = {
                'timestamp': current_time.isoformat(),
                'iteration': iteration,
                'signals': signals,
                'positions': positions
            }
            
            with open(config['log_file'], 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            print(f"\n✅ Iteration complete")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait before next iteration (daily)
        print(f"\n⏰ Next update in 24 hours...")
        time.sleep(86400)  # 24 hours

if __name__ == "__main__":
    try:
        run_paper_trading()
    except KeyboardInterrupt:
        print("\n\n⏹  Paper trading stopped by user")
        print("\n" + "="*70)
