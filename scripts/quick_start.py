#!/usr/bin/env python3
"""
Quick Start Example
===================

Demonstrates basic usage of the FX trading system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_feeds import create_data_feed
from src.core.risk_management import RiskManager, RiskLimits
from src.utils.config_loader import load_config

def main():
    print("="*60)
    print("FX Trading System - Quick Start Example")
    print("="*60)
    print()
    
    # 1. Load configuration
    print("1. Loading configuration...")
    try:
        config = load_config()
        print(f"   ✅ Config loaded")
        print(f"   Mode: {config.get('system', {}).get('mode', 'unknown')}")
        print(f"   Capital: ${config.get('system', {}).get('initial_capital', 0):,.0f}")
    except Exception as e:
        print(f"   ⚠️  Config loading failed: {e}")
        print(f"   Using defaults...")
        config = {'system': {'mode': 'paper', 'initial_capital': 100000}}
    print()
    
    # 2. Create data feed
    print("2. Creating data feed...")
    try:
        feed = create_data_feed('cached')
        print(f"   ✅ Data feed created (cached mode)")
        
        # Test getting a rate
        rate_data = feed.get_fx_rate('USDEUR')
        if rate_data:
            print(f"   EUR/USD rate: {rate_data['rate']:.4f}")
            print(f"   Source: {rate_data['source']}")
        else:
            print(f"   ⚠️  Could not get EUR/USD rate")
    except Exception as e:
        print(f"   ❌ Error creating data feed: {e}")
    print()
    
    # 3. Setup risk management
    print("3. Setting up risk management...")
    try:
        limits = RiskLimits(
            max_position_size=0.3,
            max_total_exposure=2.0,
            max_drawdown_pct=0.15,
            stop_loss_pct=0.05
        )
        
        initial_capital = config.get('system', {}).get('initial_capital', 100000)
        risk_mgr = RiskManager(limits, initial_capital)
        
        print(f"   ✅ Risk manager initialized")
        print(f"   Max position size: {limits.max_position_size*100}% of capital")
        print(f"   Max drawdown: {limits.max_drawdown_pct*100}%")
        
        # Test position limit check
        test_size = 25000
        test_price = 1.08
        is_ok = risk_mgr.check_position_limit('EUR', test_size, test_price)
        print(f"   Position check (${test_size:,.0f}): {'✅ OK' if is_ok else '❌ Exceeds limit'}")
    except Exception as e:
        print(f"   ❌ Error setting up risk management: {e}")
    print()
    
    # 4. Summary
    print("="*60)
    print("✅ Quick start example complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("  - Run backtests: python3 scripts/backtesting/run_backtest.py")
    print("  - Open notebooks: jupyter notebook notebooks/")
    print("  - See README.md for more examples")
    print()


if __name__ == '__main__':
    main()
