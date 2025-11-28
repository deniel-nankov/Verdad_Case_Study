#!/usr/bin/env python3
"""
Setup Paper Trading for ML FX Strategy
Configures OANDA practice account with ML signals
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def setup_paper_trading_config():
    """Create paper trading configuration"""
    
    print("="*70)
    print("📄 SETTING UP PAPER TRADING CONFIGURATION")
    print("="*70)
    
    # Load existing config or create new
    config_file = 'trading_config.json'
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"\n✅ Loaded existing config: {config_file}")
    else:
        config = {}
        print(f"\n📝 Creating new config: {config_file}")
    
    # Paper trading configuration
    paper_config = {
        "mode": "paper",  # paper or live
        "strategy": "ml_hybrid",  # ml_only, ml_hybrid, or ml_ensemble
        
        # Capital & Risk
        "initial_capital": 100000,
        "max_position_size": 0.30,
        "max_total_exposure": 0.70,
        "cash_reserve": 0.15,
        "max_leverage": 2.0,
        
        # ML Strategy
        "ml_currencies": ["EUR", "CHF"],
        "carry_currencies": ["AUD", "CAD", "GBP", "JPY"],
        "min_signal_strength": 0.30,
        
        # Trading Rules
        "max_spread_pips": 3,
        "max_vix_level": 30,
        "signal_confirmation_days": 2,
        "rebalance_frequency": "weekly",
        
        # Risk Management
        "hard_stop_loss": -0.02,
        "trailing_stop": 0.015,
        "daily_loss_limit": -0.03,
        "weekly_loss_limit": -0.05,
        
        # OANDA Settings
        "oanda_account_type": "practice",  # practice or live
        "oanda_account_id": os.getenv('OANDA_ACCOUNT_ID', ''),
        "oanda_api_key": os.getenv('OANDA_API_KEY', ''),
        "oanda_environment": "practice",  # practice or live
        
        # Model Settings
        "model_dir": "./ml_models",
        "retrain_frequency": "monthly",
        "feature_update_frequency": "daily",
        
        # Monitoring
        "log_file": "paper_trading.log",
        "performance_file": "paper_performance.csv",
        "send_alerts": True,
        "alert_email": os.getenv('ALERT_EMAIL', ''),
        
        # Metadata
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "version": "1.0",
        "description": "Paper trading configuration for ML FX strategy (EUR + CHF)"
    }
    
    # Update config
    config['paper_trading'] = paper_config
    
    # Save config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Paper trading config saved to: {config_file}")
    
    # Display configuration
    print("\n" + "="*70)
    print("📋 PAPER TRADING CONFIGURATION")
    print("="*70)
    
    print(f"\n💰 Capital & Risk:")
    print(f"   Initial Capital:        ${paper_config['initial_capital']:,.0f}")
    print(f"   Max Position Size:      {paper_config['max_position_size']*100:.0f}%")
    print(f"   Max Total Exposure:     {paper_config['max_total_exposure']*100:.0f}%")
    print(f"   Cash Reserve:           {paper_config['cash_reserve']*100:.0f}%")
    
    print(f"\n🎯 Strategy:")
    print(f"   Mode:                   {paper_config['strategy']}")
    print(f"   ML Currencies:          {', '.join(paper_config['ml_currencies'])}")
    print(f"   Carry Currencies:       {', '.join(paper_config['carry_currencies'])}")
    print(f"   Min Signal Strength:    {paper_config['min_signal_strength']:.2f}")
    
    print(f"\n🛡️  Risk Management:")
    print(f"   Hard Stop Loss:         {paper_config['hard_stop_loss']*100:.0f}%")
    print(f"   Trailing Stop:          {paper_config['trailing_stop']*100:.1f}%")
    print(f"   Daily Loss Limit:       {paper_config['daily_loss_limit']*100:.0f}%")
    print(f"   Weekly Loss Limit:      {paper_config['weekly_loss_limit']*100:.0f}%")
    
    print(f"\n🔄 Updates:")
    print(f"   Rebalance Frequency:    {paper_config['rebalance_frequency']}")
    print(f"   Model Retrain:          {paper_config['retrain_frequency']}")
    print(f"   Feature Update:         {paper_config['feature_update_frequency']}")
    
    print(f"\n🔌 OANDA:")
    print(f"   Environment:            {paper_config['oanda_environment']}")
    print(f"   Account Type:           {paper_config['oanda_account_type']}")
    account_id = paper_config['oanda_account_id']
    print(f"   Account ID:             {'*' * (len(account_id)-4) + account_id[-4:] if account_id else 'NOT SET'}")
    
    # Validation checks
    print("\n" + "="*70)
    print("🔍 VALIDATION CHECKS")
    print("="*70)
    
    checks = []
    
    # Check OANDA credentials
    if paper_config['oanda_account_id'] and paper_config['oanda_api_key']:
        checks.append(("✅", "OANDA credentials configured"))
    else:
        checks.append(("❌", "OANDA credentials missing - set in .env file"))
    
    # Check model files
    model_dir = paper_config['model_dir']
    ml_currencies = paper_config['ml_currencies']
    models_exist = True
    
    for currency in ml_currencies:
        currency_dir = os.path.join(model_dir, currency)
        if not os.path.exists(currency_dir):
            models_exist = False
            checks.append(("❌", f"{currency} models not found in {currency_dir}"))
        else:
            required_files = ['random_forest.pkl', 'xgboost.pkl', 'scaler.pkl']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(currency_dir, f))]
            if missing_files:
                models_exist = False
                checks.append(("❌", f"{currency} missing files: {', '.join(missing_files)}"))
            else:
                checks.append(("✅", f"{currency} models ready"))
    
    # Check configuration consistency
    if paper_config['max_position_size'] * len(ml_currencies) <= paper_config['max_total_exposure']:
        checks.append(("✅", "Position sizing consistent"))
    else:
        checks.append(("⚠️ ", "Position sizing may exceed total exposure limit"))
    
    if paper_config['cash_reserve'] + paper_config['max_total_exposure'] <= 1.0:
        checks.append(("✅", "Cash reserve configured"))
    else:
        checks.append(("⚠️ ", "Cash reserve + exposure > 100%"))
    
    # Print checks
    print()
    for status, message in checks:
        print(f"   {status} {message}")
    
    # Final recommendation
    print("\n" + "="*70)
    
    ready = all(check[0] == "✅" for check in checks)
    
    if ready:
        print("✅ PAPER TRADING READY TO START")
        print("\nNext steps:")
        print("   1. Run: python paper_trading_system.py")
        print("   2. Monitor: tail -f paper_trading.log")
        print("   3. Review: paper_performance.csv daily")
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("\nRequired actions:")
        for status, message in checks:
            if status != "✅":
                print(f"   • {message}")
    
    print("="*70 + "\n")
    
    return config

def create_paper_trading_script():
    """Create executable paper trading script"""
    
    script_content = """#!/usr/bin/env python3
\"\"\"
Paper Trading System - ML FX Strategy
Runs ML strategy in paper trading mode with OANDA practice account
\"\"\"

import json
import pandas as pd
import time
from datetime import datetime
from ml_fx.ml_strategy import MLFXStrategy
from dotenv import load_dotenv
import os

load_dotenv()

def load_config():
    with open('trading_config.json', 'r') as f:
        return json.load(f)['paper_trading']

def run_paper_trading():
    config = load_config()
    
    print("="*70)
    print("🚀 STARTING PAPER TRADING - ML FX STRATEGY")
    print("="*70)
    print(f"\\nMode: {config['strategy']}")
    print(f"Capital: ${config['initial_capital']:,.0f}")
    print(f"ML Currencies: {', '.join(config['ml_currencies'])}")
    print(f"\\n⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\\n" + "="*70 + "\\n")
    
    # Initialize strategy
    strategy = MLFXStrategy(
        fred_api_key=os.getenv('FRED_API_KEY'),
        currencies=config['ml_currencies']
    )
    
    # Load models
    for currency in config['ml_currencies']:
        print(f"📦 Loading {currency} model...")
        strategy.load_models(currency)
    
    print(f"\\n✅ All models loaded successfully\\n")
    
    # Main trading loop
    iteration = 0
    while True:
        iteration += 1
        current_time = datetime.now()
        
        print(f"\\n{'='*70}")
        print(f"🔄 Iteration {iteration} - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\\n")
        
        try:
            # Generate signals
            print("🎯 Generating ML signals...")
            signals = strategy.generate_signals()
            
            # Generate positions
            print("💰 Calculating positions...")
            positions = strategy.generate_positions(
                signals=signals,
                capital=config['initial_capital'],
                max_position_size=config['max_position_size'],
                risk_scale=1.0
            )
            
            # Display signals and positions
            print(f"\\n📊 Current Signals:")
            for currency, signal in signals.items():
                print(f"   {currency}: {signal:>7.4f}")
            
            print(f"\\n💼 Current Positions:")
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
                f.write(json.dumps(log_entry) + '\\n')
            
            print(f"\\n✅ Iteration complete")
            
        except Exception as e:
            print(f"\\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait before next iteration (daily)
        print(f"\\n⏰ Next update in 24 hours...")
        time.sleep(86400)  # 24 hours

if __name__ == "__main__":
    try:
        run_paper_trading()
    except KeyboardInterrupt:
        print("\\n\\n⏹  Paper trading stopped by user")
        print("\\n" + "="*70)
"""
    
    with open('paper_trading_system.py', 'w') as f:
        f.write(script_content)
    
    os.chmod('paper_trading_system.py', 0o755)
    
    print("✅ Created: paper_trading_system.py")

if __name__ == "__main__":
    # Setup configuration
    config = setup_paper_trading_config()
    
    # Create paper trading script
    create_paper_trading_script()
    
    print("\n🎉 Paper trading setup complete!")
    print("\nFiles created:")
    print("   - trading_config.json (configuration)")
    print("   - paper_trading_system.py (executable script)")
