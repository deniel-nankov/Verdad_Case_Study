"""
Example: How to Use the Live Trading System
This script demonstrates basic usage patterns
"""

import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# EXAMPLE 1: PAPER TRADING (Recommended Starting Point)
# ============================================================================

def example_paper_trading():
    """Start paper trading with default settings"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Paper Trading")
    print("="*60)
    
    from live_trading_system import (
        LiveTradingSystem, 
        AlphaVantageDataFeed, 
        TradingMode,
        RiskLimits,
        RiskManager
    )
    from broker_integrations import BrokerConfig, create_broker
    
    # 1. Set up data feed
    api_key = os.getenv('ALPHAVANTAGE_API_KEY', 'demo')
    data_feed = AlphaVantageDataFeed(api_key)
    
    # 2. Set up paper trading broker
    broker_config = BrokerConfig(
        broker_name='paper',
        api_key='',
        practice=True
    )
    broker = create_broker(broker_config)
    broker.connect()
    
    # 3. Configure risk limits
    risk_limits = RiskLimits(
        max_position_pct=0.30,      # 30% max per position
        max_exposure=2.0,            # 2x max total exposure
        max_drawdown_pct=0.15,       # Stop trading at 15% drawdown
        stop_loss_pct=0.05,          # 5% stop-loss per position
        daily_loss_limit_pct=0.03    # 3% max daily loss
    )
    
    initial_capital = 100000  # $100k
    risk_manager = RiskManager(risk_limits, initial_capital)
    
    # 4. Create trading system
    trading_system = LiveTradingSystem(
        data_feed=data_feed,
        broker=broker,
        risk_manager=risk_manager,
        mode=TradingMode.PAPER,
        strategy_type='optimized',
        lookback_days=756
    )
    
    # 5. Start trading (will run continuously)
    print("\n🚀 Starting paper trading system...")
    print("   Press Ctrl+C to stop")
    print("   Check trading_system.log for activity")
    print("   Monitor with: python monitoring_dashboard.py\n")
    
    # Run with daily rebalancing
    # trading_system.run(rebalance_frequency_hours=24)
    
    print("✅ Paper trading system initialized")
    print("   (Uncomment line 60 to actually run)")
    

# ============================================================================
# EXAMPLE 2: SINGLE REBALANCE (For Testing)
# ============================================================================

def example_single_rebalance():
    """Execute a single portfolio rebalance (for testing)"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Single Rebalance Test")
    print("="*60)
    
    from live_trading_system import (
        LiveTradingSystem,
        AlphaVantageDataFeed,
        TradingMode,
        RiskLimits,
        RiskManager
    )
    from broker_integrations import BrokerConfig, create_broker
    
    # Set up system (same as Example 1)
    api_key = os.getenv('ALPHAVANTAGE_API_KEY', 'demo')
    data_feed = AlphaVantageDataFeed(api_key)
    
    broker_config = BrokerConfig(broker_name='paper', api_key='', practice=True)
    broker = create_broker(broker_config)
    broker.connect()
    
    risk_limits = RiskLimits()
    risk_manager = RiskManager(risk_limits, 100000)
    
    trading_system = LiveTradingSystem(
        data_feed=data_feed,
        broker=broker,
        risk_manager=risk_manager,
        mode=TradingMode.PAPER
    )
    
    # Execute single rebalance
    print("\n🔄 Executing single rebalance...")
    success = trading_system.rebalance_portfolio()
    
    if success:
        print("✅ Rebalance completed successfully")
        
        # Check positions
        positions = broker.get_positions()
        print(f"\n📊 Current positions: {len(positions)}")
        for currency, quantity in positions.items():
            print(f"   {currency}: {quantity:+,.0f}")
    else:
        print("❌ Rebalance failed")


# ============================================================================
# EXAMPLE 3: OANDA BROKER (Requires Valid Credentials)
# ============================================================================

def example_oanda_broker():
    """Use OANDA broker for live/demo trading"""
    print("\n" + "="*60)
    print("EXAMPLE 3: OANDA Broker Integration")
    print("="*60)
    
    # Check for credentials
    api_key = os.getenv('BROKER_API_KEY')
    account_id = os.getenv('BROKER_ACCOUNT_ID')
    
    if not api_key or not account_id:
        print("⚠️  OANDA credentials not found in .env file")
        print("\n   Add these to your .env file:")
        print("   BROKER_API_KEY=your_oanda_api_key")
        print("   BROKER_ACCOUNT_ID=your_oanda_account_id")
        return
    
    from broker_integrations import BrokerConfig, create_broker
    
    # Create OANDA broker (PRACTICE mode)
    broker_config = BrokerConfig(
        broker_name='oanda',
        api_key=api_key,
        account_id=account_id,
        practice=True  # ALWAYS start with practice!
    )
    
    broker = create_broker(broker_config)
    
    # Test connection
    if broker.connect():
        print("✅ Connected to OANDA")
        
        balance = broker.get_account_balance()
        print(f"   Account balance: ${balance:,.2f}")
        
        positions = broker.get_positions()
        print(f"   Open positions: {len(positions)}")
        
    else:
        print("❌ Failed to connect to OANDA")


# ============================================================================
# EXAMPLE 4: MONITORING DASHBOARD
# ============================================================================

def example_monitoring():
    """Display monitoring dashboard"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Monitoring Dashboard")
    print("="*60)
    
    print("\n📊 To view live dashboard:")
    print("   python monitoring_dashboard.py")
    print("\n📊 To view dashboard once:")
    print("   python monitoring_dashboard.py --once")
    print("\n📊 To export HTML report:")
    print("   python monitoring_dashboard.py --export")
    print("\n💡 Dashboard shows:")
    print("   - Portfolio value and P&L")
    print("   - Performance metrics (Sharpe, drawdown)")
    print("   - Current positions")
    print("   - Risk metrics")
    print("   - Recent trading activity")


# ============================================================================
# EXAMPLE 5: ALERT SYSTEM
# ============================================================================

def example_alerts():
    """Set up alert system"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Alert System")
    print("="*60)
    
    from alert_system import AlertSystem, AlertLevel
    
    # Initialize alert system
    alert_system = AlertSystem('trading_config.json')
    
    # Send test alerts
    alert_system.send_alert("System initialized", AlertLevel.INFO)
    alert_system.alert_trade_executed("USDEUR", 10000, 0.9234)
    alert_system.alert_system_started("paper", 100000)
    
    print("\n✅ Alert system configured")
    print("\n💡 To enable email/Slack alerts:")
    print("   1. Edit trading_config.json")
    print("   2. Add your SMTP/Slack credentials")
    print("   3. Set alert_on_trade: true")


# ============================================================================
# EXAMPLE 6: RUNNING TESTS
# ============================================================================

def example_testing():
    """Run system tests"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Testing Data Feeds")
    print("="*60)
    
    print("\n🧪 To test all data feeds and connections:")
    print("   python test_data_feeds.py")
    print("\n   This will verify:")
    print("   ✓ Environment variables are set")
    print("   ✓ Alpha Vantage API is working")
    print("   ✓ FRED API is working")
    print("   ✓ Broker connection is valid")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Main menu for examples"""
    print("\n" + "="*60)
    print("🚀 LIVE TRADING SYSTEM - USAGE EXAMPLES")
    print("="*60)
    print("\nSelect an example to run:")
    print("\n1. Paper Trading (recommended start)")
    print("2. Single Rebalance Test")
    print("3. OANDA Broker Integration")
    print("4. Monitoring Dashboard")
    print("5. Alert System")
    print("6. Testing Data Feeds")
    print("0. Run all examples (demo mode)")
    
    choice = input("\nEnter choice (0-6): ").strip()
    
    examples = {
        '1': example_paper_trading,
        '2': example_single_rebalance,
        '3': example_oanda_broker,
        '4': example_monitoring,
        '5': example_alerts,
        '6': example_testing
    }
    
    if choice == '0':
        print("\n🎬 Running all examples in demo mode...\n")
        for func in examples.values():
            func()
            print("\n" + "-"*60 + "\n")
    elif choice in examples:
        examples[choice]()
    else:
        print("\n❌ Invalid choice")
    
    print("\n" + "="*60)
    print("📚 For more information:")
    print("   - Read LIVE_TRADING_SETUP.md for complete setup guide")
    print("   - Check trading_config.json for configuration options")
    print("   - Review live_trading_system.py for implementation details")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
