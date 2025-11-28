"""
Test Data Feeds for Live Trading System
Run this before starting live trading to verify all connections
"""

import os
import sys
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_alpha_vantage():
    """Test Alpha Vantage API connection"""
    print("\n" + "="*60)
    print("Testing Alpha Vantage Data Feed")
    print("="*60)
    
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if not api_key:
        print("❌ ALPHAVANTAGE_API_KEY not found in environment")
        print("   Set it in .env file or environment variables")
        return False
    
    try:
        from live_trading_system import AlphaVantageDataFeed
        
        feed = AlphaVantageDataFeed(api_key)
        
        # Test FX rate
        print("\n📊 Testing FX rate fetch...")
        fx_data = feed.get_fx_rate('USDEUR')
        if fx_data and isinstance(fx_data, dict):
            rate = fx_data.get('rate', 0)
            if rate > 0:
                print(f"✅ USD/EUR rate: {rate:.4f}")
                print(f"   Bid: {fx_data.get('bid', 0):.4f}, Ask: {fx_data.get('ask', 0):.4f}")
            else:
                print("❌ Failed to fetch FX rate")
                return False
        else:
            print("❌ Failed to fetch FX rate")
            return False
        
        # Test interest rate
        print("\n📊 Testing interest rate fetch...")
        int_rate = feed.get_interest_rate('USD')
        if int_rate is not None:
            print(f"✅ USD interest rate: {int_rate:.2f}%")
        else:
            print("❌ Failed to fetch interest rate")
            return False
        
        print("\n✅ Alpha Vantage connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Alpha Vantage test failed: {e}")
        return False


def test_fred_api():
    """Test FRED API connection"""
    print("\n" + "="*60)
    print("Testing FRED Data Feed")
    print("="*60)
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("❌ FRED_API_KEY not found in environment")
        print("   Set it in .env file or environment variables")
        return False
    
    try:
        from fredapi import Fred
        
        fred = Fred(api_key=api_key)
        
        # Test fetching a series
        print("\n📊 Testing FRED data fetch...")
        data = fred.get_series('DFF', limit=5)  # Fed Funds Rate
        
        if data is not None and len(data) > 0:
            latest_value = data.iloc[-1]
            latest_date = data.index[-1]
            print(f"✅ Fed Funds Rate: {latest_value:.2f}% (as of {latest_date.date()})")
            print("\n✅ FRED API connection successful!")
            return True
        else:
            print("❌ Failed to fetch FRED data")
            return False
            
    except Exception as e:
        print(f"❌ FRED test failed: {e}")
        return False


def test_broker_connection():
    """Test broker connection"""
    print("\n" + "="*60)
    print("Testing Broker Connection")
    print("="*60)
    
    broker_type = os.getenv('BROKER_TYPE', 'paper')
    
    if broker_type.lower() == 'paper':
        print("\n📄 Paper trading mode - no broker connection needed")
        try:
            from broker_integrations import BrokerConfig, create_broker
            
            config = BrokerConfig(
                broker_name='paper',
                api_key='',
                practice=True
            )
            
            broker = create_broker(config)
            connected = broker.connect()
            
            if connected:
                balance = broker.get_account_balance()
                print(f"✅ Paper broker initialized with ${balance:,.2f}")
                return True
            else:
                print("❌ Failed to initialize paper broker")
                return False
                
        except Exception as e:
            print(f"❌ Paper broker test failed: {e}")
            return False
    
    elif broker_type.lower() == 'oanda':
        print("\n🌐 Testing OANDA connection...")
        
        api_key = os.getenv('BROKER_API_KEY')
        account_id = os.getenv('BROKER_ACCOUNT_ID')
        
        if not api_key or not account_id:
            print("❌ BROKER_API_KEY or BROKER_ACCOUNT_ID not found")
            return False
        
        try:
            from broker_integrations import BrokerConfig, create_broker
            
            config = BrokerConfig(
                broker_name='oanda',
                api_key=api_key,
                account_id=account_id,
                practice=True  # Always test with practice first
            )
            
            broker = create_broker(config)
            connected = broker.connect()
            
            if connected:
                balance = broker.get_account_balance()
                positions = broker.get_positions()
                print(f"✅ OANDA connected - Balance: ${balance:,.2f}")
                print(f"   Open positions: {len(positions)}")
                return True
            else:
                print("❌ Failed to connect to OANDA")
                return False
                
        except Exception as e:
            print(f"❌ OANDA test failed: {e}")
            print("   Make sure oandapyV20 is installed: pip install oandapyV20")
            return False
    
    else:
        print(f"⚠️  Broker type '{broker_type}' not yet tested")
        print("   Supported: paper, oanda")
        return False


def test_environment_variables():
    """Check all required environment variables"""
    print("\n" + "="*60)
    print("Checking Environment Variables")
    print("="*60)
    
    required_vars = {
        'ALPHAVANTAGE_API_KEY': 'Alpha Vantage API key',
        'FRED_API_KEY': 'FRED API key'
    }
    
    optional_vars = {
        'BROKER_TYPE': 'Broker type (paper/oanda/ib)',
        'BROKER_API_KEY': 'Broker API key',
        'BROKER_ACCOUNT_ID': 'Broker account ID'
    }
    
    all_good = True
    
    print("\nRequired variables:")
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            masked = value[:4] + '...' + value[-4:] if len(value) > 8 else '***'
            print(f"  ✅ {var}: {masked}")
        else:
            print(f"  ❌ {var}: NOT SET - {description}")
            all_good = False
    
    print("\nOptional variables:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            masked = value[:4] + '...' + value[-4:] if len(value) > 8 else value
            print(f"  ✅ {var}: {masked}")
        else:
            print(f"  ⚠️  {var}: not set - {description}")
    
    if not all_good:
        print("\n❌ Some required variables are missing!")
        print("\n💡 Create a .env file with:")
        print("   ALPHAVANTAGE_API_KEY=your_key_here")
        print("   FRED_API_KEY=your_key_here")
        return False
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 LIVE TRADING SYSTEM - DATA FEED TESTS")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed - using system environment variables")
        print("   Install with: pip install python-dotenv")
    
    results = {}
    
    # Run tests
    results['environment'] = test_environment_variables()
    
    if results['environment']:
        results['alpha_vantage'] = test_alpha_vantage()
        results['fred'] = test_fred_api()
        results['broker'] = test_broker_connection()
    else:
        print("\n⚠️  Skipping API tests due to missing environment variables")
        results['alpha_vantage'] = False
        results['fred'] = False
        results['broker'] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - System ready for trading!")
        print("="*60)
        print("\n💡 Next steps:")
        print("   1. Review trading_config.json settings")
        print("   2. Run in paper mode first: python live_trading_system.py")
        print("   3. Monitor logs: tail -f trading_system.log")
        print("   4. Check performance: python monitoring_dashboard.py")
    else:
        print("❌ SOME TESTS FAILED - Fix issues before trading!")
        print("="*60)
        print("\n💡 Troubleshooting:")
        print("   1. Check .env file has correct API keys")
        print("   2. Install missing packages: pip install -r requirements_live.txt")
        print("   3. Verify broker credentials if using live broker")
        print("   4. Check LIVE_TRADING_SETUP.md for detailed setup guide")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
