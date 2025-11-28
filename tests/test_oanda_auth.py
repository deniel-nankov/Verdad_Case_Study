"""
Test OANDA API Authentication
Diagnoses authorization issues with OANDA API credentials
"""
import os
from dotenv import load_dotenv
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.pricing as pricing

# Load environment variables
load_dotenv()

api_key = os.getenv('BROKER_API_KEY')
account_id = os.getenv('BROKER_ACCOUNT_ID')

print("=" * 60)
print("OANDA API Authentication Test")
print("=" * 60)
print(f"\nAPI Key: {api_key[:20]}...{api_key[-20:]}")
print(f"Account ID: {account_id}")
print(f"Environment: practice (fxpractice.oanda.com)")

# Initialize client
print("\n1️⃣ Creating OANDA API client...")
try:
    client = API(access_token=api_key, environment='practice')
    print("✅ Client created successfully")
except Exception as e:
    print(f"❌ Failed to create client: {e}")
    exit(1)

# Test 1: List accounts (doesn't require account ID)
print("\n2️⃣ Testing accounts list endpoint...")
try:
    r = accounts.AccountList()
    response = client.request(r)
    print(f"✅ Success! Found {len(response.get('accounts', []))} account(s):")
    for acc in response.get('accounts', []):
        print(f"   - Account ID: {acc['id']}")
        print(f"     Tags: {acc.get('tags', [])}")
except Exception as e:
    print(f"❌ Failed: {e}")
    print("\n💡 This suggests the API key itself is invalid or expired.")
    print("   Please verify your API key from OANDA's dashboard:")
    print("   https://www.oanda.com/account/tpa/personal_token")

# Test 2: Account summary (requires account ID)
print("\n3️⃣ Testing account summary endpoint...")
try:
    r = accounts.AccountSummary(accountID=account_id)
    response = client.request(r)
    print("✅ Success! Account details:")
    acc = response['account']
    print(f"   - Currency: {acc.get('currency')}")
    print(f"   - Balance: {acc.get('balance')}")
    print(f"   - Open Trades: {acc.get('openTradeCount', 0)}")
    print(f"   - Open Positions: {acc.get('openPositionCount', 0)}")
except Exception as e:
    print(f"❌ Failed: {e}")
    print("\n💡 Possible issues:")
    print("   1. Account ID is incorrect")
    print("   2. API key doesn't have permission for this account")
    print("   3. Account is not a practice account (using wrong environment)")

# Test 3: Pricing data (requires valid instruments)
print("\n4️⃣ Testing pricing endpoint...")
try:
    params = {"instruments": "EUR_USD,GBP_USD"}
    r = pricing.PricingInfo(accountID=account_id, params=params)
    response = client.request(r)
    print(f"✅ Success! Retrieved prices for {len(response['prices'])} instruments:")
    for price in response['prices']:
        print(f"   - {price['instrument']}: {price['closeoutBid']} / {price['closeoutAsk']}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "=" * 60)
print("Diagnosis Complete")
print("=" * 60)
print("\n📋 Next Steps:")
print("   1. If Test 1 failed: Get a new API token from OANDA")
print("   2. If Test 2 failed: Verify account ID matches the token's account")
print("   3. If Test 3 failed: Check account permissions and instruments")
print("\n🔗 OANDA Token Management:")
print("   Practice: https://www.oanda.com/account/tpa/personal_token")
print("   Live: https://www1.oanda.com/account/tpa/personal_token")
print("=" * 60)
