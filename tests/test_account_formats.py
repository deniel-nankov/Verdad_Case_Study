"""
Test different OANDA account ID formats
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('BROKER_API_KEY')

# Different possible account ID formats
account_formats = [
    "19582015001",
    "101-019-19582015-001",
    "001-019-19582015-001",
    "19582015-001",
    "001-19582015-001"
]

print("=" * 70)
print("Testing Different Account ID Formats")
print("=" * 70)
print()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# First, try to list all accounts
print("🔍 Step 1: List all accounts accessible with this API key")
print("-" * 70)

url = 'https://api-fxpractice.oanda.com/v3/accounts'
try:
    response = requests.get(url, headers=headers, timeout=10)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS! Found accounts:")
        if 'accounts' in data:
            for acc in data['accounts']:
                print(f"   📌 Account ID: {acc['id']}")
                print(f"      Tags: {acc.get('tags', [])}")
        else:
            print("   No accounts found in response")
    else:
        print(f"❌ Failed: {response.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("🔍 Step 2: Try different account ID formats")
print("-" * 70)

for account_id in account_formats:
    print(f"\nTrying: {account_id}")
    url = f'https://api-fxpractice.oanda.com/v3/accounts/{account_id}'
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print(f"   ✅ SUCCESS! This is the correct format!")
            data = response.json()
            if 'account' in data:
                acc = data['account']
                print(f"   Currency: {acc.get('currency')}")
                print(f"   Balance: {acc.get('balance')}")
            break
        elif response.status_code == 404:
            print(f"   ❌ 404 Not Found - Account doesn't exist")
        elif response.status_code == 401:
            print(f"   ❌ 401 Unauthorized - Permission denied")
        else:
            print(f"   ❌ {response.status_code}: {response.text[:100]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print()
print("=" * 70)
print("🎯 RECOMMENDATION")
print("=" * 70)
print()
print("If all tests failed with 401 Unauthorized, the API key is invalid.")
print()
print("Next steps:")
print("1. Log into your OANDA account")
print("2. Go to: Manage API Access > Personal Access Tokens")
print("3. Generate a NEW token")
print("4. Note the account ID shown on that page")
print("5. Update both in .env file")
print()
