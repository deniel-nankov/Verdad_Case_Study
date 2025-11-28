"""
Advanced OANDA API Diagnostics
Tests different authentication methods and provides detailed error analysis
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('BROKER_API_KEY')
account_id = os.getenv('BROKER_ACCOUNT_ID')

print("=" * 70)
print("ADVANCED OANDA API DIAGNOSTICS")
print("=" * 70)
print()

# Display credentials (masked)
print("📋 Credentials:")
print(f"   API Key: {api_key[:10]}...{api_key[-10:]}")
print(f"   Account ID: {account_id}")
print(f"   Key Length: {len(api_key)} characters")
print()

# Test 1: Raw HTTP request to practice environment
print("1️⃣ Testing Raw HTTP Request (Practice Environment)")
print("-" * 70)

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

url = f'https://api-fxpractice.oanda.com/v3/accounts'

try:
    response = requests.get(url, headers=headers, timeout=10)
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.text[:200]}")
    
    if response.status_code == 200:
        print("   ✅ SUCCESS - API key is valid!")
        data = response.json()
        if 'accounts' in data:
            print(f"   Found {len(data['accounts'])} account(s):")
            for acc in data['accounts']:
                print(f"      - {acc['id']}")
    elif response.status_code == 401:
        print("   ❌ 401 UNAUTHORIZED")
        print()
        print("   Possible reasons:")
        print("   1. API key is invalid or expired")
        print("   2. API key is for LIVE environment (not practice)")
        print("   3. API key format is incorrect")
        print("   4. Token needs to be regenerated")
    elif response.status_code == 403:
        print("   ❌ 403 FORBIDDEN")
        print("   API key exists but doesn't have permission")
    else:
        print(f"   ❌ Unexpected error: {response.status_code}")
        
except Exception as e:
    print(f"   ❌ Request failed: {e}")

print()

# Test 2: Try LIVE environment
print("2️⃣ Testing LIVE Environment (api.oanda.com)")
print("-" * 70)

url_live = f'https://api.oanda.com/v3/accounts'

try:
    response = requests.get(url_live, headers=headers, timeout=10)
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.text[:200]}")
    
    if response.status_code == 200:
        print("   ✅ SUCCESS - Your key is for LIVE environment!")
        print()
        print("   ⚠️  WARNING: You need to use 'live' environment, not 'practice'")
        print("   Update your code: API(access_token=key, environment='live')")
    elif response.status_code == 401:
        print("   ❌ Also unauthorized in LIVE environment")
    else:
        print(f"   ❌ Error: {response.status_code}")
        
except Exception as e:
    print(f"   ❌ Request failed: {e}")

print()

# Test 3: Check API key format
print("3️⃣ API Key Format Analysis")
print("-" * 70)

# OANDA keys should have this format: xxxxx-xxxxx (with hyphen)
if '-' in api_key:
    parts = api_key.split('-')
    print(f"   ✅ Key has hyphen separator")
    print(f"   Parts: {len(parts)} segments")
    print(f"   Lengths: {[len(p) for p in parts]}")
else:
    print(f"   ❌ Key missing hyphen separator")
    print(f"   OANDA keys should be in format: xxxxx-xxxxx")

# Check for common issues
issues = []
if ' ' in api_key:
    issues.append("Contains spaces (remove them)")
if api_key != api_key.strip():
    issues.append("Has leading/trailing whitespace")
if '\n' in api_key or '\r' in api_key:
    issues.append("Contains newline characters")

if issues:
    print()
    print("   ⚠️  Potential Issues Found:")
    for issue in issues:
        print(f"      - {issue}")
else:
    print(f"   ✅ No obvious format issues detected")

print()

# Test 4: Test with specific account endpoint
print("4️⃣ Testing Account-Specific Endpoint")
print("-" * 70)

url_account = f'https://api-fxpractice.oanda.com/v3/accounts/{account_id}'

try:
    response = requests.get(url_account, headers=headers, timeout=10)
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.text[:200]}")
    
    if response.status_code == 200:
        print("   ✅ SUCCESS - Account access granted!")
    elif response.status_code == 401:
        print("   ❌ Unauthorized for this account")
        print()
        print("   This means:")
        print("   - The API key itself may be valid")
        print("   - But it doesn't have permission for account:", account_id)
        print("   - The account ID may be incorrect")
        print("   - Or the key is for a different account")
    elif response.status_code == 404:
        print("   ❌ Account not found")
        print(f"   Account ID '{account_id}' doesn't exist")
    else:
        print(f"   ❌ Error: {response.status_code}")
        
except Exception as e:
    print(f"   ❌ Request failed: {e}")

print()
print("=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)
print()
print("🔍 Next Steps:")
print()
print("1. Go to OANDA's token management page:")
print("   Practice: https://www.oanda.com/account/tpa/personal_token")
print("   Live: https://www1.oanda.com/account/tpa/personal_token")
print()
print("2. Generate a NEW Personal Access Token")
print("   (Old tokens may expire or be revoked)")
print()
print("3. Verify which account the token is for:")
print("   The token page should show your account ID")
print()
print("4. Update your .env file:")
print("   BROKER_API_KEY=<your_new_token>")
print("   BROKER_ACCOUNT_ID=<account_from_token_page>")
print()
print("5. Test again:")
print("   python test_oanda_advanced.py")
print()
print("=" * 70)
