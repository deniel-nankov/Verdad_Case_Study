# 🔴 OANDA API Token Generation Guide

**Issue**: All API keys tested are **invalid or expired**  
**Solution**: Generate a **NEW** Personal Access Token

---

## 🎯 Step-by-Step Guide

### Step 1: Log Into OANDA

**Practice Account** (Recommended for testing):
```
URL: https://www.oanda.com/
Click: "Sign In" (top right)
Select: Practice Account
```

**OR Live Account** (Real money):
```
URL: https://www1.oanda.com/
Click: "Sign In"
Select: Live Account
```

---

### Step 2: Navigate to API Token Management

Once logged in:

1. Click on your **name/profile** (top right corner)
2. Select **"Manage API Access"** or **"API Access"**
3. Click on **"Personal Access Tokens"** or **"Generate Token"**

**Direct Links**:
- Practice: https://www.oanda.com/account/tpa/personal_token
- Live: https://www1.oanda.com/account/tpa/personal_token

---

### Step 3: Generate New Token

On the token management page:

1. Click **"Generate"** or **"Regenerate Token"**
2. You'll see:
   ```
   Personal Access Token:
   xxxxxxxx-xxxxxxxx-xxxxxxxx-xxxxxxxx
   
   Account ID:
   XXX-XXX-XXXXXXXX-XXX
   ```

3. **⚠️ COPY BOTH IMMEDIATELY!**
   - The token is shown **only once**
   - You cannot retrieve it later
   - Write it down or save to secure location

---

### Step 4: Update Your .env File

```bash
# Open .env file
cd /Users/denielnankov/Documents/Verdad_Technical_Case_Study
nano .env
```

Update these lines with your NEW values:
```bash
BROKER_API_KEY=<paste_your_new_token_here>
BROKER_ACCOUNT_ID=<paste_account_id_here>
```

**Example**:
```bash
# If token page shows:
# Token: abc123-def456
# Account: 101-019-12345678-001

# Then update to:
BROKER_API_KEY=abc123-def456
BROKER_ACCOUNT_ID=101-019-12345678-001
```

Save the file (Ctrl+O, Enter, Ctrl+X in nano)

---

### Step 5: Test the New Token

```bash
source venv_fx/bin/activate
python test_oanda_advanced.py
```

**Expected Success Output**:
```
✅ SUCCESS - API key is valid!
Found 1 account(s):
   - 101-019-XXXXXXXX-001

✅ SUCCESS - Account access granted!
Currency: USD
Balance: 100000.0000
```

---

## 🔍 What You Tested (All Failed)

```
API Key 1: 397d1e8f... ❌ Invalid
API Key 2: d03b38e1... ❌ Invalid
API Key 3: 5a1b4086... ❌ Invalid (current)

All returned: 401 Unauthorized
Reason: Tokens are expired/revoked/invalid
```

---

## ❓ Common Issues

### "I don't see 'Manage API Access' option"
- Your account may not have API access enabled
- Contact OANDA support to enable API access
- Or use demo/practice account which has it by default

### "Token page shows no account"
- You may not have an account set up yet
- Create a practice account first
- Then generate token

### "Still getting 401 after generating new token"
- Make sure you copied the ENTIRE token
- Check for extra spaces or line breaks
- Verify you're using the practice token with practice.oanda.com

### "Account ID format unclear"
- OANDA format: `XXX-XXX-XXXXXXXX-XXX`
- Example: `101-019-19582015-001`
- Use EXACTLY as shown on token page

---

## 💡 Alternative: Keep Using Yahoo Finance

If OANDA token generation is too complex, **your system works perfectly without it!**

**Current Setup (Working)**:
- ✅ Real market data from Yahoo Finance
- ✅ Zero authentication issues
- ✅ Perfect for paper trading
- ✅ Auto-refreshes every 6 hours
- ✅ Already running successfully

**OANDA is NOT required** for paper trading validation!

---

## 🎯 Decision Time

### Option A: Get OANDA Working (Advanced)
**Steps**:
1. Follow guide above
2. Generate new token
3. Update .env file
4. Test with `python test_oanda_advanced.py`

**Time**: 10-15 minutes  
**Complexity**: Medium  
**Benefit**: Real-time data (not needed for 24h strategy)

### Option B: Keep Yahoo Finance (Recommended)
**Steps**:
1. Do nothing - already working!

**Time**: 0 minutes  
**Complexity**: None  
**Benefit**: Focus on trading, not API troubleshooting

---

## 📞 Need Help?

If token generation is confusing:

1. **Take screenshot** of OANDA token page
2. **Share** (hide the actual token value!)
3. I can guide you through exact format

Or just stick with Yahoo Finance - **it's the smart choice for paper trading!** ✅
