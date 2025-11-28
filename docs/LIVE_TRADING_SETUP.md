# Live Trading System Setup Guide

## 🚀 Quick Start

This guide will walk you through setting up the FX Carry live trading system for real-time execution.

---

## ⚠️ CRITICAL WARNING

**DO NOT** use real money until you have:
1. Extensively tested in paper trading mode (minimum 3 months)
2. Verified all risk controls are working
3. Validated data feeds are reliable
4. Obtained proper regulatory approvals (if required)
5. Fully understood the risks involved

**FX trading involves significant risk of loss. You can lose more than your initial investment.**

---

## 📋 Prerequisites

### 1. System Requirements
- Python 3.8 or higher
- Stable internet connection
- Minimum 8GB RAM (16GB recommended)
- 24/7 uptime (VPS recommended for production)

### 2. Required Python Packages
```bash
pip install pandas numpy requests python-dotenv schedule
pip install alpaca-trade-api  # If using Alpaca
pip install ib_insync  # If using Interactive Brokers
pip install oanda-v20  # If using OANDA
```

### 3. API Keys Needed

#### A. Data Feeds
- **Alpha Vantage** (Free tier: 5 calls/min, 500 calls/day)
  - Sign up: https://www.alphavantage.co/support/#api-key
  - Alternative: Twelve Data, Polygon.io

- **FRED API** (Free, unlimited)
  - Sign up: https://fred.stlouisfed.org/docs/api/api_key.html

#### B. Broker API (Choose One)

**Option 1: OANDA (Recommended for FX)**
- Pros: Good FX execution, reasonable spreads, easy API
- Sign up: https://www.oanda.com/us-en/trading/api/
- Paper trading: Full-featured demo account available

**Option 2: Interactive Brokers**
- Pros: Professional platform, lowest commissions
- Cons: Complex API, high minimum ($10k+)
- Sign up: https://www.interactivebrokers.com/

**Option 3: Alpaca (For stocks, crypto)**
- Pros: Simple API, good for beginners
- Cons: No FX trading (use for broader portfolio)
- Sign up: https://alpaca.markets/

---

## 🔧 Installation & Configuration

### Step 1: Clone Repository
```bash
git clone https://github.com/deniel-nankov/Verdad_Case_Study.git
cd Verdad_Case_Study
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv_live
source venv_live/bin/activate  # On Windows: venv_live\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements_live.txt
```

### Step 4: Configure API Keys

**Create `.env` file:**
```bash
# .env file (NEVER commit this to Git!)
ALPHAVANTAGE_API_KEY=your_actual_key_here
FRED_API_KEY=your_actual_key_here
BROKER_API_KEY=your_broker_key
BROKER_SECRET=your_broker_secret
BROKER_BASE_URL=https://paper-api.alpaca.markets  # Paper trading URL
```

**Update `trading_config.json`:**
```json
{
  "system": {
    "mode": "paper",  // Start with paper trading!
    "initial_capital": 100000
  },
  "api_keys": {
    "alpha_vantage": "${ALPHAVANTAGE_API_KEY}",
    "broker_api_key": "${BROKER_API_KEY}"
  }
}
```

### Step 5: Test Data Feeds
```bash
python test_data_feeds.py
```

Expected output:
```
✓ Alpha Vantage connection successful
✓ FX rate for USDEUR: 0.9234
✓ Interest rate for USD: 5.50%
✓ All data feeds operational
```

---

## 📊 Paper Trading Setup

### Starting Paper Trading Mode

**Option 1: Run directly**
```python
python live_trading_system.py
```

**Option 2: Custom script**
```python
from live_trading_system import *

# Initialize system
data_feed = AlphaVantageDataFeed(api_key="YOUR_KEY")
broker = PaperTradingBroker(initial_capital=100000)
risk_limits = RiskLimits(max_drawdown_pct=0.15)
risk_manager = RiskManager(risk_limits, 100000)

trading_system = LiveTradingSystem(
    data_feed=data_feed,
    broker=broker,
    risk_manager=risk_manager,
    mode=TradingMode.PAPER
)

# Start trading (runs 24/7)
trading_system.run(rebalance_frequency_hours=24)
```

### Monitoring Paper Trading

**Check logs:**
```bash
tail -f trading_system.log
```

**View performance:**
```python
import pandas as pd
perf = pd.read_csv('performance_log.csv')
print(perf.tail())
```

**Stop trading:**
- Press `Ctrl+C` in terminal
- Or programmatically: `trading_system.stop()`

---

## 🔴 Live Trading Setup

### ⚠️ ONLY AFTER 3+ MONTHS OF SUCCESSFUL PAPER TRADING

### Step 1: Open Live Broker Account
- Fund account with capital you can afford to lose
- Verify account is approved for FX trading
- Generate live API credentials

### Step 2: Update Configuration
```json
{
  "system": {
    "mode": "live",  // DANGER ZONE
    "initial_capital": 50000  // Start small!
  }
}
```

### Step 3: Implement Additional Safety Checks
```python
# Add manual approval for first trades
if mode == TradingMode.LIVE:
    confirm = input("LIVE TRADING MODE - Are you sure? (yes/no): ")
    if confirm.lower() != 'yes':
        exit()
```

### Step 4: Start with Minimum Capital
- Start with 10-20% of intended capital
- Monitor for 2 weeks before increasing
- Verify all risk controls work correctly

---

## 🛡️ Risk Management Checklist

Before going live, verify:

- [ ] Stop-loss triggers are working
- [ ] Daily loss limits are enforced
- [ ] Maximum position sizes are respected
- [ ] Drawdown limits halt trading
- [ ] Emergency liquidation works
- [ ] Email/SMS alerts are configured
- [ ] Backup internet connection available
- [ ] Power backup (UPS) is set up
- [ ] System logs are being saved
- [ ] Performance metrics are tracked

---

## 📈 Monitoring & Maintenance

### Daily Tasks
1. Check system logs for errors
2. Verify all positions are correct
3. Review overnight P&L
4. Confirm data feeds are updating

### Weekly Tasks
1. Analyze performance vs benchmark
2. Review risk metrics (Sharpe, drawdown)
3. Check for unusual activity
4. Update interest rate forecasts

### Monthly Tasks
1. Rebalance strategy weights (if using optimization)
2. Review transaction costs vs estimates
3. Analyze slippage and execution quality
4. Update ML models (if applicable)

---

## 🚨 Emergency Procedures

### If System Crashes
1. Immediately check open positions via broker website
2. Manually close positions if necessary
3. Review logs to identify cause
4. Test fix in paper mode before restarting

### If Market Disruption
1. System will auto-stop on drawdown limits
2. Manually liquidate if system fails
3. Contact broker support
4. Wait for market normalization

### If Data Feed Fails
1. System will pause trading automatically
2. Switch to backup data feed (if configured)
3. Manually close positions if prolonged outage

---

## 📞 Support & Resources

### Official Documentation
- Alpha Vantage API: https://www.alphavantage.co/documentation/
- OANDA API: https://developer.oanda.com/
- FRED API: https://fred.stlouisfed.org/docs/api/

### Community
- Quantitative Finance Stack Exchange
- r/algotrading on Reddit
- Elite Trader forums

### Emergency Contacts
- Broker support: [Your broker's phone number]
- Your trading partner: [If applicable]
- System administrator: [If using VPS]

---

## ⚖️ Legal & Regulatory

**Disclaimer:** This system is for educational purposes. You are responsible for:
- Complying with local trading regulations
- Reporting taxes on gains
- Obtaining necessary licenses (if applicable)
- Understanding and accepting all risks

**Recommended:** Consult with:
- Financial advisor
- Tax professional
- Legal counsel (for larger accounts)

---

## 🎯 Success Metrics

Track these metrics to evaluate performance:

| Metric | Target | Warning Level |
|--------|--------|---------------|
| Sharpe Ratio | > 0.5 | < 0.2 |
| Max Drawdown | < 15% | > 20% |
| Win Rate | > 50% | < 45% |
| Daily P&L Volatility | < 1% | > 2% |
| Execution Slippage | < 1 bp | > 3 bp |

---

## 📚 Additional Resources

### Recommended Reading
1. "Algorithmic Trading" by Ernest Chan
2. "Quantitative Trading" by Ernest Chan
3. "Inside the Black Box" by Rishi Narang

### Online Courses
1. Coursera: Machine Learning for Trading
2. QuantInsti: Algo Trading courses
3. CFI: Python for Finance

---

## ✅ Pre-Launch Checklist

Before activating live trading:

- [ ] Paper traded successfully for 3+ months
- [ ] All API keys are valid and tested
- [ ] Risk limits are properly configured
- [ ] Emergency stop procedures documented
- [ ] Backup systems are in place
- [ ] Performance monitoring is working
- [ ] You understand all code functionality
- [ ] You can afford to lose all capital
- [ ] You have read all documentation
- [ ] You have tested emergency scenarios

---

**Good luck, and trade responsibly!** 🚀

---

*Last Updated: November 5, 2025*
*Version: 1.0*
*Author: Deniel Nankov*
