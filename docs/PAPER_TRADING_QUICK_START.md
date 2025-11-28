# 🚀 QUICK START: Paper Trading in 5 Minutes

**Goal**: Start ML FX paper trading system NOW  
**Time**: 5 minutes  
**Result**: Live trading simulation with EUR + CHF

---

## ✅ Step 1: Set OANDA Credentials (2 min)

Edit your `.env` file:
```bash
nano .env
```

Add these lines:
```bash
OANDA_ACCOUNT_ID=your_practice_account_id_here
OANDA_API_KEY=your_practice_api_key_here
```

**Don't have OANDA credentials?**  
- See `HOW_TO_GET_OANDA_TOKEN.md` for instructions
- Or skip for now (will run in simulation mode)

---

## ✅ Step 2: Validate Setup (1 min)

```bash
source venv_fx/bin/activate
python setup_paper_trading.py
```

**Expected output**:
```
✅ EUR models ready
✅ CHF models ready
✅ Position sizing consistent
✅ Cash reserve configured
✅ PAPER TRADING READY TO START  ← You want this!
```

---

## ✅ Step 3: Start Paper Trading (30 sec)

```bash
python paper_trading_system.py &
```

**What happens**:
- Loads EUR + CHF ML models
- Generates daily trading signals
- Calculates optimal positions
- Logs all activity to `paper_trading.log`
- Updates every 24 hours

---

## ✅ Step 4: Monitor Performance (1 min)

**View dashboard**:
```bash
python monitoring_dashboard.py
```

**Check logs**:
```bash
tail -f paper_trading.log
```

**Open HTML report**:
```bash
open ml_performance_report.html
```

---

## 📊 What You'll See

### **Console Dashboard**:
```
📊 ML FX TRADING PERFORMANCE DASHBOARD
⏰ Generated: 2025-11-06 20:03:15

💰 PERFORMANCE SUMMARY
📈 Returns:
   Total Return:              2.30%
   Annual Return:            12.50%
   Total Profit:          $   2,300

📊 Risk Metrics:
   Sharpe Ratio:              0.82
   Max Drawdown:             -1.50%
   Win Rate:                 54.00%

🎯 vs TARGET METRICS
   ✅ Sharpe Ratio:      0.82 vs  0.70
   ✅ Annual Return:    12.5% vs 10.0%
   ✅ Max Drawdown:     -1.5% vs -18.0%
   ✅ Win Rate:         54.0% vs 52.0%

✅ STRATEGY PERFORMING WELL
```

### **Trading Signals** (in log):
```json
{
  "timestamp": "2025-11-06T20:00:00",
  "signals": {
    "EUR": 0.45,
    "CHF": 0.22
  },
  "positions": {
    "EUR": 45000,
    "CHF": 22000
  }
}
```

---

## 🎯 Success Metrics (Week 1)

**Check these daily**:
- [ ] Sharpe ratio > 0.5
- [ ] Positive cumulative return
- [ ] Max drawdown < 5%
- [ ] No system errors

**If all ✅**: Continue to Week 2  
**If any ❌**: Review `ML_INTEGRATION_GUIDE.md`

---

## 🆘 Troubleshooting

### "No models found"
```bash
python train_ml_models.py
# Wait 63 seconds, then retry
```

### "No OANDA credentials"
```bash
# It's OK! System will run in simulation mode
# Just won't execute real trades (perfect for testing)
```

### "No data in dashboard"
```bash
# Normal on first run
# Check again after 24 hours
# Or manually create test data:
python create_test_performance.py
```

---

## 📅 Daily Routine (2 minutes)

**Every morning**:
```bash
# 1. Check dashboard
python monitoring_dashboard.py

# 2. View report
open ml_performance_report.html

# 3. Check logs
tail -20 paper_trading.log
```

**That's it!** The system handles everything else.

---

## 🎉 You're Done!

Your ML FX trading system is now running in paper mode.

**Expected performance**:
- Daily updates every 24 hours
- Sharpe ratio: 0.7-0.9
- Annual return: 8-12%
- Max drawdown: <15%

**Next milestone**: Week 1 performance review

---

*Quick Start Guide - November 6, 2025*  
*For detailed info, see: `THREE_STEPS_COMPLETE.md`*
