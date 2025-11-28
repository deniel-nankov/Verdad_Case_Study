# Verdad FX Carry Strategy - Technical Case Study

## Overview
This repository contains a comprehensive analysis of FX carry strategies using daily data from 2000-2025 for 8 major currencies.

**NEW:** 🚀 **Live Trading System** - Production-ready infrastructure for real-time FX carry trading with risk management, multiple broker integrations, and monitoring dashboard.

## Files

### Research & Analysis

### 1. `fx_carry_analysis.ipynb` - Phase 1 Analysis
Complete baseline FX carry strategy analysis with:
- Data loading and preprocessing
- Currency excess returns calculation
- Risk-return analysis
- Monthly rebalanced long-short strategy construction
- Performance metrics (CAGR, Sharpe, Max Drawdown)
- Currency-equity correlation analysis
- Framework for evaluating predictive signals

### 2. `fx_carry_phase2_advanced.ipynb` - Phase 2 Advanced Analysis
Advanced quantitative analysis featuring:
- **Factor Decomposition:** Dollar factor, carry factor, momentum, value
- **Multi-Signal Framework:** IC analysis, signal combination, weight optimization
- **Machine Learning:** Regime prediction using RandomForest & Gradient Boosting
- **Portfolio Optimization:** Mean-variance optimization with transaction costs
- **Trading System:** Backtested 5 strategies with walk-forward validation
- **Results:** Optimized strategy achieves +0.178 Sharpe ratio (out-of-sample)

### 3. Live Trading System (Production-Ready) 🚀

#### `live_trading_system.py` - Core Trading Infrastructure
Production trading framework with:
- Real-time data feeds (Alpha Vantage, FRED API)
- Broker API interfaces (OANDA, Interactive Brokers, Alpaca, Paper Trading)
- Risk management system (position limits, drawdown controls, stop-loss)
- Order execution with slippage modeling
- Performance tracking and logging
- Emergency liquidation capabilities

#### `broker_integrations.py` - Multi-Broker Support
Unified broker interface supporting:
- **OANDA:** Best for FX trading (recommended)
- **Interactive Brokers:** Professional platform
- **Alpaca:** Stocks and crypto
- **Paper Trading:** Risk-free testing mode

#### `alert_system.py` - Real-time Notifications
Multi-channel alert system:
- Email notifications (SMTP)
- Slack integration
- SMS alerts via Twilio (critical events)
- Console logging

#### `monitoring_dashboard.py` - Live Performance Dashboard
Real-time monitoring featuring:
- Portfolio value and P&L tracking
- Performance metrics (Sharpe, drawdown, win rate)
- Active positions display
- Risk metrics (VaR, volatility)
- Recent trading activity log
- HTML report export

#### `test_data_feeds.py` - System Validation
Comprehensive testing suite:
- API connection verification
- Data feed validation
- Broker connectivity tests
- Environment variable checks

#### Configuration Files
- `trading_config.json` - System configuration (API keys, risk limits, strategy parameters)
- `requirements_live.txt` - Python dependencies for live trading
- `LIVE_TRADING_SETUP.md` - Complete setup and deployment guide

### Data Files
#### `verdad_fx_case_study_data.csv` - Original Data File
Daily data containing:
- Exchange rates for 8 currencies (AUD, BRL, CAD, CHF, EUR, GBP, JPY, MXN)
- Spot interest rates
- S&P 500 Total Return index
- US Federal Funds Rate

#### Additional Data (yfinance + FRED)
- **51 total columns** of macro and market data
- **yfinance data:** 15 series (equities, bonds, commodities, VIX)
- **FRED data:** 12 series (36 columns) - interest rates, credit spreads, economic indicators
- See `DATA_INTEGRATION_SUMMARY.md` for full list

### Documentation
- `memo_template.md` - Investment memo template for written deliverable
- `LIVE_TRADING_SETUP.md` - Complete setup guide for production trading system
- `DATA_INTEGRATION_SUMMARY.md` - Overview of all 51 data columns
- `DATA_SOURCES_GUIDE.md` - API integration guide (FRED, yfinance)
- `QUICK_REFERENCE.md` - Quick reference for using the analysis

---

## Quick Start

### For Analysis (Backtesting)
```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# 2. Open Phase 1 notebook
jupyter notebook fx_carry_analysis.ipynb

# 3. Or open Phase 2 advanced notebook
jupyter notebook fx_carry_phase2_advanced.ipynb
```

### For Live Trading
```bash
# 1. Install live trading dependencies
pip install -r requirements_live.txt

# 2. Set up environment variables (create .env file)
ALPHAVANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
BROKER_TYPE=paper  # Start with paper trading!

# 3. Test all connections
python test_data_feeds.py

# 4. Start paper trading
python live_trading_system.py

# 5. Monitor in real-time
python monitoring_dashboard.py
```

**⚠️ IMPORTANT:** Read `LIVE_TRADING_SETUP.md` completely before attempting live trading!

---

## Getting Started (Research)

### Step 1: Run the Notebook
```bash
# Open the notebook
jupyter notebook fx_carry_analysis.ipynb

# Or if using VS Code
# Just open the .ipynb file
```

### Step 2: Execute All Cells
Run all cells in order to:
1. Load and parse the data
2. Calculate excess returns
3. Build the carry strategy
4. Generate all visualizations
5. Compute performance metrics

### Step 3: Complete the Memo
Use `memo_template.md` as a guide:
1. Copy key statistics from notebook output
2. Include relevant charts
3. Add your analysis and interpretation
4. Export as PDF for submission

## Key Analysis Components

### 1. Currency Excess Returns
Formula: `Excess Return = ΔFX + (Foreign Rate - US Rate)`

Each currency's excess return captures both appreciation/depreciation and the carry premium.

### 2. Carry Strategy
- **Universe:** 8 currencies
- **Ranking:** By interest rate differential (Foreign - US)
- **Positions:** Long top 3, Short bottom 3
- **Weights:** Equal weight (1/3 per position)
- **Rebalancing:** Monthly

### 3. Performance Metrics
- CAGR (Compound Annual Growth Rate)
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Correlation to Equities

### 4. Risk Analysis
- Distribution analysis (skewness, kurtosis)
- Tail risk assessment
- Drawdown analysis
- Regime-dependent performance

## Key Questions Answered

1. **Is carry risk compensation or free lunch?**
   - Analysis of skewness, kurtosis, and tail risk
   - Comparison of risk-adjusted returns
   - Regime-dependent performance

2. **Strategy Performance**
   - Full period metrics (2000-2025)
   - Crisis period analysis
   - Comparison to equities

3. **Currency-Equity Relationships**
   - Individual currency correlations
   - High-carry vs low-carry patterns
   - Diversification properties

4. **Investment Appeal**
   - Benefits and risks
   - Optimal portfolio allocation
   - Implementation considerations

5. **Predictive Signals**
   - Framework for evaluation
   - Example with equity volatility
   - Statistical testing methodology

## Expected Output

### Deliverables
1. **Jupyter Notebook** with all code and visualizations ✓
2. **Investment Memo** (1-2 pages) - Use template provided
3. **Key Charts** - Already generated in notebook

### Time Estimate
- Notebook execution: 10-15 minutes
- Memo writing: 2-3 hours
- Total: 4-6 hours

## Technical Requirements

### Python Packages
```python
pandas
numpy
matplotlib
seaborn
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn
```

## Tips for Success

1. **Run cells sequentially** - Each cell builds on previous ones
2. **Review all visualizations** - They tell the story
3. **Check summary statistics** - Numbers support conclusions
4. **Compare regimes** - High vol vs low vol performance
5. **Think critically** - What does the data really say?

## Key Insights to Highlight

### In Your Memo

1. **Skewness & Kurtosis**
   - High-carry currencies show negative skew
   - Evidence of crash risk
   - Not a free lunch!

2. **Correlation Patterns**
   - Higher-carry = higher equity correlation
   - Pro-cyclical risk
   - Poor crisis diversification

3. **Performance Attribution**
   - Positive long-term returns
   - Significant drawdowns in crises
   - Sharpe ratio vs equities

4. **Investment Implications**
   - Size appropriately (5-10% allocation)
   - Combine with defensive strategies
   - Active risk management essential

---

## 🎯 Trading System Results

### Backtested Strategies (2000-2025)

| Strategy | Full Sample Sharpe | OOS Sharpe (2016-2025) | Max Drawdown |
|----------|-------------------|------------------------|--------------|
| Baseline | -0.275 | -0.464 | -66.8% |
| **Optimized** | **-0.202** | **+0.178** ✅ | **-20.98%** |
| ML-Filtered | -0.352 | -0.508 | -63.3% |
| Signal-Weighted | -0.334 | -0.483 | -62.9% |
| Ensemble | -0.241 | -0.395 | -44.0% |

**Key Finding:** Portfolio optimization with transaction costs achieves **positive out-of-sample Sharpe ratio (+0.178)** and significantly reduces drawdown vs baseline.

### Live Trading System Features

✅ **Real-time data feeds** - Alpha Vantage + FRED API  
✅ **Multi-broker support** - OANDA, Interactive Brokers, Alpaca, Paper Trading  
✅ **Risk management** - Position limits, stop-loss, drawdown controls  
✅ **Performance tracking** - Real-time P&L, Sharpe, VaR  
✅ **Monitoring dashboard** - Live metrics and alerts  
✅ **Alert system** - Email, Slack, SMS notifications  

### Production-Ready Infrastructure

The live trading system includes:
- `live_trading_system.py` - Core trading engine (745 lines)
- `broker_integrations.py` - Multi-broker API wrapper
- `alert_system.py` - Real-time notification system
- `monitoring_dashboard.py` - Live performance dashboard
- `test_data_feeds.py` - Comprehensive testing suite

**Safety First:** System defaults to paper trading mode. Complete testing required before live deployment.

---

## Common Questions

**Q: Why does the strategy lose money in some periods?**
A: Carry strategies are short volatility - they suffer when risk-off periods cause funding currencies to appreciate.

**Q: Which currencies are typically high-carry?**
A: Historically BRL, MXN, AUD (higher rates, commodity-linked)

**Q: Which are low-carry/safe havens?**
A: JPY, CHF, sometimes EUR (lower rates, flight-to-quality)

**Q: How often does the strategy rebalance?**
A: Monthly, at month-end

**Q: What about transaction costs?**
A: Not explicitly modeled, but mentioned as implementation consideration

## Contact & Support

For questions about the analysis or methodology, refer to:
- Notebook comments and docstrings
- Memo template structure
- Academic literature on carry trade

## License & Attribution

This is a technical case study for interview purposes.

---

**Good luck with your analysis!** 🚀
