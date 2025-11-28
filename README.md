# Verdad FX Carry Strategy - Technical Case Study

## 🎯 Overview

A comprehensive, production-ready FX carry trading system with machine learning, deep reinforcement learning, and live trading capabilities.

**NEW STRUCTURE**: This repository has been reorganized into a clean, modular architecture for better maintainability and scalability.

## 📁 Project Structure

```
Verdad_Technical_Case_Study/
├── src/                          # Main source code
│   ├── core/                     # Core trading system
│   │   ├── data_feeds.py        # Data feed implementations
│   │   ├── broker_integrations.py
│   │   └── risk_management.py
│   ├── strategies/               # Trading strategies
│   ├── ml/                       # Machine learning models
│   ├── factors/                  # Factor implementations
│   ├── monitoring/               # Monitoring & alerts
│   └── utils/                    # Utilities
├── scripts/                      # Executable scripts
│   ├── backtesting/             # Backtest scripts
│   ├── training/                # ML/DRL training
│   ├── live/                    # Live trading
│   └── data/                    # Data management
├── tests/                        # Test files
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
├── data/                         # Data files
│   ├── raw/                     # Original data
│   ├── external/                # External data sources
│   └── cache/                   # Cached data
├── config/                       # Configuration files
├── models/                       # Trained models
│   ├── ml/                      # ML models
│   └── drl/                     # DRL models
├── results/                      # Backtest results
│   ├── charts/                  # Performance charts
│   ├── backtests/               # Backtest CSVs
│   └── equity_curves/           # Equity curves
└── archive/                      # Archived/legacy code

```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd Verdad_Technical_Case_Study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp config/.env.template .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

**Required API Keys:**
- **Alpha Vantage**: Get free key at https://www.alphavantage.co/
- **FRED**: Get free key at https://fred.stlouisfed.org/
- **OANDA** (optional, for live trading): https://www.oanda.com/

### 3. Run Analysis

```bash
# Open Jupyter notebooks
jupyter notebook notebooks/fx_carry_analysis.ipynb

# Or run backtests
python scripts/backtesting/run_backtest.py --strategy baseline
```

### 4. Paper Trading (Safe Testing)

```bash
# Populate FX cache first
python scripts/data/populate_cache.py

# Run paper trading
python scripts/live/run_paper_trading.py
```

## 📊 Key Features

### Research & Analysis
- **Phase 1**: Baseline FX carry strategy analysis
- **Phase 2**: Advanced multi-factor models with ML
- **Backtesting**: Comprehensive backtesting framework
- **Performance**: Out-of-sample Sharpe ratio +0.178

### Live Trading System
- ✅ Real-time data feeds (Alpha Vantage, OANDA, cached)
- ✅ Multi-broker support (OANDA, IB, Alpaca, Paper)
- ✅ Risk management (position limits, stop-loss, drawdown controls)
- ✅ Performance monitoring dashboard
- ✅ Alert system (Email, Slack, SMS)

### Machine Learning
- Random Forest & Gradient Boosting models
- Feature engineering with 51+ macro indicators
- Regime prediction and signal filtering
- Walk-forward validation

### Deep Reinforcement Learning
- Probabilistic DDPG implementation
- Multi-currency training
- Adaptive position sizing

## 📖 Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 5 minutes
- **[Live Trading Setup](docs/LIVE_TRADING_SETUP.md)** - Production deployment guide
- **[Data Sources Guide](docs/guides/DATA_SOURCES_GUIDE.md)** - API integration
- **[Backtest Results](docs/results/)** - Historical performance analysis

## 🔧 Core Modules

### Data Feeds (`src/core/data_feeds.py`)

```python
from src.core.data_feeds import create_data_feed

# Cached feed (works offline, no API needed)
feed = create_data_feed('cached')

# Real-time feeds
feed = create_data_feed('alphavantage', api_key='YOUR_KEY')
feed = create_data_feed('oanda', api_key='KEY', account_id='ID')
```

### Risk Management (`src/core/risk_management.py`)

```python
from src.core.risk_management import RiskManager, RiskLimits

limits = RiskLimits(
    max_position_size=0.3,
    max_drawdown_pct=0.15,
    stop_loss_pct=0.05
)
risk_mgr = RiskManager(limits, initial_capital=100000)
```

### Configuration (`src/utils/config_loader.py`)

```python
from src.utils.config_loader import load_config, get_api_key

config = load_config()
api_key = get_api_key('alphavantage')
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific components
python tests/test_data_feeds.py
python tests/test_brokers.py
```

## 📈 Running Backtests

```bash
# Baseline strategy
python scripts/backtesting/run_backtest.py --strategy baseline

# Custom date range
python scripts/backtesting/run_backtest.py \
  --strategy baseline \
  --start-date 2010-01-01 \
  --end-date 2023-12-31

# All strategies
python scripts/backtesting/run_backtest.py --strategy all
```

## 🎓 Training ML Models

```bash
# Train ML models (consolidated script - coming soon)
python scripts/training/train_ml.py --currencies EUR CHF

# Train DRL models (consolidated script - coming soon)
python scripts/training/train_drl.py --algorithm ddpg
```

**Note**: Individual training scripts are archived in `archive/training/` for reference.

## ⚠️ Important Notes

### Security
- **Never commit `.env` file** - it contains your API keys
- API keys are loaded from environment variables, not hardcoded
- Use paper trading mode for testing before going live

### Data
- Original data: `data/raw/verdad_fx_case_study_data.csv`
- External data cached in `data/external/`
- FX cache: `fx_data_cache.json` (auto-refreshes every 6 hours)

### Legacy Code
- Old backtest files: `archive/backtests/`
- Old training scripts: `archive/training/`
- These are preserved for reference but not actively maintained

## 📊 Performance Summary

| Strategy | Full Sample Sharpe | OOS Sharpe (2016-2025) | Max Drawdown |
|----------|-------------------|------------------------|--------------|
| Baseline | -0.275 | -0.464 | -66.8% |
| **Optimized** | **-0.202** | **+0.178** ✅ | **-20.98%** |
| ML-Filtered | -0.352 | -0.508 | -63.3% |

**Key Finding**: Portfolio optimization with transaction costs achieves positive out-of-sample Sharpe ratio.

## 🤝 Contributing

This is a technical case study project. For questions or issues:
1. Check documentation in `docs/`
2. Review archived implementations in `archive/`
3. See notebook comments for methodology

## 📝 License

This is a technical case study for educational and interview purposes.

---

**Good luck with your analysis!** 🚀

For detailed setup instructions, see [docs/QUICK_START.md](docs/QUICK_START.md)
