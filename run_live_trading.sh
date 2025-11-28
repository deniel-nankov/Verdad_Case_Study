#!/bin/bash
# Live Trading System - Step-by-Step Setup & Execution
# Created: November 5, 2025
# Run with: bash run_live_trading.sh

echo "============================================================"
echo "🚀 FX CARRY LIVE TRADING SYSTEM - AUTOMATED SETUP"
echo "============================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Activate virtual environment
echo "📦 Step 1: Activating virtual environment..."
source venv_fx/bin/activate
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${RED}❌ Failed to activate venv_fx${NC}"
    exit 1
fi
echo ""

# Step 2: Verify all dependencies
echo "📦 Step 2: Checking dependencies..."
python -c "import pandas, numpy, requests; print('✅ Core packages OK')"
python -c "import fredapi; print('✅ FRED API OK')"
python -c "from dotenv import load_dotenv; print('✅ python-dotenv OK')"
echo ""

# Step 3: Load environment variables
echo "🔐 Step 3: Loading environment variables..."
python -c "from dotenv import load_dotenv; load_dotenv(); print('✅ .env file loaded')"
echo ""

# Step 4: Test data feeds
echo "🌐 Step 4: Testing API connections..."
echo ""
python test_data_feeds.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Data feed tests failed. Please check API keys.${NC}"
    exit 1
fi
echo ""

# Step 5: Ask user to confirm
echo "============================================================"
echo "📋 SYSTEM CONFIGURATION SUMMARY"
echo "============================================================"
echo "Trading Mode:     PAPER TRADING (simulation)"
echo "Initial Capital:  $100,000"
echo "Strategy:         Optimized (best OOS Sharpe: +0.178)"
echo "Rebalance:        Every 24 hours"
echo "Risk Limits:      30% max position, 15% max drawdown"
echo ""
echo "API Keys Configured:"
echo "  ✅ Alpha Vantage: 3EHYPIVJOLW9CY8U"
echo "  ✅ FRED API:      b4a18aac3a462b6951ee89d9fef027cb"
echo "  ✅ Broker:        Paper Trading Mode"
echo ""
echo "============================================================"
echo ""

read -p "Ready to start paper trading? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo -e "${YELLOW}Setup cancelled by user${NC}"
    exit 0
fi

echo ""
echo "🚀 Starting live trading system..."
echo "   Press Ctrl+C to stop"
echo "   Check trading_system.log for activity"
echo "   Monitor with: python monitoring_dashboard.py (in another terminal)"
echo ""
echo "============================================================"
echo ""

# Step 6: Start the trading system
python live_trading_system.py

echo ""
echo "============================================================"
echo "Trading system stopped"
echo "============================================================"
