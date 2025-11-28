#!/bin/bash
# Setup script for FX Trading System
# This script sets up the development environment

set -e  # Exit on error

echo "=========================================="
echo "FX Trading System - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Error: Python not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✅ Found Python $PYTHON_VERSION"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "✅ Pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    echo "✅ Dependencies installed"
elif [ -f "requirements_live.txt" ]; then
    pip install -r requirements_live.txt -q
    echo "✅ Live trading dependencies installed"
else
    echo "⚠️  No requirements.txt found, installing basic packages..."
    pip install pandas numpy matplotlib seaborn scikit-learn requests -q
    echo "✅ Basic packages installed"
fi
echo ""

# Setup configuration
echo "Setting up configuration..."
if [ ! -f ".env" ]; then
    if [ -f "config/.env.template" ]; then
        cp config/.env.template .env
        echo "✅ Created .env file from template"
        echo "⚠️  IMPORTANT: Edit .env and add your API keys!"
    else
        echo "⚠️  No .env template found"
    fi
else
    echo "✅ .env file already exists"
fi
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/cache
mkdir -p results/charts results/backtests results/equity_curves
mkdir -p models/ml models/drl
echo "✅ Directories created"
echo ""

# Download initial data (if script exists)
if [ -f "scripts/data/populate_cache.py" ]; then
    echo "Populating FX data cache..."
    $PYTHON_CMD scripts/data/populate_cache.py || echo "⚠️  Cache population failed (you can do this later)"
    echo ""
fi

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Run: source venv/bin/activate (to activate environment)"
echo "3. Run: python3 scripts/verify_setup.py (to verify installation)"
echo "4. See README.md for usage examples"
echo ""
