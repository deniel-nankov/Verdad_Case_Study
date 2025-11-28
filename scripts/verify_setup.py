#!/usr/bin/env python3
"""
System Verification Script
===========================

Verifies that the FX trading system is properly set up and all
components are working correctly.

Usage:
    python3 scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_imports():
    """Check that all core modules can be imported"""
    print("Checking module imports...")
    
    tests = [
        ("Config Loader", "from src.utils.config_loader import load_config"),
        ("Data Feeds", "from src.core.data_feeds import create_data_feed"),
        ("Risk Management", "from src.core.risk_management import RiskManager, RiskLimits"),
        ("Broker Integrations", "from src.core.broker_integrations import BrokerInterface"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {str(e)}")
            failed += 1
    
    print(f"\nImport Tests: {passed} passed, {failed} failed")
    return failed == 0


def check_directories():
    """Check that all required directories exist"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        "src/core",
        "src/ml",
        "src/factors",
        "src/monitoring",
        "src/utils",
        "scripts/backtesting",
        "scripts/training",
        "scripts/data",
        "tests",
        "notebooks",
        "docs",
        "data/raw",
        "data/external",
        "config",
        "models",
        "results",
    ]
    
    passed = 0
    failed = 0
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}")
            passed += 1
        else:
            print(f"  ❌ {dir_path} (missing)")
            failed += 1
    
    print(f"\nDirectory Tests: {passed} passed, {failed} failed")
    return failed == 0


def check_config():
    """Check configuration files"""
    print("\nChecking configuration...")
    
    config_files = [
        ("config/.env.template", "Environment template"),
        ("config/system_config.json", "System config"),
        ("config/risk_config.json", "Risk config"),
        ("config/strategy_config.json", "Strategy config"),
    ]
    
    passed = 0
    failed = 0
    
    for filepath, name in config_files:
        if Path(filepath).exists():
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name} (missing)")
            failed += 1
    
    # Check if .env exists
    if Path(".env").exists():
        print(f"  ✅ .env file exists")
        passed += 1
    else:
        print(f"  ⚠️  .env file not found (copy from config/.env.template)")
    
    print(f"\nConfig Tests: {passed} passed, {failed} failed")
    return failed == 0


def check_data():
    """Check if data files exist"""
    print("\nChecking data files...")
    
    data_file = Path("data/raw/verdad_fx_case_study_data.csv")
    
    if data_file.exists():
        print(f"  ✅ Main data file exists ({data_file.stat().st_size // 1024} KB)")
        
        # Try to load it
        try:
            import pandas as pd
            df = pd.read_csv(data_file, nrows=5)
            print(f"  ✅ Data file is readable ({len(df.columns)} columns)")
            return True
        except Exception as e:
            print(f"  ❌ Error reading data file: {e}")
            return False
    else:
        print(f"  ❌ Main data file not found: {data_file}")
        return False


def check_dependencies():
    """Check if required Python packages are installed"""
    print("\nChecking Python dependencies...")
    
    required_packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "sklearn",
        "requests",
    ]
    
    passed = 0
    failed = 0
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
            passed += 1
        except ImportError:
            print(f"  ❌ {package} (not installed)")
            failed += 1
    
    print(f"\nDependency Tests: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all verification checks"""
    print("="*60)
    print("FX Trading System - Setup Verification")
    print("="*60)
    print()
    
    results = {
        "Imports": check_imports(),
        "Directories": check_directories(),
        "Configuration": check_config(),
        "Data": check_data(),
        "Dependencies": check_dependencies(),
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_passed = True
    for category, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{category:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run backtests: python3 scripts/backtesting/run_backtest.py")
        print("  2. Open notebooks: jupyter notebook notebooks/")
        print("  3. See README.md for more examples")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Create .env file: cp config/.env.template .env")
        print("  - Check that data files are in data/raw/")
        return 1


if __name__ == '__main__':
    sys.exit(main())
