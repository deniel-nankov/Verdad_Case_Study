#!/usr/bin/env python3
"""
Quick test to diagnose paper trading issues
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("="*70)
print("🔍 DIAGNOSING PAPER TRADING ISSUES")
print("="*70)

# Test 1: Check environment
print("\n1️⃣ Checking environment...")
fred_key = os.getenv('FRED_API_KEY')
print(f"   FRED_API_KEY: {'✅ Set' if fred_key else '❌ Missing'}")

# Test 2: Import ML strategy
print("\n2️⃣ Testing ML strategy import...")
try:
    from ml_fx.ml_strategy import MLFXStrategy
    print("   ✅ MLFXStrategy imported successfully")
except Exception as e:
    print(f"   ❌ Error importing: {e}")
    sys.exit(1)

# Test 3: Initialize strategy
print("\n3️⃣ Initializing strategy...")
try:
    strategy = MLFXStrategy(
        fred_api_key=fred_key,
        currencies=['EUR', 'CHF']
    )
    print("   ✅ Strategy initialized")
except Exception as e:
    print(f"   ❌ Error initializing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load EUR model
print("\n4️⃣ Loading EUR model...")
try:
    strategy.load_models('EUR')
    print("   ✅ EUR model loaded")
except Exception as e:
    print(f"   ❌ Error loading EUR: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Load CHF model
print("\n5️⃣ Loading CHF model...")
try:
    strategy.load_models('CHF')
    print("   ✅ CHF model loaded")
except Exception as e:
    print(f"   ❌ Error loading CHF: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Generate signals (this is where it likely hangs)
print("\n6️⃣ Generating signals (this may take time)...")
try:
    print("   ⏳ Fetching data and generating signals...")
    signals = strategy.generate_signals()
    print(f"   ✅ Signals generated: {signals}")
except Exception as e:
    print(f"   ❌ Error generating signals: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Generate positions
print("\n7️⃣ Generating positions...")
try:
    positions = strategy.generate_positions(
        signals=signals,
        capital=100000,
        max_position_size=0.30,
        risk_scale=1.0
    )
    print(f"   ✅ Positions generated: {positions}")
except Exception as e:
    print(f"   ❌ Error generating positions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - Paper trading should work!")
print("="*70)
