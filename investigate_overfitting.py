"""
OVERFITTING INVESTIGATION
==========================
Why does the strategy show Train Sharpe ~16 but Test Sharpe -0.3?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv('backtest_comparison.csv')

print("="*80)
print("OVERFITTING INVESTIGATION")
print("="*80)
print()

print("1. TRAIN VS TEST PERFORMANCE GAP")
print("-"*80)
print()

for _, row in df.iterrows():
    train_sharpe = row['Train_Sharpe']
    test_sharpe = row['Test_Sharpe']
    degradation = row['Sharpe_Degradation']
    
    print(f"{row['Pair']:<12}")
    print(f"  Train Sharpe: {train_sharpe:>7.2f}")
    print(f"  Test Sharpe:  {test_sharpe:>7.2f}")
    print(f"  Degradation:  {degradation:>7.1f}%")
    print(f"  Gap:          {train_sharpe - test_sharpe:>7.2f} Sharpe points")
    print()

print("="*80)
print()

print("2. ROOT CAUSES OF OVERFITTING")
print("-"*80)
print()

print("A. UNREALISTIC TRAIN SHARPE (16.1 avg)")
print("   • Normal quant strategies: Sharpe 0.5-2.0")
print("   • Our training: Sharpe 13.8-17.9 ← RED FLAG")
print("   • Win rates: 92-95% in training ← IMPOSSIBLE to sustain")
print("   • This indicates the model is fitting NOISE, not signal")
print()

print("B. MODEL COMPLEXITY vs DATA")
print("   • Features: 27 features per prediction")
print("   • Training samples: ~2,800 days")
print("   • Model capacity: RF (100 trees, depth 10) + XGBoost (100 trees, depth 6)")
print("   • Risk: High capacity models can memorize training patterns")
print()

print("C. MARKET REGIME CHANGE")
print("   • Train period: 2010-2020 (low rates, QE, low volatility)")
print("   • Test period: 2021-2025 (rate hikes, inflation, high volatility)")
print("   • Patterns learned in 2010-2020 don't work in 2021-2025")
print()

print("D. FEATURE LEAKAGE (Potential)")
print("   • Using 21-day forward return as target")
print("   • If any feature inadvertently includes future info, this causes overfitting")
print("   • Rolling windows might create subtle look-ahead bias")
print()

print("="*80)
print()

print("3. EVIDENCE OF OVERFITTING")
print("-"*80)
print()

print("Metric                    Train        Test      Change")
print("-"*80)
print(f"Average Return       {df['Train_Return'].mean():>10.1f}%  {df['Test_Return'].mean():>10.1f}%  {df['Train_Return'].mean() - df['Test_Return'].mean():>+10.1f}%")
print(f"Average Sharpe       {df['Train_Sharpe'].mean():>10.2f}   {df['Test_Sharpe'].mean():>10.2f}   {df['Train_Sharpe'].mean() - df['Test_Sharpe'].mean():>+10.2f}")
print(f"Average MaxDD        {df['Train_MaxDD'].mean():>10.2f}%  {df['Test_MaxDD'].mean():>10.2f}%  {df['Test_MaxDD'].mean() - df['Train_MaxDD'].mean():>+10.2f}%")
print(f"Average Win Rate     {df['Train_WinRate'].mean():>10.2f}%  {df['Test_WinRate'].mean():>10.2f}%  {df['Test_WinRate'].mean() - df['Train_WinRate'].mean():>+10.2f}%")
print()

print("Key Observations:")
print("  • Sharpe drops by 16.4 points (101.9% degradation)")
print("  • Win rate drops from 94% to 49% (random = 50%)")
print("  • MaxDD worsens from -11% to -86% (8x worse)")
print("  • Model goes from 'amazing' to 'losing money'")
print()

print("="*80)
print()

print("4. WHY THIS HAPPENS")
print("-"*80)
print()

print("The model learns SPURIOUS correlations in training data:")
print()
print("  Example: 'When RSI is 65 AND momentum_21 > 0.03 AND volatility_10 < 0.02,")
print("           the market goes up 70% of the time'")
print()
print("  In training (2010-2020): This pattern exists by chance")
print("  In testing (2021-2025): This pattern no longer holds")
print()
print("With 27 features and 2,800 samples, there are ~10^27 possible combinations.")
print("The model finds combinations that worked in 2010-2020 but were just NOISE.")
print()

print("="*80)
print()

print("5. HOW TO FIX OVERFITTING")
print("-"*80)
print()

print("A. REDUCE MODEL COMPLEXITY")
print("   ✓ Use fewer features (top 5-10 most important)")
print("   ✓ Reduce tree depth (max_depth=3 instead of 10)")
print("   ✓ Reduce number of trees (30 instead of 100)")
print("   ✓ Increase regularization in XGBoost")
print()

print("B. WALK-FORWARD VALIDATION")
print("   ✓ Instead of single train/test split, use rolling windows")
print("   ✓ Train on Year 1-5, test Year 6")
print("   ✓ Train on Year 2-6, test Year 7")
print("   ✓ Ensures model works across different periods")
print()

print("C. FEATURE SELECTION")
print("   ✓ Use only fundamental features (momentum, trend, vol)")
print("   ✓ Remove complex derived features")
print("   ✓ Test features individually first")
print()

print("D. SIMPLER MODEL")
print("   ✓ Try simple trend-following (SMA crossover)")
print("   ✓ Try momentum-only strategy")
print("   ✓ Complex ML may not be needed for FX")
print()

print("E. ENSEMBLE ACROSS TIME")
print("   ✓ Train multiple models on different time windows")
print("   ✓ Average their predictions")
print("   ✓ Reduces sensitivity to specific period")
print()

print("="*80)
print()

print("6. COMPARISON WITH 'SIMPLE' STRATEGIES")
print("-"*80)
print()

print("Our ML Strategy (Test 2021-2025):")
print(f"  • Average Return: {df['Test_Return'].mean():.2f}%")
print(f"  • Average Sharpe: {df['Test_Sharpe'].mean():.3f}")
print(f"  • Win rate: {df['Test_WinRate'].mean():.1f}%")
print()

print("Buy & Hold (Test 2021-2025):")
print(f"  • Average Return: {df['BuyHold_Return'].mean():.2f}%")
print(f"  • Cost: 0 (no trading)")
print(f"  • Complexity: 0 (just hold)")
print()

profitable = (df['Test_Return'] > df['BuyHold_Return']).sum()
print(f"Strategy beats B&H: {profitable}/7 pairs ({profitable/7*100:.0f}%)")
print()

print("Verdict: The complex ML strategy with 27 features, RF+XGB ensemble,")
print("         and thousands of trades UNDERPERFORMS simple buy & hold")
print("         in 3 out of 7 currency pairs.")
print()

print("="*80)
print()

print("7. FINAL DIAGNOSIS")
print("-"*80)
print()

print("PROBLEM: Severe overfitting")
print("CAUSE:   Too much model complexity for available signal")
print("RESULT:  Train Sharpe 16.1 → Test Sharpe -0.3 (failure)")
print()

print("The strategy memorized noise patterns in 2010-2020 data.")
print("When market regime changed in 2021-2025, those patterns broke.")
print()

print("RECOMMENDATION:")
print("  1. Start with simpler models (3-5 features max)")
print("  2. Use walk-forward validation")
print("  3. Target realistic Sharpe (0.5-1.5, not 16!)")
print("  4. Consider if ML is even needed for FX")
print()

print("="*80)
print()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Train vs Test Sharpe
ax1 = axes[0, 0]
x_pos = np.arange(len(df))
width = 0.35
ax1.bar(x_pos - width/2, df['Train_Sharpe'], width, label='Train', alpha=0.8, color='green')
ax1.bar(x_pos + width/2, df['Test_Sharpe'], width, label='Test', alpha=0.8, color='red')
ax1.set_xlabel('Currency Pair')
ax1.set_ylabel('Sharpe Ratio')
ax1.set_title('Train vs Test Sharpe Ratio (Overfitting Evidence)')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df['Pair'], rotation=45)
ax1.legend()
ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax1.grid(True, alpha=0.3)

# Plot 2: Win Rates
ax2 = axes[0, 1]
ax2.bar(x_pos - width/2, df['Train_WinRate'], width, label='Train', alpha=0.8, color='green')
ax2.bar(x_pos + width/2, df['Test_WinRate'], width, label='Test', alpha=0.8, color='red')
ax2.axhline(y=50, color='black', linestyle='--', linewidth=0.8, label='Random (50%)')
ax2.set_xlabel('Currency Pair')
ax2.set_ylabel('Win Rate (%)')
ax2.set_title('Win Rate: Train (94%) vs Test (48%) - Overfitting!')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df['Pair'], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Sharpe Degradation
ax3 = axes[1, 0]
colors = ['red' if x > 80 else 'orange' if x > 50 else 'green' for x in df['Sharpe_Degradation']]
ax3.barh(df['Pair'], df['Sharpe_Degradation'], color=colors, alpha=0.8)
ax3.axvline(x=50, color='black', linestyle='--', linewidth=0.8, label='50% threshold')
ax3.axvline(x=80, color='red', linestyle='--', linewidth=0.8, label='80% severe')
ax3.set_xlabel('Sharpe Degradation (%)')
ax3.set_title('Sharpe Degradation: >80% indicates severe overfitting')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Returns comparison
ax4 = axes[1, 1]
ax4.bar(x_pos - width/2, df['Test_Return'], width, label='ML Strategy', alpha=0.8, color='blue')
ax4.bar(x_pos + width/2, df['BuyHold_Return'], width, label='Buy & Hold', alpha=0.8, color='gray')
ax4.set_xlabel('Currency Pair')
ax4.set_ylabel('Return (%)')
ax4.set_title('Test Returns: ML Strategy vs Buy & Hold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(df['Pair'], rotation=45)
ax4.legend()
ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("📊 Visualization saved to: overfitting_analysis.png")
print()

print("="*80)
print("✅ OVERFITTING INVESTIGATION COMPLETE")
print("="*80)
