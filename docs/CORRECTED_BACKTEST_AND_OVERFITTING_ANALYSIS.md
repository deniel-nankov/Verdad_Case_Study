# CORRECTED BACKTEST & OVERFITTING ANALYSIS
## Extended Backtest 2010-2025 (FINAL RESULTS)

**Date:** November 8, 2025  
**Status:** ✅ CALCULATION ERROR FIXED, OVERFITTING IDENTIFIED

---

## EXECUTIVE SUMMARY

### Critical Discovery
The original extended backtest had a **fundamental calculation error** that massively overstated returns:
- **Wrong method**: `total_return = returns.sum()` (sums daily returns)
- **Correct method**: `total_return = (1 + returns).cumprod()[-1] - 1` (compounds returns)

### Original (WRONG) Results
- USD/JPY: +16,913% return, 0.450 Sharpe
- USD/CAD: +6,143% return, 0.477 Sharpe

### Corrected (RIGHT) Results  
- **All 7 pairs show SEVERE OVERFITTING**
- **Average Test Sharpe: -0.289 (NEGATIVE)**
- **Strategy LOSES MONEY out-of-sample**

---

## PART 1: CORRECTED BACKTEST RESULTS

### Configuration
- **Period**: 2010-2025 (15 years)
- **Train**: 2010-2020 (11 years, 2,800 days)
- **Test**: 2021-2025 (5 years, 1,242 days)
- **Pairs**: 7 major currency pairs
- **Model**: Random Forest (100 trees, depth 10) + XGBoost (100 trees, depth 6)
- **Features**: 27 enhanced features
- **Calculation**: CORRECTED compounding with overflow protection

### Out-of-Sample Test Results (2021-2025)

| Pair     | Return    | Sharpe  | MaxDD   | Win Rate | Trades | vs B&H     |
|----------|-----------|---------|---------|----------|--------|------------|
| EUR/USD  | -2.31%    | 0.116   | -85.40% | 48.55%   | 600    | +17.43%    |
| GBP/USD  | -87.34%   | -1.135  | -95.74% | 47.99%   | 499    | -68.80%    |
| AUD/USD  | -14.86%   | 0.090   | -95.16% | 48.07%   | 458    | +12.97%    |
| USD/JPY  | -99.33%   | -2.343  | -99.38% | 42.03%   | 538    | -163.93%   |
| USD/CAD  | -21.88%   | -0.117  | -74.36% | 49.76%   | 578    | -56.13%    |
| NZD/USD  | +352.92%  | 0.994   | -76.16% | 51.37%   | 712    | +375.08%   |
| USD/CHF  | +38.43%   | 0.370   | -73.18% | 51.77%   | 566    | +60.49%    |
| **AVERAGE** | **+23.66%** | **-0.289** | **-85.63%** | **48.50%** | **564** | **+25.30%** |

### Performance Summary
- **Profitable pairs**: 2/7 (29%)
- **Sharpe > 1.0**: 0/7 (0%)
- **Sharpe > 0**: 4/7 (57%)
- **Beat Buy & Hold**: 4/7 (57%)

### Key Metrics
- **Average Test Sharpe**: -0.289 ❌ (NEGATIVE)
- **Average Test Return**: +23.66% (skewed by NZD outlier)
- **Average Win Rate**: 48.5% (essentially random)
- **Average MaxDD**: -85.6% (catastrophic)

### Verdict
**❌ STRATEGY FAILS**
- Negative average Sharpe ratio
- Win rate at random level (50%)
- Massive drawdowns (>85%)
- Severe overfitting detected

---

## PART 2: OVERFITTING INVESTIGATION

### The Overfitting Problem

#### Train vs Test Performance

| Metric          | Train       | Test      | Change      |
|-----------------|-------------|-----------|-------------|
| Average Return  | Astronomical| +23.7%    | -99.99%     |
| Average Sharpe  | **16.10**   | **-0.29** | **-16.39**  |
| Average MaxDD   | -13.84%     | -85.63%   | -71.78%     |
| Win Rate        | **93.74%**  | **48.50%**| **-45.23%** |

#### Sharpe Degradation by Pair

| Pair     | Train Sharpe | Test Sharpe | Degradation | Gap    |
|----------|--------------|-------------|-------------|--------|
| EUR/USD  | 16.23        | 0.12        | 99.3%       | 16.11  |
| GBP/USD  | 17.89        | -1.13       | 106.3%      | 19.03  |
| AUD/USD  | 16.81        | 0.09        | 99.5%       | 16.72  |
| USD/JPY  | 14.67        | -2.34       | 116.0%      | 17.01  |
| USD/CAD  | 16.80        | -0.12       | 100.7%      | 16.91  |
| NZD/USD  | 16.50        | 0.99        | 94.0%       | 15.51  |
| USD/CHF  | 13.81        | 0.37        | 97.3%       | 13.44  |
| **AVERAGE** | **16.10** | **-0.29**  | **101.9%**  | **16.39** |

### Root Causes of Overfitting

#### 1. Unrealistic Train Sharpe (RED FLAG 🚩)
- **Normal quant strategies**: Sharpe 0.5-2.0
- **Our training**: Sharpe 13.8-17.9 ← **IMPOSSIBLE**
- **Train win rates**: 92-95% ← **UNSUSTAINABLE**
- **Conclusion**: Model is fitting NOISE, not signal

#### 2. Model Complexity vs Available Data
- **Features**: 27 features per prediction
- **Training samples**: ~2,800 days
- **Model capacity**: RF (100 trees × depth 10) + XGB (100 trees × depth 6)
- **Possible combinations**: ~10^27
- **Problem**: High capacity models memorize training patterns

#### 3. Market Regime Change
- **Train period (2010-2020)**: 
  - Low interest rates
  - Quantitative easing (QE)
  - Low volatility ("Goldilocks economy")
  
- **Test period (2021-2025)**:
  - Rate hikes (0% → 5%+)
  - High inflation
  - Increased volatility
  - Different market dynamics

- **Result**: Patterns learned in calm 2010s don't work in volatile 2020s

#### 4. Potential Feature Leakage
- Using 21-day forward return as target
- Rolling windows might create subtle look-ahead bias
- Some features may inadvertently include future information

### Why This Happens

The model learns **SPURIOUS CORRELATIONS** in training data:

**Example pattern the model "learns":**
```
IF RSI = 65 AND momentum_21 > 0.03 AND volatility_10 < 0.02
THEN market goes up 70% of the time
```

- **In training (2010-2020)**: This pattern exists by random chance
- **In testing (2021-2025)**: This pattern no longer holds
- **Result**: Strategy fails completely

With 27 features and 2,800 samples, there are billions of possible combinations. The model finds combinations that worked historically but were just **RANDOM NOISE**.

### Evidence Summary

| Evidence Type              | Observation                          | Implication          |
|----------------------------|--------------------------------------|----------------------|
| Train Sharpe 16            | 10x higher than realistic            | Fitting noise        |
| Win rate 94% → 49%         | Drops to random (50%)                | No edge              |
| Sharpe degradation 102%    | Complete collapse                    | No generalization    |
| MaxDD -14% → -86%          | 6x worse                             | Catastrophic failure |
| 5 of 7 pairs negative Sharpe| Majority lose money                 | Strategy doesn't work|

---

## HOW TO FIX OVERFITTING

### A. Reduce Model Complexity
```python
# Current (OVERFITS):
RandomForestClassifier(n_estimators=100, max_depth=10)
XGBClassifier(n_estimators=100, max_depth=6)
27 features

# Recommended (SIMPLER):
RandomForestClassifier(n_estimators=30, max_depth=3)
XGBClassifier(n_estimators=30, max_depth=3, reg_alpha=1.0)
5-10 features (only most important)
```

### B. Walk-Forward Validation
Instead of single train/test split:
```
Train: 2010-2014 → Test: 2015
Train: 2011-2015 → Test: 2016
Train: 2012-2016 → Test: 2017
...
Train: 2020-2024 → Test: 2025
```
Ensures model works across **different market regimes**.

### C. Feature Selection
Use only **robust** features:
- ✅ Momentum (5, 21, 63 days)
- ✅ Volatility (simple rolling std)
- ✅ Trend (price vs SMA)
- ❌ Complex derived features (RSI, MACD, Bollinger)
- ❌ Too many moving average combinations

### D. Simpler Strategies
Consider if ML is even needed:
- Simple trend-following (SMA crossover)
- Momentum-only strategy
- Mean reversion
- FX carry trade

Often simpler = better for FX.

### E. Ensemble Across Time
Train multiple models on different periods, average predictions:
```
Model A: Trained on 2010-2015
Model B: Trained on 2013-2018
Model C: Trained on 2016-2021
Final prediction = average(A, B, C)
```

---

## COMPARISON: ML vs BUY & HOLD

### ML Strategy (2021-2025)
- Average Return: +23.7%
- Average Sharpe: **-0.289** ❌
- Win Rate: 48.5%
- Avg Trades: 564 per pair
- Complexity: Very high
- **Result**: Negative Sharpe despite positive return (high volatility)

### Buy & Hold (2021-2025)
- Average Return: -1.6%
- Sharpe: N/A
- Trades: 0
- Complexity: Zero
- **Result**: Loses slightly but no transaction costs

### Winner
**Neither strategy is good**, but:
- ML beats B&H in 4/7 pairs (57%)
- ML has **negative Sharpe** (risk-adjusted loss)
- B&H is simpler (no execution risk)

**Verdict**: The complex ML strategy with 27 features and thousands of trades barely outperforms doing nothing, and on a risk-adjusted basis it **LOSES MONEY**.

---

## FINAL DIAGNOSIS

### Problem
**SEVERE OVERFITTING**

### Cause
Too much model complexity for available signal in FX data

### Evidence
- Train Sharpe: 16.1 (unrealistic)
- Test Sharpe: -0.3 (negative)
- Degradation: 102% (complete failure)
- Win rate: 94% → 49% (collapse to random)

### Mechanism
1. Model has 27 features and high capacity (200 trees total)
2. Finds spurious correlations in 2010-2020 training data
3. Learns noise patterns that worked historically by chance
4. Market regime changes in 2021-2025
5. Learned patterns stop working
6. Strategy fails catastrophically

### Conclusion
**The strategy does NOT have edge in FX markets.** The earlier "excellent" results (+16,913% for JPY, +6,143% for CAD) were artifacts of incorrect return calculation. When calculated correctly, the strategy shows severe overfitting and negative risk-adjusted performance.

---

## RECOMMENDATIONS

### Immediate Actions
1. ✅ **DO NOT TRADE THIS STRATEGY** - it loses money
2. ✅ **Calculation error fixed** - use compounding, not sum
3. ✅ **Overfitting identified** - Train Sharpe 16 is a red flag

### Next Steps
1. **Simplify model**: Use 5 features max, shallow trees (depth 3)
2. **Walk-forward validation**: Test across multiple periods
3. **Lower expectations**: Target Sharpe 0.5-1.5, not 16
4. **Consider alternatives**: Simple trend-following may work better
5. **Different asset class**: FX is hard; try equities or crypto

### Reality Check
- Normal quant strategies: Sharpe 0.5-2.0
- Our train Sharpe: 16.1 ← **IMPOSSIBLE TO SUSTAIN**
- Our test Sharpe: -0.3 ← **REALITY**

When you see unrealistic performance in training, it's almost always overfitting.

---

## FILES CREATED
1. ✅ `backtest_extended_fixed.py` - Corrected backtest with proper compounding
2. ✅ `backtest_comparison.csv` - Complete results for all 7 pairs
3. ✅ `investigate_overfitting.py` - Detailed overfitting analysis
4. ✅ `overfitting_analysis.png` - 4-panel visualization
5. ✅ `analyze_backtest_discrepancy.py` - Shows wrong vs right calculation

---

## LESSONS LEARNED

### 1. Always Verify Calculation Methods
- Sum of returns ≠ Compounded returns
- Small differences compound to huge errors over time
- Always use: `(1 + returns).cumprod()[-1] - 1`

### 2. Train Sharpe > 5 is a RED FLAG
- Realistic quant strategies: Sharpe 0.5-2.0
- Sharpe > 5: Almost certainly overfitting
- Sharpe 16: Definitely overfitting

### 3. Overfitting Signs
- ✅ Unrealistic train performance (Sharpe 16)
- ✅ High train win rate (95%)
- ✅ Complete test failure (Sharpe -0.3)
- ✅ Win rate drops to random (50%)
- ✅ Market regime change (2010s vs 2020s)

### 4. Simpler is Often Better
- Complex models (27 features, 200 trees) overfit easily
- Simple models (3-5 features, shallow trees) generalize better
- For FX, may not need ML at all

### 5. Walk-Forward Validation is Critical
- Single train/test split is not enough
- Must test across multiple time periods
- Ensures robustness to regime changes

---

**Date Generated:** November 8, 2025  
**Status:** ✅ Analysis Complete  
**Verdict:** ❌ Strategy Not Viable (Negative Sharpe, Severe Overfitting)
