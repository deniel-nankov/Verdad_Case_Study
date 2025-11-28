"""
Period-by-Period Enhancement Analysis
Identify when Cross-Asset (+0.13%) and Timing (+0.14%) enhancements work vs fail

Key Questions:
1. Which market regimes favor cross-asset filtering?
2. When does volatility timing add value?
3. What conditions predict enhancement success?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔬 PERIOD-BY-PERIOD ENHANCEMENT ANALYSIS")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA AND RECREATE BACKTEST
# ============================================================================

print("\n📥 Loading market data...")

eur_usd = yf.download('EURUSD=X', start='2020-01-01', progress=False)
chf_usd = yf.download('CHFUSD=X', start='2020-01-01', progress=False)
spy = yf.download('SPY', start='2020-01-01', progress=False)

eur_usd = eur_usd['Close']
chf_usd = chf_usd['Close']

# Handle multi-level columns for SPY
if isinstance(spy, pd.DataFrame):
    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy['Close'].iloc[:, 0]
    else:
        spy_close = spy['Close']
else:
    spy_close = spy

eur_returns = eur_usd.pct_change().dropna()
chf_returns = chf_usd.pct_change().dropna()
spy_returns = spy_close.pct_change().dropna()

print(f"   EUR/USD: {len(eur_returns)} days")
print(f"   CHF/USD: {len(chf_returns)} days")
print(f"   SPY: {len(spy_returns)} days")

# ============================================================================
# STEP 2: RUN ENHANCED BACKTEST WITH DAILY TRACKING
# ============================================================================

print("\n🔬 Running detailed backtest...")

# Track daily performance for each strategy
results = pd.DataFrame(index=eur_returns.index)
results['baseline_ret'] = 0.0
results['cross_asset_ret'] = 0.0
results['timing_ret'] = 0.0
results['full_ret'] = 0.0

# Track enhancement effects
results['cross_asset_boost'] = 1.0
results['timing_factor'] = 1.0
results['spy_momentum'] = 0.0
results['vol_regime'] = 1.0

for i, date in enumerate(eur_returns.index):
    eur_ret = float(eur_returns.loc[date])
    chf_ret = float(chf_returns.loc[date])
    
    # Baseline return (equal weight, 30% exposure)
    baseline_ret = 0.30 * (0.5 * eur_ret + 0.5 * chf_ret)
    results.loc[date, 'baseline_ret'] = baseline_ret
    
    # Cross-asset enhancement
    if date in spy_returns.index:
        spy_mom = spy_returns.loc[max(spy_returns.index[spy_returns.index <= date][-21:]):date].sum()
        results.loc[date, 'spy_momentum'] = float(spy_mom)
        cross_asset_boost = 1.15 if float(spy_mom) > 0 else 0.85
        results.loc[date, 'cross_asset_boost'] = cross_asset_boost
    else:
        cross_asset_boost = 1.0
    
    cross_asset_ret = baseline_ret * cross_asset_boost
    results.loc[date, 'cross_asset_ret'] = cross_asset_ret
    
    # Volatility timing enhancement
    vol_20 = eur_returns.iloc[max(0, i-20):i+1].std()
    avg_vol = eur_returns.std()
    timing_factor = 1.03 if float(vol_20) > float(avg_vol) else 0.97
    
    results.loc[date, 'vol_regime'] = float(vol_20) / float(avg_vol)
    results.loc[date, 'timing_factor'] = timing_factor
    
    timing_ret = cross_asset_ret * timing_factor
    results.loc[date, 'timing_ret'] = timing_ret
    
    # Full enhancement
    full_ret = timing_ret
    results.loc[date, 'full_ret'] = full_ret
    
    if (i+1) % 250 == 0:
        print(f"   Processed {i+1}/{len(eur_returns)} days...")

# ============================================================================
# STEP 3: CALCULATE ENHANCEMENT EFFECTS
# ============================================================================

print("\n📊 Analyzing enhancement effects...")

# Calculate incremental returns from each enhancement
results['cross_asset_effect'] = results['cross_asset_ret'] - results['baseline_ret']
results['timing_effect'] = results['timing_ret'] - results['cross_asset_ret']
results['total_enhancement'] = results['full_ret'] - results['baseline_ret']

# Cumulative effects
results['cum_baseline'] = (1 + results['baseline_ret']).cumprod()
results['cum_cross_asset'] = (1 + results['cross_asset_ret']).cumprod()
results['cum_timing'] = (1 + results['timing_ret']).cumprod()
results['cum_full'] = (1 + results['full_ret']).cumprod()

# ============================================================================
# STEP 4: IDENTIFY WINNING/LOSING PERIODS
# ============================================================================

print("\n🎯 Identifying winning vs losing periods...")

# Monthly aggregation
results['month'] = pd.to_datetime(results.index).to_period('M')

monthly_stats = results.groupby('month').agg({
    'cross_asset_effect': 'sum',
    'timing_effect': 'sum',
    'total_enhancement': 'sum',
    'spy_momentum': 'mean',
    'vol_regime': 'mean',
    'baseline_ret': ['sum', 'std']
})

monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]

# Classify months
monthly_stats['cross_asset_works'] = monthly_stats['cross_asset_effect_sum'] > 0
monthly_stats['timing_works'] = monthly_stats['timing_effect_sum'] > 0
monthly_stats['enhancement_works'] = monthly_stats['total_enhancement_sum'] > 0

# ============================================================================
# STEP 5: REGIME ANALYSIS
# ============================================================================

print("\n📈 Analyzing market regimes...")

# Define regimes
results['spy_regime'] = pd.cut(results['spy_momentum'], 
                                bins=[-np.inf, -0.02, 0.02, np.inf],
                                labels=['Bear', 'Neutral', 'Bull'])

results['vol_regime_label'] = pd.cut(results['vol_regime'],
                                      bins=[0, 0.8, 1.2, np.inf],
                                      labels=['Low Vol', 'Normal', 'High Vol'])

# Performance by regime
regime_analysis = results.groupby(['spy_regime', 'vol_regime_label']).agg({
    'cross_asset_effect': 'mean',
    'timing_effect': 'mean',
    'total_enhancement': 'mean'
}).reset_index()

print("\n📊 Performance by Market Regime:")
print(regime_analysis.to_string(index=False))

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

print("\n📊 Creating visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# 1. Cumulative Performance
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(results.index, (results['cum_baseline'] - 1) * 100, label='Baseline', linewidth=2)
ax1.plot(results.index, (results['cum_cross_asset'] - 1) * 100, label='+ Cross-Asset', linewidth=2, alpha=0.8)
ax1.plot(results.index, (results['cum_timing'] - 1) * 100, label='+ Timing', linewidth=2, alpha=0.8)
ax1.plot(results.index, (results['cum_full'] - 1) * 100, label='Full Enhanced', linewidth=2.5)
ax1.set_title('Cumulative Returns Over Time', fontsize=14, fontweight='bold')
ax1.set_ylabel('Return (%)')
ax1.legend(loc='best')
ax1.grid(alpha=0.3)

# 2. Cross-Asset Effect Over Time
ax2 = fig.add_subplot(gs[1, 0])
ax2.fill_between(results.index, 0, results['cross_asset_effect'] * 100,
                  where=results['cross_asset_effect'] > 0, color='green', alpha=0.3, label='Positive')
ax2.fill_between(results.index, 0, results['cross_asset_effect'] * 100,
                  where=results['cross_asset_effect'] <= 0, color='red', alpha=0.3, label='Negative')
ax2.set_title('Cross-Asset Enhancement Effect', fontsize=12, fontweight='bold')
ax2.set_ylabel('Daily Effect (%)')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Timing Effect Over Time
ax3 = fig.add_subplot(gs[1, 1])
ax3.fill_between(results.index, 0, results['timing_effect'] * 100,
                  where=results['timing_effect'] > 0, color='green', alpha=0.3, label='Positive')
ax3.fill_between(results.index, 0, results['timing_effect'] * 100,
                  where=results['timing_effect'] <= 0, color='red', alpha=0.3, label='Negative')
ax3.set_title('Volatility Timing Effect', fontsize=12, fontweight='bold')
ax3.set_ylabel('Daily Effect (%)')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.legend()
ax3.grid(alpha=0.3)

# 4. SPY Momentum vs Cross-Asset Performance
ax4 = fig.add_subplot(gs[2, 0])
scatter_data = results[['spy_momentum', 'cross_asset_effect']].dropna()
ax4.scatter(scatter_data['spy_momentum'], scatter_data['cross_asset_effect'] * 100,
            alpha=0.3, s=10)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax4.set_xlabel('SPY 21-Day Momentum')
ax4.set_ylabel('Cross-Asset Effect (%)')
ax4.set_title('SPY Momentum vs Enhancement Effect', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

# 5. Volatility Regime vs Timing Performance
ax5 = fig.add_subplot(gs[2, 1])
scatter_data2 = results[['vol_regime', 'timing_effect']].dropna()
ax5.scatter(scatter_data2['vol_regime'], scatter_data2['timing_effect'] * 100,
            alpha=0.3, s=10)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax5.axvline(x=1, color='red', linestyle='--', linewidth=1)
ax5.set_xlabel('Volatility Regime (20d vol / avg vol)')
ax5.set_ylabel('Timing Effect (%)')
ax5.set_title('Vol Regime vs Timing Effect', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3)

# 6. Monthly Win Rate
ax6 = fig.add_subplot(gs[3, :])
monthly_cum = monthly_stats[['cross_asset_effect_sum', 'timing_effect_sum', 'total_enhancement_sum']].cumsum()
monthly_cum.index = [str(p) for p in monthly_stats.index]
x = range(len(monthly_cum))
width = 0.25

ax6.bar([i - width for i in x], monthly_cum['cross_asset_effect_sum'] * 100, width, 
        label='Cross-Asset', alpha=0.8)
ax6.bar(x, monthly_cum['timing_effect_sum'] * 100, width,
        label='Timing', alpha=0.8)
ax6.bar([i + width for i in x], monthly_cum['total_enhancement_sum'] * 100, width,
        label='Total', alpha=0.8)

ax6.set_xlabel('Month')
ax6.set_ylabel('Cumulative Effect (%)')
ax6.set_title('Monthly Cumulative Enhancement Effects', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(monthly_cum.index, rotation=45, ha='right')
ax6.legend()
ax6.grid(alpha=0.3)
ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('backtest_analysis.png', dpi=150, bbox_inches='tight')
print("   ✅ Chart saved: backtest_analysis.png")

# ============================================================================
# STEP 7: SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("📊 ENHANCEMENT ANALYSIS SUMMARY")
print("="*70)

# Overall statistics
print("\n📈 Overall Performance:")
print(f"   Cross-Asset total effect: {results['cross_asset_effect'].sum()*100:+.2f}%")
print(f"   Timing total effect: {results['timing_effect'].sum()*100:+.2f}%")
print(f"   Total enhancement: {results['total_enhancement'].sum()*100:+.2f}%")

# Win rates
cross_asset_win_rate = (results['cross_asset_effect'] > 0).sum() / len(results)
timing_win_rate = (results['timing_effect'] > 0).sum() / len(results)
total_win_rate = (results['total_enhancement'] > 0).sum() / len(results)

print(f"\n📊 Daily Win Rates:")
print(f"   Cross-Asset: {cross_asset_win_rate:.1%}")
print(f"   Timing: {timing_win_rate:.1%}")
print(f"   Total: {total_win_rate:.1%}")

# Best/Worst months
best_month = monthly_stats['total_enhancement_sum'].idxmax()
worst_month = monthly_stats['total_enhancement_sum'].idxmin()

print(f"\n🏆 Best Month: {best_month}")
print(f"   Total enhancement: {monthly_stats.loc[best_month, 'total_enhancement_sum']*100:+.2f}%")
print(f"   SPY momentum: {monthly_stats.loc[best_month, 'spy_momentum_mean']:.3f}")
print(f"   Vol regime: {monthly_stats.loc[best_month, 'vol_regime_mean']:.2f}x")

print(f"\n📉 Worst Month: {worst_month}")
print(f"   Total enhancement: {monthly_stats.loc[worst_month, 'total_enhancement_sum']*100:+.2f}%")
print(f"   SPY momentum: {monthly_stats.loc[worst_month, 'spy_momentum_mean']:.3f}")
print(f"   Vol regime: {monthly_stats.loc[worst_month, 'vol_regime_mean']:.2f}x")

# Regime-specific performance
print(f"\n🌐 Performance by SPY Regime:")
spy_regime_perf = results.groupby('spy_regime').agg({
    'cross_asset_effect': ['mean', 'sum', lambda x: (x > 0).mean()],
    'timing_effect': ['mean', 'sum'],
    'total_enhancement': ['mean', 'sum']
})
spy_regime_perf.columns = ['CA_Avg', 'CA_Total', 'CA_WinRate', 'Timing_Avg', 'Timing_Total', 'Total_Avg', 'Total_Total']
print(spy_regime_perf.to_string())

print(f"\n💨 Performance by Vol Regime:")
vol_regime_perf = results.groupby('vol_regime_label').agg({
    'cross_asset_effect': ['mean', 'sum'],
    'timing_effect': ['mean', 'sum', lambda x: (x > 0).mean()],
    'total_enhancement': ['mean', 'sum']
})
vol_regime_perf.columns = ['CA_Avg', 'CA_Total', 'Timing_Avg', 'Timing_Total', 'Timing_WinRate', 'Total_Avg', 'Total_Total']
print(vol_regime_perf.to_string())

# ============================================================================
# STEP 8: KEY INSIGHTS
# ============================================================================

print("\n" + "="*70)
print("🔍 KEY INSIGHTS")
print("="*70)

# Cross-asset insights
cross_asset_bull = results[results['spy_regime'] == 'Bull']['cross_asset_effect'].mean()
cross_asset_bear = results[results['spy_regime'] == 'Bear']['cross_asset_effect'].mean()

print("\n📊 Cross-Asset Filter:")
if cross_asset_bull > cross_asset_bear:
    print(f"   ✅ Works best in BULL markets ({cross_asset_bull*100:+.3f}% vs {cross_asset_bear*100:+.3f}%)")
else:
    print(f"   ✅ Works best in BEAR markets ({cross_asset_bear*100:+.3f}% vs {cross_asset_bull*100:+.3f}%)")

# Timing insights  
timing_high_vol = results[results['vol_regime_label'] == 'High Vol']['timing_effect'].mean()
timing_low_vol = results[results['vol_regime_label'] == 'Low Vol']['timing_effect'].mean()

print(f"\n💨 Volatility Timing:")
if timing_high_vol > timing_low_vol:
    print(f"   ✅ Works best in HIGH VOL ({timing_high_vol*100:+.3f}% vs {timing_low_vol*100:+.3f}%)")
else:
    print(f"   ✅ Works best in LOW VOL ({timing_low_vol*100:+.3f}% vs {timing_high_vol*100:+.3f}%)")

print("\n💡 Recommendations:")
print("   1. Use cross-asset filter during trending SPY markets")
print("   2. Increase timing factor during volatility regime shifts")
print("   3. Monitor monthly performance for regime changes")
print("   4. Consider disabling enhancements during low conviction periods")

print("\n✅ Analysis complete! See backtest_analysis.png for visualizations.")
