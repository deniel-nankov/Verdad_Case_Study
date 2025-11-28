#!/usr/bin/env python3
"""
FINAL ATTEMPT - Test Original Model on Multiple Currency Pairs
===============================================================
The original EUR/USD model works (+1.41% / 0.896 Sharpe).
Let's test if the SAME approach works better on other pairs.

Hypothesis: EUR/USD is too efficient. Less liquid pairs may have more alpha.

Testing: GBP/USD, AUD/USD, USD/JPY, USD/CAD, NZD/USD
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print('='*100)
print('🌍 MULTI-CURRENCY TEST - Original Model on Different Pairs')
print('='*100)
print()

# Currency pairs to test
pairs = {
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/JPY': 'JPY=X',
    'USD/CAD': 'CAD=X',
    'NZD/USD': 'NZDUSD=X',
    'USD/CHF': 'CHF=X'
}

print(f'📊 Testing {len(pairs)} currency pairs with the ORIGINAL successful model')
print()

def create_features(df):
    """Create exact same features as original successful model"""
    features = pd.DataFrame(index=df.index)
    
    # Momentum
    for window in [5, 10, 21, 42, 63]:
        features[f'mom_{window}'] = df['Close'].pct_change(window)
    
    # Volatility
    for window in [5, 10, 21, 42]:
        features[f'vol_{window}'] = df['returns'].rolling(window).std()
    
    # Moving averages
    for window in [5, 10, 21, 42, 63]:
        ma = df['Close'].rolling(window).mean()
        features[f'sma_{window}'] = (df['Close'] - ma) / df['Close']
    
    # Volume
    features['volume_ratio_5'] = df['Volume'] / (df['Volume'].rolling(5).mean() + 1e-8)
    features['volume_ratio_21'] = df['Volume'] / (df['Volume'].rolling(21).mean() + 1e-8)
    features['volume_vol_21'] = df['Volume'].rolling(21).std() / (df['Volume'].rolling(21).mean() + 1e-8)
    
    # Trend
    features['trend_21'] = (df['Close'].rolling(21).mean() - df['Close'].rolling(63).mean()) / df['Close']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Price position
    features['price_pos_21'] = (df['Close'] - df['Close'].rolling(21).min()) / (df['Close'].rolling(21).max() - df['Close'].rolling(21).min() + 1e-8)
    
    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    features['bb_position'] = (df['Close'] - sma_20) / (2 * std_20 + 1e-8)
    features['bb_width'] = (4 * std_20) / (sma_20 + 1e-8)
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    features['macd'] = (ema_12 - ema_26) / df['Close']
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    
    # Volatility ratios
    features['vol_ratio_5_21'] = features['vol_5'] / (features['vol_21'] + 1e-8)
    features['vol_ratio_21_42'] = features['vol_21'] / (features['vol_42'] + 1e-8)
    
    # Momentum divergence
    features['mom_div_5_21'] = features['mom_5'] / (features['mom_21'] + 1e-8)
    
    return features.fillna(0)

# Results storage
all_results = []

# Test each pair
for pair_name, ticker in pairs.items():
    print(f'\n{"="*100}')
    print(f'💱 Testing {pair_name}')
    print(f'{"="*100}')
    
    try:
        # Load data
        data = yf.download(ticker, start='2018-01-01', progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data['returns'] = data['Close'].pct_change()
        data = data.dropna()
        
        if len(data) < 1000:
            print(f'   ❌ Insufficient data ({len(data)} days)')
            continue
        
        print(f'   ✅ Loaded {len(data)} days')
        
        # Create features
        features = create_features(data)
        
        # Create target (21-day forward return)
        target = data['Close'].pct_change(21).shift(-21)
        
        # Split
        train_mask = data.index < '2024-01-01'
        X_train = features[train_mask].iloc[:-21]
        y_train = target[train_mask].iloc[:-21]
        
        # Remove NaN
        valid = ~y_train.isna()
        X_train = X_train[valid]
        y_train = y_train[valid]
        
        if len(X_train) < 500:
            print(f'   ❌ Insufficient training data ({len(X_train)} samples)')
            continue
        
        # Train EXACT same model as original
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
        
        rf.fit(X_train_scaled, y_train)
        xgb.fit(X_train_scaled, y_train)
        
        print(f'   ✅ Trained on {len(X_train)} samples')
        
        # Test
        X_test = features[~train_mask]
        y_test = target[~train_mask]
        
        valid_test = ~y_test.isna()
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]
        
        X_test_scaled = scaler.transform(X_test)
        
        # Predict (50/50 ensemble)
        predictions = 0.5 * rf.predict(X_test_scaled) + 0.5 * xgb.predict(X_test_scaled)
        
        # Position sizing (exactly like original)
        positions = np.clip(predictions * 10, -1, 1)
        
        # Returns
        strategy_returns = positions * y_test
        
        # Metrics
        total_return = strategy_returns.sum()
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
        
        equity = (1 + strategy_returns).cumprod()
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()
        
        winning = strategy_returns[strategy_returns > 0]
        win_rate = len(winning) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        position_changes = np.abs(np.diff(positions))
        trades = np.sum(position_changes > 0.1)
        
        # Store
        all_results.append({
            'Pair': pair_name,
            'Return': total_return * 100,
            'Sharpe': sharpe,
            'Max_DD': max_dd * 100,
            'Win_Rate': win_rate * 100,
            'Trades': trades,
            'Test_Days': len(y_test)
        })
        
        print(f'   Return: {total_return*100:+.2f}% | Sharpe: {sharpe:+.3f} | DD: {max_dd*100:.2f}% | Trades: {trades}')
        
    except Exception as e:
        print(f'   ❌ Error: {str(e)}')
        continue

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print()
print('='*100)
print('📊 MULTI-CURRENCY RESULTS SUMMARY')
print('='*100)
print()

if len(all_results) > 0:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Sharpe', ascending=False)
    
    print(results_df.to_string(index=False))
    print()
    
    # Best performers
    print('🏆 TOP PERFORMERS:')
    print()
    
    best_sharpe = results_df.iloc[0]
    print(f'   Best Sharpe: {best_sharpe["Pair"]} ({best_sharpe["Sharpe"]:+.3f})')
    
    best_return = results_df.loc[results_df['Return'].idxmax()]
    print(f'   Best Return: {best_return["Pair"]} ({best_return["Return"]:+.2f}%)')
    
    # Compare to EUR/USD original
    eur_result = results_df[results_df['Pair'] == 'EUR/USD']
    if len(eur_result) > 0:
        eur_sharpe = eur_result.iloc[0]['Sharpe']
        eur_return = eur_result.iloc[0]['Return']
        
        print()
        print(f'📊 Comparison to EUR/USD baseline:')
        print(f'   EUR/USD: {eur_return:+.2f}% return, {eur_sharpe:+.3f} Sharpe')
        
        better_pairs = results_df[results_df['Sharpe'] > eur_sharpe]
        if len(better_pairs) > 0:
            print()
            print(f'   ✅ Found {len(better_pairs)} pairs with better Sharpe than EUR/USD!')
            print()
            for _, row in better_pairs.iterrows():
                improvement = row['Sharpe'] - eur_sharpe
                print(f'      {row["Pair"]}: {row["Return"]:+.2f}% ({row["Sharpe"]:+.3f} Sharpe, +{improvement:.3f} vs EUR)')
        else:
            print(f'   ⚠️  No pairs beat EUR/USD Sharpe')
    
    # Statistics
    print()
    print('📈 Portfolio Statistics:')
    print(f'   Average Return: {results_df["Return"].mean():+.2f}%')
    print(f'   Average Sharpe: {results_df["Sharpe"].mean():+.3f}')
    print(f'   Positive Returns: {(results_df["Return"] > 0).sum()} / {len(results_df)}')
    print(f'   Sharpe > 0.5: {(results_df["Sharpe"] > 0.5).sum()} / {len(results_df)}')
    
    # Multi-asset opportunity
    positive_sharpe = results_df[results_df['Sharpe'] > 0.5]
    if len(positive_sharpe) >= 3:
        print()
        print('💡 MULTI-ASSET OPPORTUNITY:')
        print(f'   {len(positive_sharpe)} pairs with Sharpe > 0.5')
        print(f'   Portfolio diversification could improve returns!')
        print()
        print('   Top pairs for portfolio:')
        for _, row in positive_sharpe.head(5).iterrows():
            print(f'      {row["Pair"]:10s}: {row["Return"]:+6.2f}% | Sharpe {row["Sharpe"]:+.3f}')

else:
    print('   ❌ No successful results')

print()
print('='*100)
print('🎯 FINAL RECOMMENDATION')
print('='*100)
print()

if len(all_results) > 0 and len(positive_sharpe) > 0:
    print('✅ SUCCESS PATHWAY IDENTIFIED:')
    print()
    print('   The original ML model DOES work, but performance varies by pair.')
    print(f'   Build a MULTI-CURRENCY PORTFOLIO with the {len(positive_sharpe)} best pairs.')
    print()
    print('   Expected portfolio benefits:')
    print('   • Diversification across currency regimes')
    print('   • Lower correlation = lower drawdowns')
    print('   • Capture alpha from multiple markets')
    print()
    
    # Estimate portfolio performance (assuming 0.5 correlation)
    avg_return = positive_sharpe['Return'].mean()
    avg_sharpe = positive_sharpe['Sharpe'].mean()
    n_assets = len(positive_sharpe)
    
    # Diversification benefit (rough estimate)
    portfolio_sharpe = avg_sharpe * np.sqrt(n_assets / (1 + 0.5 * (n_assets - 1)))
    
    print(f'   Estimated portfolio Sharpe: {portfolio_sharpe:.3f}')
    print(f'   (vs {avg_sharpe:.3f} average single-pair Sharpe)')
else:
    print('📚 KEY INSIGHT:')
    print()
    print('   EUR/USD at +1.41% / 0.896 Sharpe is genuinely excellent.')
    print('   FX markets are highly efficient - this level of alpha is hard to beat.')
    print()
    print('   To improve further, you need:')
    print('   • Alternative data sources (not just price/volume)')
    print('   • Higher frequency data (intraday patterns)')
    print('   • Order flow / microstructure data')
    print('   • Fundamental factors (rates, growth, sentiment)')

print()
print('='*100)
