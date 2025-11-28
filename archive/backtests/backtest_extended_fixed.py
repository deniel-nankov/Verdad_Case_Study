"""
CORRECTED Extended Backtest (2010-2025) with Proper Return Calculation
========================================================================
Fixes the calculation error AND handles edge cases properly
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
PAIRS = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDJPY=X', 'USDCAD=X', 'NZDUSD=X', 'USDCHF=X']
START_DATE = '2010-01-01'
END_DATE = '2025-11-08'
TRAIN_CUTOFF = '2020-12-31'

def calculate_metrics(returns):
    """Calculate performance metrics with proper compounding"""
    if len(returns) == 0:
        return 0, 0, 0
    
    # Clip returns to prevent overflow (-50% to +50% per day max)
    returns_clipped = np.clip(returns, -0.5, 0.5)
    
    # Calculate cumulative returns properly
    equity_curve = (1 + returns_clipped).cumprod()
    total_return = equity_curve.iloc[-1] - 1
    
    # Sharpe ratio
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    
    # Max drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    max_dd = drawdown.min()
    
    return total_return, sharpe, max_dd

def create_features(df):
    """Create 27 enhanced features"""
    df = df.copy()
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Price changes
    df['returns'] = df['Close'].pct_change()
    
    # Momentum features
    for period in [5, 10, 21, 42, 63]:
        df[f'momentum_{period}'] = df['Close'].pct_change(period)
    
    # Volatility features
    for period in [5, 10, 21, 63]:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
    
    # Moving averages
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_21'] = df['Close'].rolling(21).mean()
    df['sma_63'] = df['Close'].rolling(63).mean()
    
    # Price position
    df['price_to_sma5'] = df['Close'] / (df['sma_5'] + 1e-8)
    df['price_to_sma21'] = df['Close'] / (df['sma_21'] + 1e-8)
    
    # Volume features
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['volume_sma'] = df['Volume'].rolling(21).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma'] + 1e-8)
    else:
        df['volume_ratio'] = 1.0
    
    # Technical indicators
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_position'] = (df['Close'] - df['bb_middle']) / (bb_std + 1e-8)
    
    # Trend
    df['trend'] = (df['Close'] > df['sma_21']).astype(int)
    
    # Target: 21-day forward return
    df['target'] = df['Close'].pct_change(21).shift(-21)
    
    # Drop NaN
    df = df.dropna()
    
    return df

def backtest_pair(symbol):
    """Backtest a single currency pair"""
    print(f"\n{'='*70}")
    print(f"Backtesting: {symbol}")
    print(f"{'='*70}")
    
    # Download data
    print("\n1. Downloading Data")
    data = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)
    if len(data) < 100:
        print(f"  ❌ Insufficient data")
        return None
    print(f"  ✅ {len(data)} days")
    
    # Create features
    print("\n2. Feature Engineering")
    df = create_features(data)
    print(f"  Total samples: {len(df)}")
    
    # Feature columns (exclude target and non-features)
    feature_cols = [col for col in df.columns if col not in 
                   ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'target', 
                    'sma_5', 'sma_21', 'sma_63', 'bb_middle', 'volume_sma']]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    # Train/test split
    print("\n3. Train/Test Split")
    train_mask = df.index <= TRAIN_CUTOFF
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    
    print(f"  Train: {len(X_train)} days")
    print(f"  Test:  {len(X_test)} days")
    
    if len(X_test) < 50:
        print(f"  ❌ Insufficient test data")
        return None
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n4. Model Training")
    print("  Training Random Forest... ", end="")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, (y_train > 0).astype(int))
    print("✅")
    
    print("  Training XGBoost... ", end="")
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, 
                        random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_train_scaled, (y_train > 0).astype(int))
    print("✅")
    
    # Generate predictions
    print("\n5. Generating Predictions")
    train_pred_rf = rf.predict_proba(X_train_scaled)[:, 1]
    train_pred_xgb = xgb.predict_proba(X_train_scaled)[:, 1]
    test_pred_rf = rf.predict_proba(X_test_scaled)[:, 1]
    test_pred_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
    
    # Ensemble: average of both models
    train_ensemble = (train_pred_rf + train_pred_xgb) / 2
    test_ensemble = (test_pred_rf + test_pred_xgb) / 2
    
    # Convert to positions: scale to [-1, 1] and clip
    train_positions = np.clip((train_ensemble - 0.5) * 10, -1, 1)
    test_positions = np.clip((test_ensemble - 0.5) * 10, -1, 1)
    
    print(f"  Train positions: [{train_positions.min():.3f}, {train_positions.max():.3f}]")
    print(f"  Test positions:  [{test_positions.min():.3f}, {test_positions.max():.3f}]")
    
    # Calculate returns
    print("\n6. Backtesting Performance")
    train_returns = pd.Series(train_positions * y_train)
    test_returns = pd.Series(test_positions * y_test)
    
    # Calculate metrics with proper compounding
    train_total, train_sharpe, train_dd = calculate_metrics(train_returns)
    test_total, test_sharpe, test_dd = calculate_metrics(test_returns)
    
    # Count trades
    train_trades = (np.abs(np.diff(train_positions)) > 0.1).sum()
    test_trades = (np.abs(np.diff(test_positions)) > 0.1).sum()
    
    # Win rate
    train_wins = (train_returns > 0).sum() / len(train_returns) * 100
    test_wins = (test_returns > 0).sum() / len(test_returns) * 100
    
    # Buy & Hold
    buy_hold_return = float((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1)
    
    print(f"\n  TRAIN PERIOD (2010-2020):")
    print(f"    Total Return:  {train_total*100:>8.2f}%")
    print(f"    Sharpe Ratio:  {train_sharpe:>8.3f}")
    print(f"    Max Drawdown:  {train_dd*100:>8.2f}%")
    print(f"    Win Rate:      {train_wins:>8.2f}%")
    print(f"    Num Trades:    {train_trades:>8d}")
    
    print(f"\n  TEST PERIOD (2021-2025):")
    print(f"    Total Return:  {test_total*100:>8.2f}%")
    print(f"    Sharpe Ratio:  {test_sharpe:>8.3f}")
    print(f"    Max Drawdown:  {test_dd*100:>8.2f}%")
    print(f"    Win Rate:      {test_wins:>8.2f}%")
    print(f"    Num Trades:    {test_trades:>8d}")
    
    print(f"\n  BUY & HOLD (Full Period):")
    print(f"    Total Return:  {buy_hold_return*100:>8.2f}%")
    
    outperformance = test_total - buy_hold_return
    print(f"\n  OUT-OF-SAMPLE COMPARISON:")
    print(f"    Strategy vs B&H: {outperformance*100:>+8.2f}%")
    
    # Check for overfitting
    sharpe_degradation = ((train_sharpe - test_sharpe) / (abs(train_sharpe) + 1e-8)) * 100
    if train_sharpe > 5:
        print(f"    ⚠️  Train Sharpe unusually high ({train_sharpe:.2f}) - severe overfitting")
    elif test_sharpe < 0:
        print(f"    ❌ Test Sharpe negative - strategy loses money")
    elif sharpe_degradation > 80:
        print(f"    ⚠️  Sharpe degradation: {sharpe_degradation:.1f}% - overfitting")
    else:
        print(f"    ✅ Sharpe degradation: {sharpe_degradation:.1f}%")
    
    return {
        'Pair': symbol.replace('=X', '').replace('USD', '/USD'),
        'Train_Days': len(X_train),
        'Test_Days': len(X_test),
        'Train_Return': train_total * 100,
        'Train_Sharpe': train_sharpe,
        'Train_MaxDD': train_dd * 100,
        'Train_WinRate': train_wins,
        'Train_Trades': train_trades,
        'Test_Return': test_total * 100,
        'Test_Sharpe': test_sharpe,
        'Test_MaxDD': test_dd * 100,
        'Test_WinRate': test_wins,
        'Test_Trades': test_trades,
        'BuyHold_Return': buy_hold_return * 100,
        'Outperformance': outperformance * 100,
        'Sharpe_Degradation': sharpe_degradation
    }

def main():
    print("="*80)
    print("CORRECTED EXTENDED BACKTESTING: 2010-2025")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Train period: 2010-2020 (11 years)")
    print(f"  Test period: 2021-2025 (5 years)")
    print(f"  Currency pairs: {len(PAIRS)}")
    print(f"  Models: Random Forest + XGBoost ensemble")
    print(f"  Return calculation: CORRECTED (compounding with clipping)")
    print()
    
    results = []
    for pair in PAIRS:
        result = backtest_pair(pair)
        if result:
            results.append(result)
    
    # Create summary
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('backtest_comparison.csv', index=False)
    
    print("\nOUT-OF-SAMPLE TEST RESULTS (2021-2025):")
    print("-"*80)
    print(f"{'Pair':<12} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7}  {'vs B&H':>10}")
    print("-"*80)
    
    for _, row in df_results.iterrows():
        print(f"{row['Pair']:<12} {row['Test_Return']:>9.2f}% {row['Test_Sharpe']:>8.3f} "
              f"{row['Test_MaxDD']:>7.2f}% {row['Test_WinRate']:>7.2f}% {int(row['Test_Trades']):>7d}  "
              f"{row['Outperformance']:>+9.2f}%")
    
    print("-"*80)
    print(f"{'AVERAGE':<12} {df_results['Test_Return'].mean():>9.2f}% "
          f"{df_results['Test_Sharpe'].mean():>8.3f} {df_results['Test_MaxDD'].mean():>7.2f}% "
          f"{df_results['Test_WinRate'].mean():>7.2f}% {int(df_results['Test_Trades'].mean()):>7d}  "
          f"{df_results['Outperformance'].mean():>+9.2f}%")
    print("-"*80)
    
    # Performance breakdown
    print("\nPERFORMANCE BREAKDOWN:")
    profitable = (df_results['Test_Return'] > 0).sum()
    sharpe_gt_1 = (df_results['Test_Sharpe'] > 1.0).sum()
    sharpe_gt_0 = (df_results['Test_Sharpe'] > 0).sum()
    beat_bh = (df_results['Outperformance'] > 0).sum()
    
    print(f"  Profitable pairs:        {profitable}/{len(df_results)} ({profitable/len(df_results)*100:.0f}%)")
    print(f"  Sharpe > 1.0:            {sharpe_gt_1}/{len(df_results)} ({sharpe_gt_1/len(df_results)*100:.0f}%)")
    print(f"  Sharpe > 0:              {sharpe_gt_0}/{len(df_results)} ({sharpe_gt_0/len(df_results)*100:.0f}%)")
    print(f"  Beat Buy & Hold:         {beat_bh}/{len(df_results)} ({beat_bh/len(df_results)*100:.0f}%)")
    
    # Train vs Test
    print("\nTRAIN vs TEST COMPARISON:")
    avg_train_sharpe = df_results['Train_Sharpe'].mean()
    avg_test_sharpe = df_results['Test_Sharpe'].mean()
    avg_degradation = df_results['Sharpe_Degradation'].mean()
    
    print(f"  Avg Train Sharpe:        {avg_train_sharpe:.3f}")
    print(f"  Avg Test Sharpe:         {avg_test_sharpe:.3f}")
    print(f"  Avg Degradation:         {avg_degradation:.1f}%")
    
    if avg_train_sharpe > 5:
        print(f"  ⚠️  Train Sharpe > 5: SEVERE OVERFITTING")
    if avg_test_sharpe < 0:
        print(f"  ❌ Test Sharpe negative: STRATEGY LOSES MONEY")
    elif avg_degradation > 80:
        print(f"  ⚠️  Degradation > 80%: Model doesn't generalize")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if avg_test_sharpe < 0:
        verdict = "❌ FAILS"
        explanation = "Strategy loses money out-of-sample"
    elif avg_test_sharpe < 0.5:
        verdict = "⚠️  WEAK"
        explanation = "Below minimum viable Sharpe (0.5)"
    elif avg_test_sharpe < 1.0:
        verdict = "⚡ MARGINAL"
        explanation = "Positive but modest edge"
    elif avg_test_sharpe < 2.0:
        verdict = "✅ GOOD"
        explanation = "Solid performance"
    else:
        verdict = "🚀 EXCELLENT"
        explanation = "Strong performance"
    
    print(f"{verdict} {explanation}")
    print(f"📊 Test period: 5 years (2021-2025) across {len(df_results)} currency pairs")
    print(f"📈 Average test Sharpe: {avg_test_sharpe:.3f}")
    print(f"💰 Profitable pairs: {profitable}/{len(df_results)}")
    
    print("\n✅ Results saved to: backtest_comparison.csv")
    print("\n" + "="*80)
    print("✅ CORRECTED EXTENDED BACKTESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
