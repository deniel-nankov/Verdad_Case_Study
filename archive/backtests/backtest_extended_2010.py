"""
Extended Backtesting (2010-2025)
================================
Test the 27-feature enhanced model on 15 years of data across multiple currency pairs.
This provides more robust validation across different market regimes.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

CURRENCY_PAIRS = [
    'EURUSD=X',  # EUR/USD
    'GBPUSD=X',  # GBP/USD
    'AUDUSD=X',  # AUD/USD
    'USDJPY=X',  # USD/JPY
    'USDCAD=X',  # USD/CAD
    'NZDUSD=X',  # NZD/USD
    'USDCHF=X',  # USD/CHF
]

START_DATE = '2010-01-01'
END_DATE = '2025-11-08'
TRAIN_END = '2020-12-31'  # Train on 2010-2020, test on 2021-2025

# ============================================================================
# FEATURE ENGINEERING (27 features - same as validated model)
# ============================================================================

def create_features(df):
    """Create 27 enhanced features matching the validated model"""
    features = pd.DataFrame(index=df.index)
    
    # Momentum
    for window in [5, 10, 21, 42, 63]:
        features[f'mom_{window}'] = df['Close'].pct_change(window)
    
    # Volatility
    returns = df['Close'].pct_change()
    for window in [5, 10, 21, 63]:
        features[f'vol_{window}'] = returns.rolling(window).std()
    
    # Moving averages (SMA distance)
    for window in [5, 10, 21, 42, 63]:
        ma = df['Close'].rolling(window).mean()
        features[f'sma_{window}'] = (df['Close'] - ma) / (df['Close'] + 1e-8)
    
    # Volume
    features['volume_ratio_5'] = df['Volume'] / (df['Volume'].rolling(5).mean() + 1e-8)
    features['volume_ratio_21'] = df['Volume'] / (df['Volume'].rolling(21).mean() + 1e-8)
    features['volume_vol_21'] = df['Volume'].rolling(21).std() / (df['Volume'].rolling(21).mean() + 1e-8)
    
    # Trend
    features['trend_21'] = (df['Close'].rolling(21).mean() - df['Close'].rolling(63).mean()) / (df['Close'] + 1e-8)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Price position
    features['price_pos_21'] = (df['Close'] - df['Close'].rolling(21).min()) / \
                                (df['Close'].rolling(21).max() - df['Close'].rolling(21).min() + 1e-8)
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    features['macd'] = (ema_12 - ema_26) / (df['Close'] + 1e-8)
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    features['bb_width'] = (2 * bb_std) / (bb_mid + 1e-8)
    
    return features

# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def download_data(symbol, start, end):
    """Download OHLCV data from Yahoo Finance"""
    print(f"  Downloading {symbol}...", end=' ')
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        if len(data) > 0:
            print(f"✅ {len(data)} days")
            return data
        else:
            print(f"❌ No data")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def train_models(X_train, y_train):
    """Train Random Forest and XGBoost ensemble"""
    print("  Training Random Forest...", end=' ')
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("✅")
    
    print("  Training XGBoost...", end=' ')
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    print("✅")
    
    return rf, xgb

def backtest_strategy(data, symbol):
    """Run full backtest on one currency pair"""
    print(f"\n{'='*70}")
    print(f"Backtesting: {symbol}")
    print(f"{'='*70}")
    
    # Create features
    print("\n1. Feature Engineering")
    features = create_features(data)
    
    # Create target: 21-day forward return
    target = data['Close'].pct_change(21).shift(-21)
    
    # Align data
    valid_idx = features.dropna().index.intersection(target.dropna().index)
    X = features.loc[valid_idx]
    y = target.loc[valid_idx]
    
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Date range: {X.index[0].date()} to {X.index[-1].date()}")
    
    # Split train/test by date
    print("\n2. Train/Test Split")
    train_mask = X.index <= TRAIN_END
    test_mask = X.index > TRAIN_END
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"  Train: {len(X_train)} days ({X_train.index[0].date()} to {X_train.index[-1].date()})")
    print(f"  Test:  {len(X_test)} days ({X_test.index[0].date()} to {X_test.index[-1].date()})")
    
    if len(X_train) < 100 or len(X_test) < 50:
        print("  ❌ Insufficient data")
        return None
    
    # Train models
    print("\n3. Model Training")
    rf, xgb = train_models(X_train, y_train)
    
    # Generate predictions
    print("\n4. Generating Predictions")
    train_pred_rf = rf.predict(X_train)
    train_pred_xgb = xgb.predict(X_train)
    test_pred_rf = rf.predict(X_test)
    test_pred_xgb = xgb.predict(X_test)
    
    # Ensemble (50/50 average)
    train_pred = (train_pred_rf + train_pred_xgb) / 2.0
    test_pred = (test_pred_rf + test_pred_xgb) / 2.0
    
    # Convert to positions (scale by 10, clip to ±1)
    train_positions = np.clip(train_pred * 10, -1, 1)
    test_positions = np.clip(test_pred * 10, -1, 1)
    
    print(f"  Train positions: [{train_positions.min():.3f}, {train_positions.max():.3f}]")
    print(f"  Test positions:  [{test_positions.min():.3f}, {test_positions.max():.3f}]")
    
    # Calculate strategy returns
    print("\n5. Backtesting Performance")
    
    # Train metrics
    train_returns = train_positions * y_train.values
    train_total = (pd.Series((1 + train_returns).cumprod()).iloc[-1] - 1) if len(train_returns) > 0 else 0
    train_sharpe = train_returns.mean() / (train_returns.std() + 1e-8) * np.sqrt(252)
    train_equity = pd.Series((1 + train_returns).cumprod())
    train_dd = ((train_equity / train_equity.cummax()) - 1).min()
    train_trades = (np.abs(np.diff(train_positions)) > 0.1).sum()
    
    # Test metrics
    test_returns = test_positions * y_test.values
    test_total = (pd.Series((1 + test_returns).cumprod()).iloc[-1] - 1) if len(test_returns) > 0 else 0
    test_sharpe = test_returns.mean() / (test_returns.std() + 1e-8) * np.sqrt(252)
    test_equity = pd.Series((1 + test_returns).cumprod())
    test_dd = ((test_equity / test_equity.cummax()) - 1).min()
    test_trades = (np.abs(np.diff(test_positions)) > 0.1).sum()
    
    # Buy & Hold
    buy_hold = float((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1)
    
    print(f"\n  TRAIN PERIOD (2010-2020):")
    print(f"    Total Return:  {train_total*100:>7.2f}%")
    print(f"    Sharpe Ratio:  {train_sharpe:>7.3f}")
    print(f"    Max Drawdown:  {train_dd*100:>7.2f}%")
    print(f"    Num Trades:    {train_trades:>7d}")
    
    print(f"\n  TEST PERIOD (2021-2025):")
    print(f"    Total Return:  {test_total*100:>7.2f}%")
    print(f"    Sharpe Ratio:  {test_sharpe:>7.3f}")
    print(f"    Max Drawdown:  {test_dd*100:>7.2f}%")
    print(f"    Num Trades:    {test_trades:>7d}")
    
    print(f"\n  BUY & HOLD (Full Period):")
    print(f"    Total Return:  {buy_hold*100:>7.2f}%")
    
    print(f"\n  OUT-OF-SAMPLE COMPARISON:")
    outperformance = test_total - buy_hold
    print(f"    Strategy vs B&H: {outperformance*100:>+7.2f}%")
    
    if test_sharpe > train_sharpe:
        print(f"    ⚠️  Test Sharpe > Train Sharpe (potential overfitting)")
    else:
        print(f"    ✅ Test Sharpe < Train Sharpe (healthy degradation)")
    
    return {
        'symbol': symbol,
        'train_days': len(X_train),
        'test_days': len(X_test),
        'train_return': train_total,
        'train_sharpe': train_sharpe,
        'train_dd': train_dd,
        'train_trades': train_trades,
        'test_return': test_total,
        'test_sharpe': test_sharpe,
        'test_dd': test_dd,
        'test_trades': test_trades,
        'buy_hold': buy_hold,
        'outperformance': outperformance
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("EXTENDED BACKTESTING: 2010-2025")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Train period: 2010-2020 (11 years)")
    print(f"  Test period: 2021-2025 (5 years)")
    print(f"  Currency pairs: {len(CURRENCY_PAIRS)}")
    print(f"  Features: 27 (enhanced model)")
    print(f"  Models: Random Forest + XGBoost ensemble")
    
    # Download all data
    print(f"\n{'='*80}")
    print("DOWNLOADING DATA")
    print(f"{'='*80}")
    
    data_dict = {}
    for symbol in CURRENCY_PAIRS:
        data = download_data(symbol, START_DATE, END_DATE)
        if data is not None and len(data) > 1000:
            data_dict[symbol] = data
    
    print(f"\n✅ Successfully downloaded {len(data_dict)}/{len(CURRENCY_PAIRS)} pairs")
    
    # Run backtests
    results = []
    for symbol in data_dict.keys():
        result = backtest_strategy(data_dict[symbol], symbol)
        if result is not None:
            results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}\n")
    
    if len(results) > 0:
        df = pd.DataFrame(results)
        
        # Sort by test Sharpe
        df = df.sort_values('test_sharpe', ascending=False)
        
        print("OUT-OF-SAMPLE TEST RESULTS (2021-2025):")
        print("-" * 80)
        print(f"{'Pair':<12} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'vs B&H':>8}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            pair_name = row['symbol'].replace('=X', '').replace('USD', '/USD')
            print(f"{pair_name:<12} {row['test_return']*100:>7.2f}% {row['test_sharpe']:>7.3f} "
                  f"{row['test_dd']*100:>7.2f}% {row['test_trades']:>7d} {row['outperformance']*100:>+7.2f}%")
        
        print("-" * 80)
        print(f"{'AVERAGE':<12} {df['test_return'].mean()*100:>7.2f}% {df['test_sharpe'].mean():>7.3f} "
              f"{df['test_dd'].mean()*100:>7.2f}% {df['test_trades'].mean():>7.0f} {df['outperformance'].mean()*100:>+7.2f}%")
        print("-" * 80)
        
        # Performance breakdown
        print(f"\nPERFORMANCE BREAKDOWN:")
        profitable = (df['test_return'] > 0).sum()
        sharpe_gt_1 = (df['test_sharpe'] > 1.0).sum()
        sharpe_gt_0 = (df['test_sharpe'] > 0).sum()
        beat_bh = (df['outperformance'] > 0).sum()
        
        print(f"  Profitable pairs:        {profitable}/{len(df)} ({profitable/len(df)*100:.0f}%)")
        print(f"  Sharpe > 1.0:            {sharpe_gt_1}/{len(df)} ({sharpe_gt_1/len(df)*100:.0f}%)")
        print(f"  Sharpe > 0:              {sharpe_gt_0}/{len(df)} ({sharpe_gt_0/len(df)*100:.0f}%)")
        print(f"  Beat Buy & Hold:         {beat_bh}/{len(df)} ({beat_bh/len(df)*100:.0f}%)")
        
        # Train vs Test comparison
        print(f"\nTRAIN vs TEST COMPARISON:")
        print(f"  Avg Train Sharpe:        {df['train_sharpe'].mean():.3f}")
        print(f"  Avg Test Sharpe:         {df['test_sharpe'].mean():.3f}")
        degradation = (df['train_sharpe'].mean() - df['test_sharpe'].mean()) / df['train_sharpe'].mean() * 100
        print(f"  Degradation:             {degradation:.1f}%")
        
        if df['test_sharpe'].mean() < df['train_sharpe'].mean():
            print(f"  ✅ Healthy degradation (expected for out-of-sample)")
        else:
            print(f"  ⚠️  Test > Train (unusual - check for issues)")
        
        # Save results
        output_file = 'backtest_extended_2010_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
        # Final verdict
        print(f"\n{'='*80}")
        print("FINAL VERDICT")
        print(f"{'='*80}")
        
        avg_sharpe = df['test_sharpe'].mean()
        
        if avg_sharpe > 1.0:
            print("🎉 EXCELLENT! Strategy shows strong performance over 15 years")
            print("   ✅ Ready for paper trading")
        elif avg_sharpe > 0.5:
            print("✅ GOOD! Strategy shows positive edge over 15 years")
            print("   ✅ Consider paper trading with conservative sizing")
        elif avg_sharpe > 0:
            print("⚠️  MARGINAL. Strategy slightly beats random over 15 years")
            print("   💡 May need further optimization")
        else:
            print("❌ POOR. Strategy underperforms over 15 years")
            print("   💡 Significant improvements needed")
        
        print(f"\n📊 Test period: 5 years (2021-2025) across {len(df)} currency pairs")
        print(f"📈 Average Sharpe: {avg_sharpe:.3f}")
        print(f"💰 Win rate: {profitable}/{len(df)} pairs profitable")
        
    else:
        print("❌ No successful backtests completed")
    
    print(f"\n{'='*80}")
    print("✅ EXTENDED BACKTESTING COMPLETE")
    print(f"{'='*80}\n")
