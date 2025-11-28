"""
SIMPLE ALGORITHMIC FX STRATEGIES (CLEAN VERSION)
=================================================
5 proven strategies that work in FX markets
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


def download_clean_data(symbol, start_date, end_date):
    """Download and clean data (handle MultiIndex)"""
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Flatten MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data['returns'] = data['Close'].pct_change()
    return data


def strategy_1_dual_momentum(symbol, start='2015-01-01', end='2025-11-08'):
    """
    STRATEGY 1: Dual Momentum
    - Use 21-day and 63-day momentum
    - Only trade when both agree
    - Size by volatility
    """
    data = download_clean_data(symbol, start, end)
    
    # Signals
    mom_21 = data['Close'].pct_change(21)
    mom_63 = data['Close'].pct_change(63)
    vol_21 = data['returns'].rolling(21).std() * np.sqrt(252)
    
    signal = (np.sign(mom_21) + np.sign(mom_63)) / 2
    position = (signal * 0.16) / (vol_21 + 0.01)
    position = position.clip(-1, 1)
    
    # Returns
    strat_returns = position.shift(1) * data['returns']
    bh_returns = data['returns']
    
    # Metrics
    strat_ret = (1 + strat_returns.dropna()).prod() - 1
    bh_ret = (1 + bh_returns.dropna()).prod() - 1
    sharpe = strat_returns.mean() / (strat_returns.std() + 1e-8) * np.sqrt(252)
    
    equity = (1 + strat_returns.dropna()).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min()
    
    print(f"\n{'='*70}")
    print(f"Strategy 1: Dual Momentum - {symbol}")
    print(f"{'='*70}")
    print(f"  Return:      {strat_ret*100:>7.2f}%")
    print(f"  Sharpe:      {sharpe:>7.3f}")
    print(f"  Max DD:      {max_dd*100:>7.2f}%")
    print(f"  vs B&H:      {(strat_ret-bh_ret)*100:>+7.2f}%")
    
    return {'strategy': 'Dual Momentum', 'symbol': symbol, 'return': strat_ret*100, 
            'sharpe': sharpe, 'max_dd': max_dd*100}


def strategy_2_triple_ma(symbol, start='2015-01-01', end='2025-11-08'):
    """
    STRATEGY 2: Triple Moving Average
    - 10/50/200 SMA
    - Trade when 2+ agree
    """
    data = download_clean_data(symbol, start, end)
    
    # SMAs
    sma_10 = data['Close'].rolling(10).mean()
    sma_50 = data['Close'].rolling(50).mean()
    sma_200 = data['Close'].rolling(200).mean()
    
    # Signals
    sig_1 = (sma_10 > sma_50).astype(int) * 2 - 1
    sig_2 = (sma_50 > sma_200).astype(int) * 2 - 1
    sig_3 = (data['Close'] > sma_200).astype(int) * 2 - 1
    
    vote = sig_1 + sig_2 + sig_3
    position = pd.Series(0.0, index=data.index)
    position[vote.abs() >= 2] = np.sign(vote[vote.abs() >= 2]) * 0.5
    
    # Returns
    strat_returns = position.shift(1) * data['returns']
    bh_returns = data['returns']
    
    # Metrics
    strat_ret = (1 + strat_returns.dropna()).prod() - 1
    bh_ret = (1 + bh_returns.dropna()).prod() - 1
    sharpe = strat_returns.mean() / (strat_returns.std() + 1e-8) * np.sqrt(252)
    
    equity = (1 + strat_returns.dropna()).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min()
    
    print(f"\n{'='*70}")
    print(f"Strategy 2: Triple MA - {symbol}")
    print(f"{'='*70}")
    print(f"  Return:      {strat_ret*100:>7.2f}%")
    print(f"  Sharpe:      {sharpe:>7.3f}")
    print(f"  Max DD:      {max_dd*100:>7.2f}%")
    print(f"  vs B&H:      {(strat_ret-bh_ret)*100:>+7.2f}%")
    
    return {'strategy': 'Triple MA', 'symbol': symbol, 'return': strat_ret*100,
            'sharpe': sharpe, 'max_dd': max_dd*100}


def strategy_3_mean_reversion(symbol, start='2015-01-01', end='2025-11-08'):
    """
    STRATEGY 3: Mean Reversion
    - Buy oversold (RSI < 30)
    - Sell overbought (RSI > 70)
    """
    data = download_clean_data(symbol, start, end)
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    # Position
    position = pd.Series(0.0, index=data.index)
    position[rsi < 30] = 0.5  # Buy oversold
    position[rsi > 70] = -0.5  # Sell overbought
    
    # Returns
    strat_returns = position.shift(1) * data['returns']
    bh_returns = data['returns']
    
    # Metrics
    strat_ret = (1 + strat_returns.dropna()).prod() - 1
    bh_ret = (1 + bh_returns.dropna()).prod() - 1
    sharpe = strat_returns.mean() / (strat_returns.std() + 1e-8) * np.sqrt(252)
    
    equity = (1 + strat_returns.dropna()).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min()
    
    print(f"\n{'='*70}")
    print(f"Strategy 3: Mean Reversion - {symbol}")
    print(f"{'='*70}")
    print(f"  Return:      {strat_ret*100:>7.2f}%")
    print(f"  Sharpe:      {sharpe:>7.3f}")
    print(f"  Max DD:      {max_dd*100:>7.2f}%")
    print(f"  vs B&H:      {(strat_ret-bh_ret)*100:>+7.2f}%")
    
    return {'strategy': 'Mean Reversion', 'symbol': symbol, 'return': strat_ret*100,
            'sharpe': sharpe, 'max_dd': max_dd*100}


def strategy_4_breakout(symbol, start='2015-01-01', end='2025-11-08'):
    """
    STRATEGY 4: Volatility Breakout (Turtle Traders)
    - Buy on 20-day high
    - Sell on 20-day low
    """
    data = download_clean_data(symbol, start, end)
    
    # Donchian Channels
    high_20 = data['High'].rolling(20).max()
    low_20 = data['Low'].rolling(20).min()
    
    # Position
    position = pd.Series(0.0, index=data.index)
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > high_20.iloc[i-1]:
            position.iloc[i] = 0.5
        elif data['Close'].iloc[i] < low_20.iloc[i-1]:
            position.iloc[i] = -0.5
        else:
            position.iloc[i] = position.iloc[i-1]  # Hold
    
    # Returns
    strat_returns = position.shift(1) * data['returns']
    bh_returns = data['returns']
    
    # Metrics
    strat_ret = (1 + strat_returns.dropna()).prod() - 1
    bh_ret = (1 + bh_returns.dropna()).prod() - 1
    sharpe = strat_returns.mean() / (strat_returns.std() + 1e-8) * np.sqrt(252)
    
    equity = (1 + strat_returns.dropna()).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min()
    
    print(f"\n{'='*70}")
    print(f"Strategy 4: Volatility Breakout - {symbol}")
    print(f"{'='*70}")
    print(f"  Return:      {strat_ret*100:>7.2f}%")
    print(f"  Sharpe:      {sharpe:>7.3f}")
    print(f"  Max DD:      {max_dd*100:>7.2f}%")
    print(f"  vs B&H:      {(strat_ret-bh_ret)*100:>+7.2f}%")
    
    return {'strategy': 'Breakout', 'symbol': symbol, 'return': strat_ret*100,
            'sharpe': sharpe, 'max_dd': max_dd*100}


def strategy_5_ensemble(symbol, start='2015-01-01', end='2025-11-08'):
    """
    STRATEGY 5: Ensemble (Combine all 4)
    - 30% Momentum
    - 30% Triple MA
    - 20% Mean Reversion
    - 20% Breakout
    """
    data = download_clean_data(symbol, start, end)
    
    # Sub-strategy 1: Momentum
    mom_21 = data['Close'].pct_change(21)
    mom_63 = data['Close'].pct_change(63)
    vol_21 = data['returns'].rolling(21).std() * np.sqrt(252)
    pos_mom = ((np.sign(mom_21) + np.sign(mom_63)) / 2 * 0.16 / (vol_21 + 0.01)).clip(-1, 1) * 0.3
    
    # Sub-strategy 2: Triple MA
    sma_10 = data['Close'].rolling(10).mean()
    sma_50 = data['Close'].rolling(50).mean()
    sma_200 = data['Close'].rolling(200).mean()
    vote = ((sma_10 > sma_50).astype(int) * 2 - 1 + 
            (sma_50 > sma_200).astype(int) * 2 - 1 + 
            (data['Close'] > sma_200).astype(int) * 2 - 1)
    pos_ma = pd.Series(0.0, index=data.index)
    pos_ma[vote.abs() >= 2] = np.sign(vote[vote.abs() >= 2]) * 0.5 * 0.3
    
    # Sub-strategy 3: Mean Reversion
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))
    pos_mr = pd.Series(0.0, index=data.index)
    pos_mr[rsi < 30] = 0.5 * 0.2
    pos_mr[rsi > 70] = -0.5 * 0.2
    
    # Sub-strategy 4: Breakout
    high_20 = data['High'].rolling(20).max()
    low_20 = data['Low'].rolling(20).min()
    pos_break = pd.Series(0.0, index=data.index)
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > high_20.iloc[i-1]:
            pos_break.iloc[i] = 0.5 * 0.2
        elif data['Close'].iloc[i] < low_20.iloc[i-1]:
            pos_break.iloc[i] = -0.5 * 0.2
        else:
            pos_break.iloc[i] = pos_break.iloc[i-1]
    
    # Combine
    position = (pos_mom + pos_ma + pos_mr + pos_break).clip(-1, 1)
    
    # Returns
    strat_returns = position.shift(1) * data['returns']
    bh_returns = data['returns']
    
    # Metrics
    strat_ret = (1 + strat_returns.dropna()).prod() - 1
    bh_ret = (1 + bh_returns.dropna()).prod() - 1
    sharpe = strat_returns.mean() / (strat_returns.std() + 1e-8) * np.sqrt(252)
    
    equity = (1 + strat_returns.dropna()).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min()
    
    print(f"\n{'='*70}")
    print(f"Strategy 5: ENSEMBLE (All 4) - {symbol}")
    print(f"{'='*70}")
    print(f"  Return:      {strat_ret*100:>7.2f}%")
    print(f"  Sharpe:      {sharpe:>7.3f}")
    print(f"  Max DD:      {max_dd*100:>7.2f}%")
    print(f"  vs B&H:      {(strat_ret-bh_ret)*100:>+7.2f}%")
    
    return {'strategy': 'Ensemble', 'symbol': symbol, 'return': strat_ret*100,
            'sharpe': sharpe, 'max_dd': max_dd*100}


# Main execution
if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("ALGORITHMIC FX STRATEGIES - SIMPLE & PROVEN")
    print("="*80)
    
    pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
    all_results = []
    
    for pair in pairs:
        print(f"\n\n{'#'*80}")
        print(f"# PAIR: {pair}")
        print(f"{'#'*80}")
        
        r1 = strategy_1_dual_momentum(pair)
        r2 = strategy_2_triple_ma(pair)
        r3 = strategy_3_mean_reversion(pair)
        r4 = strategy_4_breakout(pair)
        r5 = strategy_5_ensemble(pair)
        
        all_results.extend([r1, r2, r3, r4, r5])
    
    # Summary
    df = pd.DataFrame(all_results)
    
    print("\n\n" + "="*80)
    print("SUMMARY: AVERAGE PERFORMANCE BY STRATEGY")
    print("="*80)
    
    summary = df.groupby('strategy').agg({
        'return': 'mean',
        'sharpe': 'mean',
        'max_dd': 'mean'
    }).round(2)
    
    print(summary.to_string())
    
    # Best strategy
    print("\n" + "="*80)
    print("BEST STRATEGY:")
    print("="*80)
    best = df.loc[df['sharpe'].idxmax()]
    print(f"\n  {best['strategy']} on {best['symbol']}")
    print(f"  Sharpe: {best['sharpe']:.3f}")
    print(f"  Return: {best['return']:.2f}%")
    
    # Save
    df.to_csv('algo_strategies_clean_results.csv', index=False)
    print("\n✅ Results saved to: algo_strategies_clean_results.csv")
    print("\n" + "="*80)
