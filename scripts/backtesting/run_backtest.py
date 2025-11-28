#!/usr/bin/env python3
"""
Consolidated Backtest Script for FX Carry Strategies
=====================================================

Run backtests for various FX carry strategies with configurable parameters.

Usage:
    python scripts/backtesting/run_backtest.py --strategy baseline
    python scripts/backtesting/run_backtest.py --strategy all --start-date 2010-01-01
    python scripts/backtesting/run_backtest.py --help

Author: Deniel Nankov
Date: November 27, 2025
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from archived backtest files (we'll use the best one as reference)
# For now, this is a placeholder that shows the structure


def load_data(start_date=None, end_date=None):
    """Load FX carry data"""
    data_path = Path('data/raw/verdad_fx_case_study_data.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    logger.info(f"Loaded data from {df.index[0]} to {df.index[-1]} ({len(df)} days)")
    return df


def run_baseline_strategy(df):
    """Run baseline 3x3 carry strategy"""
    logger.info("Running baseline strategy...")
    
    currencies = ['AUD', 'BRL', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'MXN']
    
    # Calculate excess returns
    excess_returns = {}
    for curr in currencies:
        fx_col = f'{curr}_FX'
        rate_col = f'{curr}_Rate'
        
        if fx_col in df.columns and rate_col in df.columns:
            fx_return = df[fx_col].pct_change()
            carry = (df[rate_col] - df['USD_Rate']) / 252  # Daily carry
            excess_returns[curr] = fx_return + carry
    
    er_df = pd.DataFrame(excess_returns)
    
    # Simple 3x3 strategy
    monthly_returns = []
    for date in er_df.resample('M').last().index:
        month_data = er_df[:date].tail(21)  # Last month
        
        if len(month_data) < 21:
            continue
        
        # Rank by interest rate differential
        rates = {curr: df.loc[date, f'{curr}_Rate'] - df.loc[date, 'USD_Rate'] 
                for curr in currencies if f'{curr}_Rate' in df.columns}
        
        sorted_curr = sorted(rates.items(), key=lambda x: x[1], reverse=True)
        
        long_curr = [c for c, _ in sorted_curr[:3]]
        short_curr = [c for c, _ in sorted_curr[-3:]]
        
        # Calculate next month return
        next_month = er_df[date:].head(21)
        if len(next_month) < 21:
            break
        
        long_ret = next_month[long_curr].mean(axis=1).mean() * (1/3)
        short_ret = -next_month[short_curr].mean(axis=1).mean() * (1/3)
        
        monthly_returns.append({
            'date': date,
            'return': long_ret + short_ret
        })
    
    results_df = pd.DataFrame(monthly_returns)
    results_df['cumulative'] = (1 + results_df['return']).cumprod()
    
    # Calculate metrics
    total_return = results_df['cumulative'].iloc[-1] - 1
    ann_return = (1 + total_return) ** (12 / len(results_df)) - 1
    ann_vol = results_df['return'].std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd = (results_df['cumulative'] / results_df['cumulative'].cummax() - 1).min()
    
    metrics = {
        'strategy': 'Baseline',
        'total_return': total_return,
        'annual_return': ann_return,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }
    
    logger.info(f"Baseline Strategy Results:")
    logger.info(f"  Annual Return: {ann_return*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
    logger.info(f"  Max Drawdown: {max_dd*100:.2f}%")
    
    return results_df, metrics


def save_results(results_df, metrics, output_dir='results/backtests'):
    """Save backtest results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = output_path / f"backtest_{metrics['strategy'].lower()}_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    logger.info(f"Results saved to {csv_file}")
    
    # Save chart
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['date'], results_df['cumulative'])
    plt.title(f"{metrics['strategy']} Strategy - Cumulative Returns")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    
    chart_file = output_path.parent / 'charts' / f"backtest_{metrics['strategy'].lower()}_{timestamp}.png"
    chart_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    logger.info(f"Chart saved to {chart_file}")
    plt.close()
    
    return csv_file, chart_file


def main():
    parser = argparse.ArgumentParser(description='Run FX carry strategy backtests')
    parser.add_argument('--strategy', type=str, default='baseline',
                       choices=['baseline', 'optimized', 'ml', 'multi-factor', 'all'],
                       help='Strategy to backtest')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='results/backtests',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("FX Carry Strategy Backtest")
    logger.info("="*60)
    
    # Load data
    df = load_data(args.start_date, args.end_date)
    
    # Run strategy
    if args.strategy == 'baseline' or args.strategy == 'all':
        results_df, metrics = run_baseline_strategy(df)
        save_results(results_df, metrics, args.output)
    
    if args.strategy == 'all':
        logger.warning("Other strategies not yet implemented in consolidated script")
        logger.info("See archive/backtests/ for individual strategy implementations")
    
    logger.info("="*60)
    logger.info("Backtest complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
