#!/usr/bin/env python3
"""
ML FX Trading Performance Monitoring Dashboard
Real-time tracking of ML strategy performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

sns.set_style('whitegrid')

class TradingMonitor:
    """Monitor ML trading strategy performance"""
    
    def __init__(self, performance_file='paper_performance.csv', log_file='paper_trading.log'):
        self.performance_file = performance_file
        self.log_file = log_file
        self.performance_data = None
        self.log_data = None
        
    def load_data(self):
        """Load performance and log data"""
        
        # Load performance data
        if os.path.exists(self.performance_file):
            self.performance_data = pd.read_csv(self.performance_file, parse_dates=['date'])
            print(f"✅ Loaded {len(self.performance_data)} days of performance data")
        else:
            print(f"⚠️  Performance file not found: {self.performance_file}")
            self.performance_data = pd.DataFrame()
        
        # Load log data
        if os.path.exists(self.log_file):
            logs = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        logs.append(json.loads(line))
                    except:
                        pass
            
            if logs:
                self.log_data = pd.DataFrame(logs)
                self.log_data['timestamp'] = pd.to_datetime(self.log_data['timestamp'])
                print(f"✅ Loaded {len(self.log_data)} log entries")
            else:
                self.log_data = pd.DataFrame()
        else:
            print(f"⚠️  Log file not found: {self.log_file}")
            self.log_data = pd.DataFrame()
    
    def calculate_metrics(self):
        """Calculate key performance metrics"""
        
        if self.performance_data.empty:
            return None
        
        df = self.performance_data.copy()
        
        # Calculate returns
        if 'capital' in df.columns:
            df['daily_return'] = df['capital'].pct_change()
            
            # Calculate metrics
            total_return = (df['capital'].iloc[-1] - df['capital'].iloc[0]) / df['capital'].iloc[0]
            
            days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
            annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
            
            sharpe = np.sqrt(252) * df['daily_return'].mean() / df['daily_return'].std() if df['daily_return'].std() > 0 else 0
            
            # Max drawdown
            running_max = df['capital'].expanding().max()
            drawdown = (df['capital'] - running_max) / running_max
            max_dd = drawdown.min()
            
            # Win rate
            win_rate = (df['daily_return'] > 0).sum() / len(df['daily_return'].dropna())
            
            # Current drawdown
            current_dd = (df['capital'].iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'current_drawdown': current_dd,
                'win_rate': win_rate,
                'trading_days': len(df),
                'current_capital': df['capital'].iloc[-1],
                'initial_capital': df['capital'].iloc[0],
                'total_profit': df['capital'].iloc[-1] - df['capital'].iloc[0],
                'avg_daily_return': df['daily_return'].mean(),
                'volatility': df['daily_return'].std() * np.sqrt(252),
                'best_day': df['daily_return'].max(),
                'worst_day': df['daily_return'].min()
            }
            
            return metrics
        
        return None
    
    def generate_dashboard(self):
        """Generate comprehensive monitoring dashboard"""
        
        print("\n" + "="*70)
        print("📊 ML FX TRADING PERFORMANCE DASHBOARD")
        print("="*70)
        print(f"\n⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        metrics = self.calculate_metrics()
        
        if metrics is None:
            print("\n⚠️  No performance data available yet")
            print("\n💡 Start paper trading to see metrics:")
            print("   python paper_trading_system.py")
            return
        
        # Performance Summary
        print("\n" + "="*70)
        print("💰 PERFORMANCE SUMMARY")
        print("="*70)
        
        print(f"\n📈 Returns:")
        print(f"   Total Return:        {metrics['total_return']*100:>8.2f}%")
        print(f"   Annual Return:       {metrics['annual_return']*100:>8.2f}%")
        print(f"   Total Profit:        ${metrics['total_profit']:>12,.0f}")
        
        print(f"\n💼 Capital:")
        print(f"   Initial:             ${metrics['initial_capital']:>12,.0f}")
        print(f"   Current:             ${metrics['current_capital']:>12,.0f}")
        
        print(f"\n📊 Risk Metrics:")
        print(f"   Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
        print(f"   Max Drawdown:        {metrics['max_drawdown']*100:>8.2f}%")
        print(f"   Current Drawdown:    {metrics['current_drawdown']*100:>8.2f}%")
        print(f"   Annual Volatility:   {metrics['volatility']*100:>8.2f}%")
        
        print(f"\n📅 Trade Statistics:")
        print(f"   Trading Days:        {metrics['trading_days']:>8d}")
        print(f"   Win Rate:            {metrics['win_rate']*100:>8.2f}%")
        print(f"   Avg Daily Return:    {metrics['avg_daily_return']*100:>8.4f}%")
        print(f"   Best Day:            {metrics['best_day']*100:>8.2f}%")
        print(f"   Worst Day:           {metrics['worst_day']*100:>8.2f}%")
        
        # Target comparison
        print(f"\n" + "="*70)
        print("🎯 vs TARGET METRICS")
        print("="*70)
        
        targets = {
            'Sharpe Ratio': (metrics['sharpe_ratio'], 0.70),
            'Annual Return': (metrics['annual_return']*100, 10.0),
            'Max Drawdown': (metrics['max_drawdown']*100, -18.0),
            'Win Rate': (metrics['win_rate']*100, 52.0)
        }
        
        print()
        for metric_name, (actual, target) in targets.items():
            if metric_name == 'Max Drawdown':
                status = "✅" if actual >= target else "❌"
            else:
                status = "✅" if actual >= target else "❌"
            
            print(f"   {status} {metric_name:15s}: {actual:>7.2f} vs {target:>7.2f}")
        
        # Overall status
        passing = sum([
            metrics['sharpe_ratio'] >= 0.70,
            metrics['annual_return']*100 >= 10.0,
            metrics['max_drawdown']*100 >= -18.0,
            metrics['win_rate']*100 >= 52.0
        ])
        
        print(f"\n📊 Passing Metrics: {passing}/4")
        
        print("\n" + "="*70)
        if passing >= 3:
            print("✅ STRATEGY PERFORMING WELL")
            print("   Continue monitoring and prepare for live trading")
        elif passing >= 2:
            print("⚠️  STRATEGY PERFORMANCE MIXED")
            print("   Monitor closely and consider adjustments")
        else:
            print("❌ STRATEGY UNDERPERFORMING")
            print("   Review signals and consider retraining models")
        print("="*70)
        
        # Create visualizations
        self.create_charts(metrics)
        
        return metrics
    
    def create_charts(self, metrics):
        """Create performance visualization charts"""
        
        if self.performance_data.empty:
            return
        
        df = self.performance_data.copy()
        df['daily_return'] = df['capital'].pct_change()
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['date'], df['capital'], linewidth=2, color='darkblue', label='Equity')
        ax1.axhline(y=df['capital'].iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.fill_between(df['date'], df['capital'].iloc[0], df['capital'], 
                         alpha=0.2, color='green' if metrics['total_return'] > 0 else 'red')
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Capital ($)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2. Cumulative Returns
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.fill_between(df['date'], 0, df['cumulative_return']*100, 
                         alpha=0.3, color='green' if metrics['total_return'] > 0 else 'red')
        ax2.plot(df['date'], df['cumulative_return']*100, linewidth=2, 
                color='darkgreen' if metrics['total_return'] > 0 else 'darkred')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Return (%)')
        ax2.grid(alpha=0.3)
        
        # 3. Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        running_max = df['capital'].expanding().max()
        drawdown = (df['capital'] - running_max) / running_max * 100
        ax3.fill_between(df['date'], 0, drawdown, alpha=0.3, color='red')
        ax3.plot(df['date'], drawdown, linewidth=2, color='darkred')
        ax3.axhline(y=-18, color='orange', linestyle='--', alpha=0.5, label='Target Limit')
        ax3.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Rolling Sharpe (21-day)
        ax4 = fig.add_subplot(gs[2, 0])
        if len(df) >= 21:
            rolling_sharpe = df['daily_return'].rolling(21).mean() / df['daily_return'].rolling(21).std() * np.sqrt(252)
            ax4.plot(df['date'], rolling_sharpe, linewidth=2, color='purple')
            ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target (0.7)')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_title('Rolling Sharpe Ratio (21-day)', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Sharpe Ratio')
            ax4.legend()
            ax4.grid(alpha=0.3)
        
        # 5. Daily Returns Distribution
        ax5 = fig.add_subplot(gs[2, 1])
        returns_clean = df['daily_return'].dropna()
        ax5.hist(returns_clean*100, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax5.axvline(x=returns_clean.mean()*100, color='red', linestyle='--', linewidth=2, label='Mean')
        ax5.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Daily Return (%)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # Add overall metrics text
        fig.text(0.99, 0.01, 
                f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
                f"Return: {metrics['annual_return']*100:.1f}% | "
                f"Max DD: {metrics['max_drawdown']*100:.1f}% | "
                f"Win Rate: {metrics['win_rate']*100:.1f}%",
                ha='right', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig('ml_monitoring_dashboard.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Dashboard saved: ml_monitoring_dashboard.png")
        
        try:
            plt.show()
        except:
            pass  # In case display is not available
    
    def generate_html_report(self, metrics):
        """Generate HTML performance report"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ML FX Trading Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric {{ display: inline-block; margin: 15px; padding: 20px; background: #ecf0f1; border-radius: 8px; min-width: 200px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .status {{ padding: 10px 20px; border-radius: 5px; display: inline-block; margin: 10px 0; }}
        .status-good {{ background: #d4edda; color: #155724; }}
        .status-warning {{ background: #fff3cd; color: #856404; }}
        .status-bad {{ background: #f8d7da; color: #721c24; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        .timestamp {{ color: #7f8c8d; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 ML FX Trading Performance Dashboard</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>💰 Performance Summary</h2>
        <div>
            <div class="metric">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if metrics['total_return'] > 0 else 'negative'}">
                    {metrics['total_return']*100:.2f}%
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Annual Return</div>
                <div class="metric-value {'positive' if metrics['annual_return'] > 0 else 'negative'}">
                    {metrics['annual_return']*100:.2f}%
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value {'positive' if metrics['sharpe_ratio'] > 0.7 else 'negative'}">
                    {metrics['sharpe_ratio']:.2f}
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">
                    {metrics['max_drawdown']*100:.2f}%
                </div>
            </div>
        </div>
        
        <h2>📊 Detailed Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Target</th><th>Status</th></tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{metrics['sharpe_ratio']:.2f}</td>
                <td>0.70</td>
                <td>{'✅ Pass' if metrics['sharpe_ratio'] >= 0.7 else '❌ Fail'}</td>
            </tr>
            <tr>
                <td>Annual Return</td>
                <td>{metrics['annual_return']*100:.2f}%</td>
                <td>10.0%</td>
                <td>{'✅ Pass' if metrics['annual_return']*100 >= 10 else '❌ Fail'}</td>
            </tr>
            <tr>
                <td>Max Drawdown</td>
                <td>{metrics['max_drawdown']*100:.2f}%</td>
                <td>-18.0%</td>
                <td>{'✅ Pass' if metrics['max_drawdown']*100 >= -18 else '❌ Fail'}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{metrics['win_rate']*100:.2f}%</td>
                <td>52.0%</td>
                <td>{'✅ Pass' if metrics['win_rate']*100 >= 52 else '❌ Fail'}</td>
            </tr>
        </table>
        
        <h2>📈 Capital</h2>
        <table>
            <tr><th>Period</th><th>Value</th></tr>
            <tr><td>Initial Capital</td><td>${metrics['initial_capital']:,.0f}</td></tr>
            <tr><td>Current Capital</td><td>${metrics['current_capital']:,.0f}</td></tr>
            <tr><td>Total Profit</td><td class="{'positive' if metrics['total_profit'] > 0 else 'negative'}">${metrics['total_profit']:,.0f}</td></tr>
        </table>
        
        <p style="margin-top: 40px; color: #7f8c8d; font-size: 12px;">
            Auto-generated by monitoring_dashboard.py
        </p>
    </div>
</body>
</html>
"""
        
        with open('ml_performance_report.html', 'w') as f:
            f.write(html)
        
        print(f"📄 HTML report saved: ml_performance_report.html")

def main():
    """Main monitoring function"""
    
    monitor = TradingMonitor()
    
    # Load data
    monitor.load_data()
    
    # Generate dashboard
    metrics = monitor.generate_dashboard()
    
    # Generate HTML report if metrics available
    if metrics:
        monitor.generate_html_report(metrics)
        
        print(f"\n📁 Files generated:")
        print(f"   - ml_monitoring_dashboard.png (charts)")
        print(f"   - ml_performance_report.html (detailed report)")

if __name__ == "__main__":
    main()
