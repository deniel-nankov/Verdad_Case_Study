"""
Trading System Monitoring Dashboard
====================================

Real-time dashboard for monitoring:
- Current positions
- Performance metrics
- Risk status
- Recent trades
- Data feed status

Usage:
    python monitoring_dashboard.py
"""

import pandas as pd
import time
import os
import json
from datetime import datetime, timedelta
from broker_integrations import PaperTradingBroker, BrokerConfig

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def load_cache_status():
    """Load FX cache status"""
    try:
        if os.path.exists('fx_data_cache.json'):
            with open('fx_data_cache.json', 'r') as f:
                cache = json.load(f)
                
            # Get timestamp of first entry
            if cache:
                first_key = list(cache.keys())[0]
                timestamp_str = cache[first_key].get('timestamp', '')
                source = cache[first_key].get('source', 'Unknown')
                
                return {
                    'pairs': len(cache),
                    'source': source,
                    'timestamp': timestamp_str,
                    'status': '✅ ACTIVE'
                }
        return {'pairs': 0, 'source': 'None', 'timestamp': 'N/A', 'status': '❌ NO CACHE'}
    except:
        return {'pairs': 0, 'source': 'Error', 'timestamp': 'N/A', 'status': '❌ ERROR'}

import os
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def load_performance_log():
    """Load performance log CSV"""
    log_file = Path('performance_log.csv')
    
    if not log_file.exists():
        return None
    
    try:
        df = pd.read_csv(log_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading performance log: {e}")
        return None


def load_trading_log():
    """Load recent trading activity from log file"""
    log_file = Path('trading_system.log')
    
    if not log_file.exists():
        return []
    
    try:
        # Read last 50 lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return lines[-50:] if len(lines) > 50 else lines
    except Exception as e:
        print(f"Error loading trading log: {e}")
        return []


def calculate_metrics(perf_df):
    """Calculate key performance metrics"""
    if perf_df is None or len(perf_df) == 0:
        return None
    
    latest = perf_df.iloc[-1]
    
    # Calculate returns
    perf_df['returns'] = perf_df['total_value'].pct_change()
    
    # Calculate metrics
    total_return = (latest['total_value'] / perf_df.iloc[0]['total_value'] - 1) * 100
    
    # Sharpe ratio (annualized)
    if len(perf_df) > 1:
        mean_return = perf_df['returns'].mean() * 252
        std_return = perf_df['returns'].std() * np.sqrt(252)
        sharpe = mean_return / std_return if std_return > 0 else 0
    else:
        sharpe = 0
    
    # Max drawdown
    cum_returns = (1 + perf_df['returns']).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdowns.min() * 100
    
    # Win rate
    if len(perf_df) > 1:
        wins = (perf_df['returns'] > 0).sum()
        total_trades = len(perf_df) - 1
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    else:
        win_rate = 0
    
    # Trading days
    trading_days = (latest['timestamp'] - perf_df.iloc[0]['timestamp']).days
    
    return {
        'current_value': latest['total_value'],
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'num_positions': latest.get('num_positions', 0),
        'trading_days': trading_days,
        'last_update': latest['timestamp']
    }


def format_pnl(value, pct=False):
    """Format P&L with color"""
    if pct:
        formatted = f"{value:+.2f}%"
    else:
        formatted = f"${value:+,.2f}"
    
    if value > 0:
        return Colors.GREEN + formatted + Colors.END
    elif value < 0:
        return Colors.RED + formatted + Colors.END
    else:
        return formatted


def display_dashboard():
    """Display real-time dashboard"""
    clear_screen()
    
    # Header
    print(Colors.BOLD + Colors.BLUE + "="*80 + Colors.END)
    print(Colors.BOLD + Colors.BLUE + "📊 FX CARRY TRADING SYSTEM - LIVE DASHBOARD" + Colors.END)
    print(Colors.BOLD + Colors.BLUE + "="*80 + Colors.END)
    print(f"{Colors.CYAN}Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}\n")
    
    # Data Feed Status (new section)
    cache_status = load_cache_status()
    print(Colors.BOLD + "📡 DATA FEED STATUS" + Colors.END)
    print("-" * 80)
    print(f"  Status:              {cache_status['status']}")
    print(f"  Source:              {cache_status['source']}")
    print(f"  Currency Pairs:      {cache_status['pairs']}")
    if cache_status['timestamp'] != 'N/A':
        try:
            from dateutil import parser
            ts = parser.parse(cache_status['timestamp'])
            age_mins = (datetime.now() - ts.replace(tzinfo=None)).total_seconds() / 60
            age_color = Colors.GREEN if age_mins < 360 else Colors.YELLOW if age_mins < 1440 else Colors.RED
            print(f"  Data Age:            {age_color}{age_mins:.0f} minutes{Colors.END}")
            print(f"  Last Update:         {ts.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            print(f"  Last Update:         {cache_status['timestamp']}")
    print()
    
    # Load data
    perf_df = load_performance_log()
    
    if perf_df is None or len(perf_df) == 0:
        print(Colors.YELLOW + "⚠️  No performance data available yet" + Colors.END)
        print(Colors.YELLOW + "   Start the trading system with: python live_trading_system.py" + Colors.END)
        print()
        print(Colors.BOLD + "🔧 QUICK ACTIONS:" + Colors.END)
        print("   - Refresh cache:  ./refresh_cache.sh")
        print("   - Start trading:  python live_trading_system.py &")
        print("   - View logs:      tail -f trading_system.log")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(perf_df)
    
    # Performance Overview
    print(Colors.BOLD + "📈 PERFORMANCE OVERVIEW" + Colors.END)
    print("-" * 80)
    print(f"  Portfolio Value:     ${metrics['current_value']:,.2f}")
    print(f"  Total Return:        {format_pnl(metrics['total_return'], pct=True)}")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:        {format_pnl(metrics['max_drawdown'], pct=True)}")
    print(f"  Win Rate:            {metrics['win_rate']:.1f}%")
    print(f"  Trading Days:        {metrics['trading_days']}")
    print(f"  Active Positions:    {int(metrics['num_positions'])}")
    print()
    
    # Recent Performance (last 7 days)
    recent_df = perf_df[perf_df['timestamp'] >= datetime.now() - timedelta(days=7)]
    if len(recent_df) > 1:
        recent_return = (recent_df.iloc[-1]['total_value'] / recent_df.iloc[0]['total_value'] - 1) * 100
        
        print(Colors.BOLD + "📅 RECENT PERFORMANCE (7 Days)" + Colors.END)
        print("-" * 80)
        print(f"  7-Day Return:        {format_pnl(recent_return, pct=True)}")
        print()
    
    # Current Positions
    latest_row = perf_df.iloc[-1]
    
    print(Colors.BOLD + "💼 CURRENT POSITIONS" + Colors.END)
    print("-" * 80)
    
    # Try to parse positions from log (simplified)
    if 'num_positions' in latest_row and latest_row['num_positions'] > 0:
        print(f"  {int(latest_row['num_positions'])} active positions")
        print(f"  (See trading_system.log for details)")
    else:
        print("  No active positions")
    print()
    
    # Risk Metrics
    print(Colors.BOLD + "🛡️  RISK METRICS" + Colors.END)
    print("-" * 80)
    
    # Current drawdown
    cum_returns = (1 + perf_df['returns']).cumprod()
    rolling_max = cum_returns.expanding().max()
    current_dd = ((cum_returns.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1]) * 100
    
    dd_color = Colors.RED if current_dd < -10 else (Colors.YELLOW if current_dd < -5 else Colors.GREEN)
    print(f"  Current Drawdown:    {dd_color}{current_dd:+.2f}%{Colors.END}")
    
    # Daily volatility
    if len(perf_df) > 20:
        recent_vol = perf_df['returns'].tail(20).std() * np.sqrt(252) * 100
        print(f"  Annualized Vol:      {recent_vol:.2f}%")
    
    # VaR (95%)
    if len(perf_df) > 1:
        var_95 = np.percentile(perf_df['returns'].dropna(), 5) * 100
        print(f"  VaR (95%):           {format_pnl(var_95, pct=True)}")
    
    print()
    
    # Recent Activity
    print(Colors.BOLD + "📝 RECENT ACTIVITY (Last 10 Log Entries)" + Colors.END)
    print("-" * 80)
    
    log_lines = load_trading_log()
    if log_lines:
        for line in log_lines[-10:]:
            line = line.strip()
            if 'ERROR' in line:
                print(Colors.RED + line + Colors.END)
            elif 'WARNING' in line:
                print(Colors.YELLOW + line + Colors.END)
            elif 'order' in line.lower() or 'position' in line.lower():
                print(Colors.CYAN + line + Colors.END)
            else:
                print(line)
    else:
        print("  No recent activity")
    
    print()
    
    # System Status
    print(Colors.BOLD + "⚙️  SYSTEM STATUS" + Colors.END)
    print("-" * 80)
    
    # Check if system is running (last update within 1 hour)
    time_since_update = datetime.now() - metrics['last_update']
    if time_since_update < timedelta(hours=1):
        status_color = Colors.GREEN
        status_text = "🟢 RUNNING"
    elif time_since_update < timedelta(hours=6):
        status_color = Colors.YELLOW
        status_text = "🟡 IDLE"
    else:
        status_color = Colors.RED
        status_text = "🔴 STOPPED"
    
    print(f"  Status:              {status_color}{status_text}{Colors.END}")
    print(f"  Last Update:         {metrics['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Time Since Update:   {time_since_update}")
    print()
    
    # Footer
    print(Colors.BLUE + "="*80 + Colors.END)
    print(f"{Colors.CYAN}Press Ctrl+C to exit | Auto-refresh every 30 seconds{Colors.END}")


def run_dashboard(refresh_interval=30):
    """Run dashboard with auto-refresh"""
    print("Starting monitoring dashboard...")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to exit\n")
    time.sleep(2)
    
    try:
        while True:
            display_dashboard()
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        clear_screen()
        print("\n" + Colors.YELLOW + "Dashboard stopped by user" + Colors.END)
        print("Goodbye! 👋\n")


def export_report(output_file='trading_report.html'):
    """Export performance report to HTML"""
    perf_df = load_performance_log()
    
    if perf_df is None or len(perf_df) == 0:
        print("No data to export")
        return
    
    metrics = calculate_metrics(perf_df)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
            .metric {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .positive {{ color: #27ae60; font-weight: bold; }}
            .negative {{ color: #e74c3c; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; background: white; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #34495e; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 FX Carry Trading Performance Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metric">
            <h2>Performance Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Portfolio Value</td><td>${metrics['current_value']:,.2f}</td></tr>
                <tr><td>Total Return</td><td class="{'positive' if metrics['total_return'] > 0 else 'negative'}">{metrics['total_return']:+.2f}%</td></tr>
                <tr><td>Sharpe Ratio</td><td>{metrics['sharpe_ratio']:.3f}</td></tr>
                <tr><td>Max Drawdown</td><td class="negative">{metrics['max_drawdown']:.2f}%</td></tr>
                <tr><td>Win Rate</td><td>{metrics['win_rate']:.1f}%</td></tr>
                <tr><td>Trading Days</td><td>{metrics['trading_days']}</td></tr>
            </table>
        </div>
        
        <div class="metric">
            <h2>Recent Performance Data</h2>
            {perf_df.tail(20).to_html(index=False)}
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Report exported to: {output_file}")
    print(f"   Open in browser: open {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Trading System Monitoring Dashboard')
    parser.add_argument('--refresh', type=int, default=30, help='Refresh interval in seconds (default: 30)')
    parser.add_argument('--export', action='store_true', help='Export performance report to HTML and exit')
    parser.add_argument('--once', action='store_true', help='Display dashboard once without auto-refresh')
    
    args = parser.parse_args()
    
    if args.export:
        export_report()
    elif args.once:
        display_dashboard()
    else:
        run_dashboard(refresh_interval=args.refresh)
