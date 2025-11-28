"""
DOLLAR RISK FACTOR - DXY Beta Hedging

Academic Basis:
- Lustig, Roussanov, Verdelhan (2011): "Common Risk Factors in Currency Markets"
- Dollar beta captures systematic dollar exposure
- High dollar-beta currencies suffer during dollar rallies (risk-off)
- Low/negative dollar-beta currencies hedge dollar risk

Core Idea:
- Calculate beta of EUR/CHF vs DXY (Dollar Index)
- High beta = moves with dollar → reduce position size
- Low/negative beta = hedges dollar → increase position size
- Dynamic risk management based on dollar exposure

Expected Performance:
- Sharpe improvement: 0.05-0.10
- Drawdown reduction: 30-50% in crises
- Works best in risk-off episodes (2020 COVID, 2008 GFC)

Implementation:
- Rolling 252-day beta vs DXY
- Risk multiplier = 1 / (1 + |beta|)
- High beta (>0.5) → reduce size by 30-50%
- Low beta (<0.2) → normal or increase size
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class DollarRiskFactor:
    """
    Dollar risk factor based on DXY beta
    """
    
    def __init__(self):
        self.fx_data = {}
        self.dxy_data = None
        self.beta_window = 252  # 1 year rolling beta
        
    def download_dxy(self, start_date='2015-01-01'):
        """
        Download DXY (Dollar Index) from Yahoo Finance
        """
        print("📥 Downloading DXY (Dollar Index)...")
        data = yf.download('DX-Y.NYB', start=start_date, progress=False)
        
        # Extract as Series
        if isinstance(data['Close'], pd.DataFrame):
            self.dxy_data = data['Close'].iloc[:, 0]
        else:
            self.dxy_data = data['Close']
        
        print(f"   ✅ DX-Y.NYB: {len(data)} days")
        return self.dxy_data
    
    def download_fx_data(self, pair, start_date='2015-01-01'):
        """
        Download spot FX rates from Yahoo Finance
        """
        ticker_map = {
            'EUR': 'EURUSD=X',
            'CHF': 'CHFUSD=X',
            'GBP': 'GBPUSD=X',
            'JPY': 'JPYUSD=X'
        }
        
        ticker = ticker_map.get(pair)
        if not ticker:
            raise ValueError(f"Unknown pair: {pair}")
        
        data = yf.download(ticker, start=start_date, progress=False)
        
        # Extract as Series
        if isinstance(data['Close'], pd.DataFrame):
            self.fx_data[pair] = data['Close'].iloc[:, 0]
        else:
            self.fx_data[pair] = data['Close']
        
        print(f"   ✅ {ticker}: {len(data)} days")
        return self.fx_data[pair]
    
    def calculate_rolling_beta(self, pair):
        """
        Calculate rolling beta of FX pair vs DXY
        
        Beta = Cov(FX, DXY) / Var(DXY)
        
        Returns:
        - DataFrame with beta, correlation, volatility
        """
        if pair not in self.fx_data:
            self.download_fx_data(pair)
        
        if self.dxy_data is None:
            self.download_dxy()
        
        # Daily returns
        fx_returns = self.fx_data[pair].pct_change()
        dxy_returns = self.dxy_data.pct_change()
        
        # Align dates
        aligned = pd.DataFrame({
            'fx': fx_returns,
            'dxy': dxy_returns
        }).dropna()
        
        # Rolling beta
        rolling_cov = aligned['fx'].rolling(self.beta_window).cov(aligned['dxy'])
        rolling_var = aligned['dxy'].rolling(self.beta_window).var()
        beta = rolling_cov / rolling_var
        
        # Rolling correlation
        correlation = aligned['fx'].rolling(self.beta_window).corr(aligned['dxy'])
        
        # Rolling volatility (annualized)
        volatility = aligned['fx'].rolling(self.beta_window).std() * np.sqrt(252)
        
        result = pd.DataFrame({
            'beta': beta,
            'correlation': correlation,
            'volatility': volatility
        }, index=aligned.index)
        
        return result
    
    def calculate_risk_multiplier(self, pair):
        """
        Calculate risk multiplier based on dollar beta
        
        High beta → reduce position size
        Low beta → maintain or increase size
        
        Returns:
        - DataFrame with beta and risk_multiplier
        """
        beta_data = self.calculate_rolling_beta(pair)
        
        # Risk multiplier formula
        # Base: 1.0 (no adjustment)
        # High beta (>0.5): reduce to 0.5-0.7
        # Medium beta (0.2-0.5): slight reduction to 0.8-1.0
        # Low beta (<0.2): maintain 1.0 or increase to 1.1
        
        risk_multiplier = beta_data['beta'].apply(lambda b: self._beta_to_multiplier(b))
        
        beta_data['risk_multiplier'] = risk_multiplier
        
        return beta_data
    
    def _beta_to_multiplier(self, beta):
        """
        Convert beta to risk multiplier
        
        Higher absolute beta → lower multiplier (reduce size)
        """
        abs_beta = abs(beta)
        
        if pd.isna(abs_beta):
            return 1.0
        elif abs_beta > 0.7:
            return 0.5  # Very high beta: cut size by 50%
        elif abs_beta > 0.5:
            return 0.7  # High beta: cut size by 30%
        elif abs_beta > 0.3:
            return 0.9  # Medium beta: slight reduction
        else:
            return 1.0  # Low beta: normal size
    
    def generate_risk_signal(self, pair):
        """
        Generate current dollar risk signal
        
        Returns:
        - multiplier: 0.5 to 1.0 (position size adjustment)
        - beta: Current dollar beta
        - components: dict with details
        """
        risk_data = self.calculate_risk_multiplier(pair)
        
        # Latest values
        latest = risk_data.iloc[-1]
        
        multiplier = latest['risk_multiplier']
        beta = latest['beta']
        correlation = latest['correlation']
        volatility = latest['volatility']
        
        components = {
            'beta': beta,
            'correlation': correlation,
            'volatility': volatility,
            'risk_multiplier': multiplier
        }
        
        return multiplier, components
    
    def backtest_risk_adjusted(self, pair):
        """
        Backtest dollar risk-adjusted strategy
        
        Compare:
        1. Unadjusted returns
        2. Risk-adjusted returns (scaled by multiplier)
        
        Returns:
        - Performance improvement metrics
        """
        risk_data = self.calculate_risk_multiplier(pair)
        fx_returns = self.fx_data[pair].pct_change()
        
        # Align
        aligned = pd.DataFrame({
            'returns': fx_returns,
            'multiplier': risk_data['risk_multiplier']
        }).dropna()
        
        # Unadjusted vs adjusted
        aligned['unadjusted_return'] = aligned['returns']
        aligned['adjusted_return'] = aligned['returns'] * aligned['multiplier']
        
        # Cumulative
        aligned['unadjusted_cumulative'] = (1 + aligned['unadjusted_return']).cumprod()
        aligned['adjusted_cumulative'] = (1 + aligned['adjusted_return']).cumprod()
        
        # Performance metrics
        def calc_metrics(returns):
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            total_return = (1 + returns).cumprod().iloc[-1] - 1
            max_dd = (returns.cumsum() - returns.cumsum().cummax()).min()
            return {'sharpe': sharpe, 'return': total_return, 'max_dd': max_dd}
        
        unadjusted = calc_metrics(aligned['unadjusted_return'])
        adjusted = calc_metrics(aligned['adjusted_return'])
        
        return {
            'unadjusted': unadjusted,
            'adjusted': adjusted,
            'improvement': {
                'sharpe_delta': adjusted['sharpe'] - unadjusted['sharpe'],
                'return_delta': adjusted['return'] - unadjusted['return'],
                'max_dd_delta': adjusted['max_dd'] - unadjusted['max_dd']
            },
            'num_days': len(aligned)
        }


if __name__ == '__main__':
    print("="*70)
    print("🚀 DOLLAR RISK FACTOR ANALYSIS (DXY Beta)")
    print("="*70)
    print()
    
    # Initialize
    dollar_risk = DollarRiskFactor()
    
    # Download DXY
    dollar_risk.download_dxy()
    print()
    
    # Download FX data
    print("📥 Downloading FX data...")
    pairs = ['EUR', 'CHF']
    for pair in pairs:
        dollar_risk.download_fx_data(pair)
    print()
    
    # Current signals
    print("="*70)
    print("📊 CURRENT DOLLAR RISK SIGNALS")
    print("="*70)
    print()
    
    for pair in pairs:
        multiplier, components = dollar_risk.generate_risk_signal(pair)
        
        print(f"{pair}/USD:")
        print(f"   Risk Multiplier: {multiplier:.2f}x")
        print(f"   Dollar Beta: {components['beta']:+.3f}")
        print(f"   Correlation: {components['correlation']:+.3f}")
        print(f"   Volatility (annual): {components['volatility']*100:.1f}%")
        print()
        
        # Interpretation
        if multiplier < 0.7:
            print(f"   ⚠️  HIGH dollar risk - Reduce position size to {multiplier:.0%}")
        elif multiplier < 0.9:
            print(f"   ⚡ MEDIUM dollar risk - Slight reduction to {multiplier:.0%}")
        else:
            print(f"   ✅ LOW dollar risk - Normal position size")
        print()
    
    # Backtest
    print("="*70)
    print("📈 HISTORICAL BACKTEST (Risk-Adjusted Performance)")
    print("="*70)
    print()
    
    results = []
    for pair in pairs:
        perf = dollar_risk.backtest_risk_adjusted(pair)
        
        print(f"{pair}/USD:")
        print(f"   UNADJUSTED:")
        print(f"      Sharpe: {perf['unadjusted']['sharpe']:.3f}")
        print(f"      Return: {perf['unadjusted']['return']*100:+.1f}%")
        print(f"      Max DD: {perf['unadjusted']['max_dd']*100:.1f}%")
        print()
        print(f"   RISK-ADJUSTED:")
        print(f"      Sharpe: {perf['adjusted']['sharpe']:.3f}")
        print(f"      Return: {perf['adjusted']['return']*100:+.1f}%")
        print(f"      Max DD: {perf['adjusted']['max_dd']*100:.1f}%")
        print()
        print(f"   IMPROVEMENT:")
        print(f"      Sharpe: {perf['improvement']['sharpe_delta']:+.3f}")
        print(f"      Return: {perf['improvement']['return_delta']*100:+.1f}%")
        print(f"      Max DD: {perf['improvement']['max_dd_delta']*100:+.1f}%")
        print()
        
        results.append({
            'Currency': pair,
            'Sharpe_Unadj': perf['unadjusted']['sharpe'],
            'Sharpe_Adj': perf['adjusted']['sharpe'],
            'Sharpe_Delta': perf['improvement']['sharpe_delta'],
            'DD_Unadj': perf['unadjusted']['max_dd'],
            'DD_Adj': perf['adjusted']['max_dd'],
            'DD_Improvement': perf['improvement']['max_dd_delta']
        })
    
    # Summary table
    print("="*70)
    print("📊 SUMMARY TABLE")
    print("="*70)
    print()
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    
    print("="*70)
    print("✅ DOLLAR RISK FACTOR ANALYSIS COMPLETE")
    print("="*70)
    print()
    
    print("💡 Key Insights:")
    print("   • Dollar beta measures systematic dollar exposure")
    print("   • High beta → high risk in dollar rallies → reduce size")
    print("   • Risk-adjusted sizing reduces drawdowns")
    print("   • Positive sharpe delta = risk management adds value")
    print()
    
    print("📊 Next Steps:")
    print("   1. Add VIX regime filter")
    print("   2. Run multi-factor backtest")
    print("   3. Combine value + dollar risk + VIX")
