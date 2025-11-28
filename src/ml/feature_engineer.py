"""
Advanced Feature Engineering for ML FX Strategy
Creates 60+ features across multiple categories
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Comprehensive feature creation for FX ML models"""
    
    def __init__(self, currencies: List[str] = None):
        if currencies is None:
            self.currencies = ['AUD', 'BRL', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'MXN']
        else:
            self.currencies = currencies
            
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from raw data
        
        Args:
            data: DataFrame with FX rates, interest rates, market data
            
        Returns:
            DataFrame with 60+ engineered features
        """
        print("🔧 Engineering features...")
        
        features = pd.DataFrame(index=data.index)
        
        # 1. Carry Features
        print("  ├─ Carry features...")
        carry_features = self._create_carry_features(data)
        features = features.join(carry_features)
        
        # 2. Momentum Features
        print("  ├─ Momentum features...")
        momentum_features = self._create_momentum_features(data)
        features = features.join(momentum_features)
        
        # 3. Volatility Features
        print("  ├─ Volatility features...")
        vol_features = self._create_volatility_features(data)
        features = features.join(vol_features)
        
        # 4. Market Risk Features
        print("  ├─ Market risk features...")
        risk_features = self._create_risk_features(data)
        features = features.join(risk_features)
        
        # 5. Dollar Beta Features
        print("  ├─ Dollar beta features...")
        dollar_features = self._create_dollar_features(data)
        features = features.join(dollar_features)
        
        # 6. Macro Features
        print("  ├─ Macro features...")
        macro_features = self._create_macro_features(data)
        features = features.join(macro_features)
        
        # 7. Technical Indicators
        print("  ├─ Technical indicators...")
        technical_features = self._create_technical_features(data)
        features = features.join(technical_features)
        
        # 8. Interaction Features
        print("  └─ Interaction features...")
        interaction_features = self._create_interaction_features(features)
        features = features.join(interaction_features)
        
        # Drop NaN rows (from rolling calculations)
        features = features.fillna(method='ffill').dropna()
        
        print(f"✅ Created {len(features.columns)} features")
        
        return features
    
    def _create_carry_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Carry-based features"""
        carry = pd.DataFrame(index=data.index)
        
        for curr in self.currencies:
            rate_col = f'{curr}_rate'
            if rate_col in data.columns and 'USD_rate' in data.columns:
                # Interest rate differential
                carry[f'{curr}_rate_diff'] = data[rate_col] - data.get('USD_rate', 0)
                
                # Z-score of rate differential
                carry[f'{curr}_carry_zscore'] = self._zscore(
                    carry[f'{curr}_rate_diff'], window=252
                )
                
                # Percentile rank
                carry[f'{curr}_carry_rank'] = carry[f'{curr}_rate_diff'].rolling(
                    252).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x))
        
        return carry
    
    def _create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Momentum features at multiple time scales"""
        mom = pd.DataFrame(index=data.index)
        
        # Time windows (in days)
        windows = {
            '1m': 21,
            '3m': 63,
            '6m': 126,
            '12m': 252
        }
        
        for curr in self.currencies:
            fx_col = f'{curr}_USD'
            if fx_col in data.columns:
                # Calculate returns for each window
                for name, window in windows.items():
                    # Raw momentum
                    mom[f'{curr}_mom_{name}'] = data[fx_col].pct_change(window)
                    
                    # Volatility-adjusted momentum
                    vol = data[fx_col].pct_change().rolling(window).std()
                    mom[f'{curr}_mom_{name}_vol_adj'] = (
                        mom[f'{curr}_mom_{name}'] / (vol * np.sqrt(window))
                    )
                
                # Skip-1-month momentum (classic academic)
                mom[f'{curr}_mom_12_1'] = (
                    data[fx_col].pct_change(252).shift(21)
                )
        
        return mom
    
    def _create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Volatility-based features"""
        vol = pd.DataFrame(index=data.index)
        
        windows = [21, 63, 126]
        
        for curr in self.currencies:
            fx_col = f'{curr}_USD'
            if fx_col in data.columns:
                returns = data[fx_col].pct_change()
                
                # Realized volatility at different horizons
                for window in windows:
                    vol[f'{curr}_vol_{window}d'] = (
                        returns.rolling(window).std() * np.sqrt(252)
                    )
                
                # Volatility of volatility
                vol[f'{curr}_vol_of_vol'] = (
                    vol[f'{curr}_vol_21d'].rolling(21).std()
                )
                
                # Volatility ratio (short/long)
                vol[f'{curr}_vol_ratio'] = (
                    vol[f'{curr}_vol_21d'] / vol[f'{curr}_vol_63d']
                )
                
                # Downside volatility
                negative_returns = returns.copy()
                negative_returns[negative_returns > 0] = 0
                vol[f'{curr}_downside_vol'] = (
                    negative_returns.rolling(63).std() * np.sqrt(252)
                )
        
        return vol
    
    def _create_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Market risk features"""
        risk = pd.DataFrame(index=data.index)
        
        # VIX features
        if 'VIX' in data.columns:
            risk['vix'] = data['VIX']
            risk['vix_change'] = data['VIX'].diff()
            risk['vix_zscore'] = self._zscore(data['VIX'], window=252)
            risk['vix_spike'] = (data['VIX'] > data['VIX'].rolling(252).mean() + 
                                2 * data['VIX'].rolling(252).std()).astype(int)
        
        # Credit spreads
        if 'credit_spread_ig' in data.columns:
            risk['credit_spread'] = data['credit_spread_ig']
            risk['credit_spread_change'] = data['credit_spread_ig'].diff()
            risk['credit_spread_wide'] = (
                data['credit_spread_ig'] > 
                data['credit_spread_ig'].rolling(252).quantile(0.75)
            ).astype(int)
        
        # Term spread
        if 'term_spread' in data.columns:
            risk['term_spread'] = data['term_spread']
            risk['term_spread_negative'] = (data['term_spread'] < 0).astype(int)
        
        # Equity market features
        if 'SPX_ret' in data.columns:
            risk['equity_ret_21d'] = data['SPX_ret'].rolling(21).sum()
            risk['equity_vol_21d'] = (
                data['SPX_ret'].rolling(21).std() * np.sqrt(252)
            )
            risk['equity_drawdown'] = self._calculate_drawdown(data['SPX'])
        
        return risk
    
    def _create_dollar_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Dollar index features"""
        dollar = pd.DataFrame(index=data.index)
        
        if 'DXY' in data.columns:
            # Dollar level
            dollar['dxy'] = data['DXY']
            
            # Dollar returns
            dollar['dxy_ret_1m'] = data['DXY'].pct_change(21)
            dollar['dxy_ret_3m'] = data['DXY'].pct_change(63)
            dollar['dxy_ret_6m'] = data['DXY'].pct_change(126)
            
            # Dollar momentum
            dollar['dxy_mom_zscore'] = self._zscore(
                data['DXY'].pct_change(63), window=252
            )
            
            # Dollar trend
            dollar['dxy_above_ma'] = (
                data['DXY'] > data['DXY'].rolling(126).mean()
            ).astype(int)
            
            # Calculate beta to dollar for each currency
            for curr in self.currencies:
                fx_col = f'{curr}_USD'
                if fx_col in data.columns:
                    dollar[f'{curr}_dxy_beta'] = self._rolling_beta(
                        data[fx_col].pct_change(),
                        data['DXY'].pct_change(),
                        window=126
                    )
        
        return dollar
    
    def _create_macro_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Macroeconomic features"""
        macro = pd.DataFrame(index=data.index)
        
        # GDP growth
        if 'gdp_us_yoy' in data.columns:
            macro['gdp_growth'] = data['gdp_us_yoy']
        
        # Inflation
        if 'cpi_us_yoy' in data.columns:
            macro['inflation'] = data['cpi_us_yoy']
        
        # Unemployment
        if 'unemployment_us' in data.columns:
            macro['unemployment'] = data['unemployment_us']
            macro['unemployment_change'] = data['unemployment_us'].diff(12)
        
        return macro
    
    def _create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Technical indicators"""
        technical = pd.DataFrame(index=data.index)
        
        for curr in self.currencies:
            fx_col = f'{curr}_USD'
            if fx_col in data.columns:
                price = data[fx_col]
                
                # Moving averages
                technical[f'{curr}_ma_20'] = price.rolling(20).mean()
                technical[f'{curr}_ma_50'] = price.rolling(50).mean()
                technical[f'{curr}_ma_200'] = price.rolling(200).mean()
                
                # MA crossovers
                technical[f'{curr}_ma_cross'] = (
                    (technical[f'{curr}_ma_20'] > technical[f'{curr}_ma_50'])
                    .astype(int)
                )
                
                # RSI
                technical[f'{curr}_rsi'] = self._calculate_rsi(price, window=14)
                
                # Price vs MA
                technical[f'{curr}_price_vs_ma200'] = (
                    (price - technical[f'{curr}_ma_200']) / 
                    technical[f'{curr}_ma_200']
                )
        
        return technical
    
    def _create_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Interaction features between different factors"""
        interactions = pd.DataFrame(index=features.index)
        
        for curr in self.currencies:
            # Carry × Momentum
            carry_col = f'{curr}_rate_diff'
            mom_col = f'{curr}_mom_12m'
            if carry_col in features.columns and mom_col in features.columns:
                interactions[f'{curr}_carry_x_mom'] = (
                    features[carry_col] * features[mom_col]
                )
            
            # Carry × Volatility
            vol_col = f'{curr}_vol_63d'
            if carry_col in features.columns and vol_col in features.columns:
                interactions[f'{curr}_carry_x_vol'] = (
                    features[carry_col] / (features[vol_col] + 1e-10)
                )
            
            # Momentum × VIX
            if mom_col in features.columns and 'vix' in features.columns:
                interactions[f'{curr}_mom_x_vix'] = (
                    features[mom_col] * features['vix']
                )
        
        return interactions
    
    # Helper functions
    
    def _zscore(self, series: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling z-score"""
        return (series - series.rolling(window).mean()) / series.rolling(window).std()
    
    def _rolling_beta(self, y: pd.Series, x: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling beta (covariance / variance)"""
        # Simple rolling correlation * (std(y) / std(x))
        # This approximates beta and is computationally efficient
        corr = y.rolling(window).corr(x)
        std_y = y.rolling(window).std()
        std_x = x.rolling(window).std()
        return corr * (std_y / std_x)
    
    def _calculate_drawdown(self, series: pd.Series) -> pd.Series:
        """Calculate running maximum drawdown"""
        cummax = series.expanding().max()
        drawdown = (series - cummax) / cummax
        return drawdown
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_feature_groups(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Group features by category for analysis"""
        groups = {
            'carry': [],
            'momentum': [],
            'volatility': [],
            'risk': [],
            'dollar': [],
            'macro': [],
            'technical': [],
            'interaction': []
        }
        
        for col in features.columns:
            if 'rate_diff' in col or 'carry' in col:
                groups['carry'].append(col)
            elif 'mom' in col:
                groups['momentum'].append(col)
            elif 'vol' in col:
                groups['volatility'].append(col)
            elif 'vix' in col or 'credit' in col or 'term' in col or 'equity' in col:
                groups['risk'].append(col)
            elif 'dxy' in col:
                groups['dollar'].append(col)
            elif 'gdp' in col or 'inflation' in col or 'unemployment' in col:
                groups['macro'].append(col)
            elif 'ma' in col or 'rsi' in col:
                groups['technical'].append(col)
            elif '_x_' in col:
                groups['interaction'].append(col)
        
        return groups


if __name__ == "__main__":
    # Test feature engineering
    print("Testing FeatureEngineer...")
    
    # Create dummy data
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    data = pd.DataFrame(index=dates)
    
    # Add dummy FX rates
    for curr in ['EUR', 'GBP', 'JPY']:
        data[f'{curr}_USD'] = 1.0 + np.random.randn(len(dates)).cumsum() * 0.01
        data[f'{curr}_rate'] = 2.0 + np.random.randn(len(dates)).cumsum() * 0.1
    
    # Add market data
    data['VIX'] = 20 + np.random.randn(len(dates)).cumsum() * 2
    data['DXY'] = 100 + np.random.randn(len(dates)).cumsum() * 1
    data['SPX'] = 3000 + np.random.randn(len(dates)).cumsum() * 50
    data['SPX_ret'] = data['SPX'].pct_change()
    
    # Create features
    fe = FeatureEngineer(currencies=['EUR', 'GBP', 'JPY'])
    features = fe.create_all_features(data)
    
    print(f"\n✅ Created {len(features.columns)} features")
    print(f"   Sample features: {list(features.columns[:10])}")
    
    # Show feature groups
    groups = fe.get_feature_groups(features)
    print(f"\n📊 Feature groups:")
    for name, cols in groups.items():
        print(f"   {name}: {len(cols)} features")
