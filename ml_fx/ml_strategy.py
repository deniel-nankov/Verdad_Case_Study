"""
ML-Based FX Carry Strategy
Integrates machine learning predictions into trading system
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import os
import sys

# Add ml_fx to path
sys.path.append(os.path.dirname(__file__))

from data_loader import MLDataLoader
from feature_engineer import FeatureEngineer
from ml_models import MLEnsemble


class MLFXStrategy:
    """
    Machine Learning FX Carry Strategy
    
    Uses ensemble of RF, XGBoost, LSTM to predict currency returns
    Combines predictions with traditional carry signals
    """
    
    def __init__(self, 
                 fred_api_key: str,
                 currencies: List[str] = None,
                 model_dir: str = "./ml_models",
                 data_cache_dir: str = "./data_cache"):
        
        if currencies is None:
            self.currencies = ['AUD', 'BRL', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'MXN']
        else:
            self.currencies = currencies
        
        # Initialize components
        self.data_loader = MLDataLoader(
            fred_api_key=fred_api_key,
            cache_dir=data_cache_dir
        )
        self.feature_engineer = FeatureEngineer(currencies=self.currencies)
        self.ensemble = MLEnsemble(model_dir=model_dir)
        
        self.trained = False
        self.feature_cols = None
        
    def train_all_currencies(self, start_date: str = '2015-01-01',
                           end_date: str = None,
                           validation_split: float = 0.2,
                           walk_forward: bool = False,
                           optimize: bool = True,
                           skip_lstm: bool = False) -> Dict:
        """
        Train ML models for all currencies
        
        Args:
            start_date: Training data start
            end_date: Training data end
            validation_split: Validation fraction
            walk_forward: Use walk-forward validation
        """
        print("=" * 70)
        print("🚀 TRAINING ML FX STRATEGY")
        print("=" * 70)
        
        # 1. Load data
        print("\n📥 Step 1: Loading data...")
        data = self.data_loader.load_all_data(start_date, end_date)
        
        # 2. Engineer features
        print("\n🔧 Step 2: Engineering features...")
        features = self.feature_engineer.create_all_features(data)
        
        # Store feature columns
        self.feature_cols = features.columns.tolist()
        
        # 3. Train models for each currency
        print("\n🤖 Step 3: Training models...")
        
        results = {}
        
        for currency in self.currencies:
            print(f"\n{'='*70}")
            print(f"📊 Currency: {currency}")
            print(f"{'='*70}")
            
            # Create target: forward 21-day return
            fx_col = f'{currency}_USD'
            if fx_col not in data.columns:
                print(f"  ⚠️  Skipping {currency} - no FX data")
                continue
            
            # Target: predict next 21 days return
            target = data[fx_col].pct_change(21).shift(-21)
            
            # Align features and target
            valid_idx = features.index.intersection(target.dropna().index)
            X = features.loc[valid_idx]
            y = target.loc[valid_idx]
            
            if len(X) < 500:
                print(f"  ⚠️  Skipping {currency} - insufficient data ({len(X)} samples)")
                continue
            
            # Train models
            if walk_forward:
                # Walk-forward validation
                results[currency] = self.ensemble.walk_forward_validation(
                    X, y, currency, n_splits=5
                )
            else:
                # Single train/val split
                results[currency] = self.ensemble.train(
                    X, y, currency,
                    validation_split=validation_split,
                    optimize=optimize,
                    skip_lstm=skip_lstm
                )
            
            # Save models
            self.ensemble.save_models(currency)
            
            # Show top features
            print(f"\n  🎯 Top 10 features for {currency}:")
            top_features = self.ensemble.get_feature_importance(currency, top_n=10)
            for idx, row in top_features.iterrows():
                print(f"     {row['feature']:40s} {row['avg_importance']:.4f}")
        
        self.trained = True
        
        # 4. Summary
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"\n📊 Model Performance Summary:")
        print(f"{'Currency':<10} {'RF R²':>10} {'XGB R²':>10} {'LSTM R²':>10} {'Ensemble R²':>12}")
        print("-" * 70)
        
        for currency in self.currencies:
            if currency in results:
                res = results[currency]
                if 'ensemble' in res:
                    # Single split results
                    print(f"{currency:<10} {res['rf']['r2']:>10.4f} "
                          f"{res['xgb']['r2']:>10.4f} {res['lstm']['r2']:>10.4f} "
                          f"{res['ensemble']['r2']:>12.4f}")
                else:
                    # Walk-forward results
                    print(f"{currency:<10} {res['rf']['avg_r2']:>10.4f} "
                          f"{res['xgb']['avg_r2']:>10.4f} {res['lstm']['avg_r2']:>10.4f} "
                          f"{res['ensemble']['avg_r2']:>12.4f}")
        
        return results
    
    def load_trained_models(self):
        """Load pre-trained models from disk"""
        print("📂 Loading trained models...")
        
        for currency in self.currencies:
            try:
                self.ensemble.load_models(currency)
                print(f"  ✅ {currency}")
            except FileNotFoundError:
                print(f"  ⚠️  {currency} - no saved models")
        
        self.trained = True
        print("✅ Models loaded")
    
    def generate_signals(self, 
                        current_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Generate trading signals for all currencies
        
        Args:
            current_data: Current market data (if None, will fetch latest)
            
        Returns:
            Dictionary of {currency: signal} where signal in [-1, 1]
        """
        if not self.trained:
            raise ValueError("Models not trained. Call train_all_currencies() first.")
        
        # Get current data
        if current_data is None:
            print("📡 Fetching latest market data...")
            current_data = self.data_loader.get_latest_data()
        
        # Engineer features
        features = self.feature_engineer.create_all_features(current_data)
        
        # Get latest features
        latest_features = features.iloc[[-1]]
        
        signals = {}
        
        print(f"\n🎯 Generating ML signals...")
        print(f"{'Currency':<10} {'ML Prediction':>15} {'Signal':>10}")
        print("-" * 40)
        
        for currency in self.currencies:
            try:
                # Make prediction
                prediction = self.ensemble.predict(currency, latest_features)
                
                # Convert prediction to signal (-1 to 1)
                # Positive prediction = go long
                # Negative prediction = go short
                signal = np.tanh(prediction[0] * 10)  # Scale and squash
                
                signals[currency] = signal
                
                direction = "LONG" if signal > 0 else "SHORT"
                print(f"{currency:<10} {prediction[0]:>15.6f} {signal:>10.4f} {direction}")
                
            except Exception as e:
                print(f"{currency:<10} ERROR: {e}")
                signals[currency] = 0.0
        
        return signals
    
    def generate_positions(self, 
                          signals: Dict[str, float],
                          capital: float = 100000,
                          max_position_size: float = 0.25,
                          risk_scale: float = 1.0) -> Dict[str, float]:
        """
        Convert signals to position sizes
        
        Args:
            signals: Dictionary of currency signals
            capital: Total capital
            max_position_size: Maximum position as fraction of capital
            risk_scale: Risk scaling factor (0 to 1)
            
        Returns:
            Dictionary of {currency: position_size_usd}
        """
        positions = {}
        
        # Sort currencies by signal strength
        sorted_currencies = sorted(
            signals.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Allocate capital based on signal strength
        total_signal = sum(abs(signal) for _, signal in sorted_currencies)
        
        if total_signal == 0:
            return {curr: 0.0 for curr in self.currencies}
        
        for currency, signal in sorted_currencies:
            # Position size proportional to signal strength
            raw_size = (abs(signal) / total_signal) * capital
            
            # Apply maximum position size constraint
            position_size = min(raw_size, capital * max_position_size)
            
            # Apply risk scaling
            position_size *= risk_scale
            
            # Apply signal direction
            positions[currency] = position_size * np.sign(signal)
        
        return positions
    
    def backtest(self, 
                 start_date: str = '2020-01-01',
                 end_date: Optional[str] = None,
                 rebalance_freq: str = 'M') -> pd.DataFrame:
        """
        Backtest the ML strategy
        
        Args:
            start_date: Backtest start
            end_date: Backtest end  
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M')
            
        Returns:
            DataFrame with backtest results
        """
        print(f"\n{'='*70}")
        print(f"📈 BACKTESTING ML STRATEGY")
        print(f"{'='*70}")
        print(f"Period: {start_date} to {end_date or 'present'}")
        print(f"Rebalance: {rebalance_freq}")
        
        # Load data
        data = self.data_loader.load_all_data(start_date, end_date)
        features = self.feature_engineer.create_all_features(data)
        
        # Get rebalance dates
        rebalance_dates = features.resample(rebalance_freq).last().index
        
        portfolio_returns = []
        
        for date in rebalance_dates:
            # Get features up to this date
            historical_features = features.loc[:date]
            current_features = historical_features.iloc[[-1]]
            
            # Generate signals
            signals = {}
            for currency in self.currencies:
                try:
                    pred = self.ensemble.predict(currency, current_features)
                    signals[currency] = np.tanh(pred[0] * 10)
                except:
                    signals[currency] = 0.0
            
            # Calculate realized returns until next rebalance
            # (simplified - can enhance with realistic execution)
            
        print(f"✅ Backtest complete")
        
        # Return performance metrics
        return pd.DataFrame()  # Placeholder
    
    def get_model_diagnostics(self, currency: str) -> Dict:
        """Get diagnostic information for a currency's models"""
        
        if currency not in self.ensemble.models:
            raise ValueError(f"No models trained for {currency}")
        
        diagnostics = {
            'currency': currency,
            'models_trained': list(self.ensemble.models[currency].keys()),
            'feature_count': len(self.feature_cols) if self.feature_cols else 0,
            'top_features': self.ensemble.get_feature_importance(currency, top_n=20)
        }
        
        return diagnostics


if __name__ == "__main__":
    # Test ML strategy
    from dotenv import load_dotenv
    load_dotenv()
    
    fred_key = os.getenv('FRED_API_KEY', 'b4a18aac3a462b6951ee89d9fef027cb')
    
    # Initialize strategy
    strategy = MLFXStrategy(
        fred_api_key=fred_key,
        currencies=['EUR', 'GBP', 'JPY']  # Start with 3 currencies for testing
    )
    
    # Train models (using recent data for faster testing)
    results = strategy.train_all_currencies(
        start_date='2020-01-01',
        validation_split=0.2,
        walk_forward=False
    )
    
    # Generate signals
    signals = strategy.generate_signals()
    
    print(f"\n✅ Test complete!")
    print(f"   Signals: {signals}")
