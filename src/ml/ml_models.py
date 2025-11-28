"""
Machine Learning Models for FX Trading
Implements Random Forest, XGBoost, and LSTM Neural Network
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MLEnsemble:
    """
    Ensemble of ML models for FX return prediction
    - Random Forest
    - XGBoost
    - LSTM Neural Network
    """
    
    def __init__(self, model_dir: str = "./ml_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.ensemble_weights = {'rf': 0.35, 'xgb': 0.35, 'lstm': 0.30}
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
              currency: str,
              validation_split: float = 0.2,
              optimize: bool = True,
              skip_lstm: bool = False) -> Dict:
        """
        Train all models for a specific currency
        
        Args:
            X: Features (n_samples, n_features)
            y: Target returns (n_samples,)
            currency: Currency code (e.g., 'EUR')
            validation_split: Fraction for validation
            optimize: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with training results
        """
        print(f"\n🤖 Training ML ensemble for {currency}...")
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"  Train: {len(X_train)} samples | Val: {len(X_val)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        self.scalers[currency] = scaler
        
        # Initialize models dictionary for this currency
        self.models[currency] = {}
        self.feature_importance[currency] = {}
        
        results = {}
        
        # 1. Random Forest
        print("  ├─ Training Random Forest...")
        rf_results = self._train_random_forest(
            X_train_scaled, y_train, X_val_scaled, y_val, 
            X_train.columns, optimize
        )
        self.models[currency]['rf'] = rf_results['model']
        self.feature_importance[currency]['rf'] = rf_results['importance']
        results['rf'] = rf_results['metrics']
        
        # 2. XGBoost
        print("  ├─ Training XGBoost...")
        xgb_results = self._train_xgboost(
            X_train_scaled, y_train, X_val_scaled, y_val,
            X_train.columns, optimize
        )
        self.models[currency]['xgb'] = xgb_results['model']
        self.feature_importance[currency]['xgb'] = xgb_results['importance']
        results['xgb'] = xgb_results['metrics']
        
        # 3. LSTM Neural Network (optional - can be skipped for speed)
        if not skip_lstm:
            print("  ├─ Training LSTM...")
            lstm_results = self._train_lstm(
                X_train_scaled, y_train, X_val_scaled, y_val
            )
            self.models[currency]['lstm'] = lstm_results['model']
            results['lstm'] = lstm_results['metrics']
        else:
            print("  ├─ Skipping LSTM (fast mode)")
            # Use average of RF and XGB as placeholder
            results['lstm'] = {
                'r2': (results['rf']['r2'] + results['xgb']['r2']) / 2,
                'mse': (results['rf']['mse'] + results['xgb']['mse']) / 2,
                'rmse': (results['rf']['rmse'] + results['xgb']['rmse']) / 2
            }
        
        # 4. Ensemble performance
        print("  └─ Evaluating ensemble...")
        ensemble_pred = self._ensemble_predict(
            currency, X_val_scaled, X_val.columns
        )
        
        ensemble_r2 = r2_score(y_val, ensemble_pred)
        ensemble_mse = mean_squared_error(y_val, ensemble_pred)
        
        results['ensemble'] = {
            'r2': ensemble_r2,
            'mse': ensemble_mse,
            'rmse': np.sqrt(ensemble_mse)
        }
        
        print(f"\n  ✅ Results:")
        print(f"     RF:       R² = {results['rf']['r2']:.4f}")
        print(f"     XGB:      R² = {results['xgb']['r2']:.4f}")
        if not skip_lstm:
            print(f"     LSTM:     R² = {results['lstm']['r2']:.4f}")
        else:
            print(f"     LSTM:     R² = {results['lstm']['r2']:.4f} (skipped)")
        print(f"     Ensemble: R² = {results['ensemble']['r2']:.4f}")
        
        return results
    
    def _train_random_forest(self, X_train, y_train, X_val, y_val,
                            feature_names, optimize: bool) -> Dict:
        """Train Random Forest model"""
        
        if optimize:
            # Optimized hyperparameters
            params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            # Default parameters
            params = {
                'n_estimators': 100,
                'max_depth': 8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Metrics
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'importance': importance,
            'metrics': {'r2': r2, 'mse': mse, 'rmse': np.sqrt(mse)}
        }
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val,
                       feature_names, optimize: bool) -> Dict:
        """Train XGBoost model"""
        
        if optimize:
            params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.05,
                'random_state': 42,
                'n_jobs': -1
            }
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Metrics
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'importance': importance,
            'metrics': {'r2': r2, 'mse': mse, 'rmse': np.sqrt(mse)}
        }
    
    def _train_lstm(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train LSTM Neural Network (Optimized for speed)"""
        
        print("      Training LSTM (this may take 2-3 minutes)...")
        
        # Reshape for LSTM (samples, timesteps, features)
        # For now, using 1 timestep (can enhance with sequences later)
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        
        # Build SMALLER, FASTER model
        model = keras.Sequential([
            layers.LSTM(32, input_shape=(1, X_train.shape[1])),  # Reduced from 50
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),  # Reduced from 10, removed second LSTM
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),  # Increased LR for faster convergence
            loss='mse',
            metrics=['mae']
        )
        
        # Aggressive early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # Reduced from 10
            restore_best_weights=True
        )
        
        # Train with FEWER epochs
        history = model.fit(
            X_train_lstm, y_train,
            validation_data=(X_val_lstm, y_val),
            epochs=30,  # Reduced from 100
            batch_size=64,  # Increased from 32 for faster training
            callbacks=[early_stop],
            verbose=0
        )
        
        print(f"      LSTM trained in {len(history.history['loss'])} epochs")
        
        # Predict
        y_pred = model.predict(X_val_lstm, verbose=0).flatten()
        
        # Metrics
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        
        return {
            'model': model,
            'metrics': {'r2': r2, 'mse': mse, 'rmse': np.sqrt(mse)},
            'history': history.history
        }
    
    def _ensemble_predict(self, currency: str, X: np.ndarray,
                         feature_names: List[str]) -> np.ndarray:
        """Make ensemble prediction"""
        
        predictions = {}
        
        # Random Forest
        if 'rf' in self.models[currency]:
            predictions['rf'] = self.models[currency]['rf'].predict(X)
        
        # XGBoost
        if 'xgb' in self.models[currency]:
            predictions['xgb'] = self.models[currency]['xgb'].predict(X)
        
        # LSTM
        if 'lstm' in self.models[currency]:
            X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
            predictions['lstm'] = self.models[currency]['lstm'].predict(
                X_lstm, verbose=0
            ).flatten()
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for model_name, pred in predictions.items():
            weight = self.ensemble_weights[model_name]
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict(self, currency: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for a currency
        
        Args:
            currency: Currency code
            X: Features
            
        Returns:
            Predicted returns
        """
        # Scale features
        X_scaled = self.scalers[currency].transform(X)
        
        # Ensemble prediction
        predictions = self._ensemble_predict(currency, X_scaled, X.columns)
        
        return predictions
    
    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series,
                               currency: str,
                               n_splits: int = 5) -> Dict:
        """
        Walk-forward validation for time series
        
        Args:
            X: Features
            y: Target
            currency: Currency code
            n_splits: Number of splits
            
        Returns:
            Validation results
        """
        print(f"\n🔄 Walk-forward validation for {currency} ({n_splits} splits)...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {
            'rf': {'r2': [], 'mse': []},
            'xgb': {'r2': [], 'mse': []},
            'lstm': {'r2': [], 'mse': []},
            'ensemble': {'r2': [], 'mse': []}
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"  Fold {fold}/{n_splits}...", end=' ')
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train models
            fold_results = self.train(
                X_train, y_train, currency,
                validation_split=0.0,  # Already split
                optimize=False
            )
            
            # Store results
            for model_name in ['rf', 'xgb', 'lstm', 'ensemble']:
                results[model_name]['r2'].append(fold_results[model_name]['r2'])
                results[model_name]['mse'].append(fold_results[model_name]['mse'])
            
            print(f"Ensemble R² = {fold_results['ensemble']['r2']:.4f}")
        
        # Calculate average metrics
        summary = {}
        for model_name in ['rf', 'xgb', 'lstm', 'ensemble']:
            summary[model_name] = {
                'avg_r2': np.mean(results[model_name]['r2']),
                'std_r2': np.std(results[model_name]['r2']),
                'avg_mse': np.mean(results[model_name]['mse']),
                'std_mse': np.std(results[model_name]['mse'])
            }
        
        print(f"\n  ✅ Walk-forward results (average ± std):")
        for model_name in ['rf', 'xgb', 'lstm', 'ensemble']:
            avg_r2 = summary[model_name]['avg_r2']
            std_r2 = summary[model_name]['std_r2']
            print(f"     {model_name.upper():8s}: R² = {avg_r2:.4f} ± {std_r2:.4f}")
        
        return summary
    
    def get_feature_importance(self, currency: str, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances across models"""
        
        # Combine RF and XGB importances
        rf_imp = self.feature_importance[currency]['rf'].copy()
        xgb_imp = self.feature_importance[currency]['xgb'].copy()
        
        # Merge and average
        rf_imp.columns = ['feature', 'rf_importance']
        xgb_imp.columns = ['feature', 'xgb_importance']
        
        combined = rf_imp.merge(xgb_imp, on='feature')
        combined['avg_importance'] = (
            combined['rf_importance'] + combined['xgb_importance']
        ) / 2
        
        combined = combined.sort_values('avg_importance', ascending=False)
        
        return combined.head(top_n)
    
    def save_models(self, currency: str):
        """Save trained models to disk"""
        save_dir = f"{self.model_dir}/{currency}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save RF and XGB with joblib
        if 'rf' in self.models[currency]:
            joblib.dump(
                self.models[currency]['rf'],
                f"{save_dir}/random_forest.pkl"
            )
        
        if 'xgb' in self.models[currency]:
            joblib.dump(
                self.models[currency]['xgb'],
                f"{save_dir}/xgboost.pkl"
            )
        
        # Save LSTM with Keras
        if 'lstm' in self.models[currency]:
            self.models[currency]['lstm'].save(f"{save_dir}/lstm_model.keras")
        
        # Save scaler
        joblib.dump(self.scalers[currency], f"{save_dir}/scaler.pkl")
        
        # Save feature importance
        for model_name in ['rf', 'xgb']:
            if model_name in self.feature_importance[currency]:
                self.feature_importance[currency][model_name].to_csv(
                    f"{save_dir}/{model_name}_importance.csv", index=False
                )
        
        print(f"✅ Saved models for {currency} to {save_dir}")
    
    def load_models(self, currency: str):
        """Load trained models from disk"""
        load_dir = f"{self.model_dir}/{currency}"
        
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"No saved models found for {currency}")
        
        self.models[currency] = {}
        
        # Load RF and XGB
        if os.path.exists(f"{load_dir}/random_forest.pkl"):
            self.models[currency]['rf'] = joblib.load(
                f"{load_dir}/random_forest.pkl"
            )
        
        if os.path.exists(f"{load_dir}/xgboost.pkl"):
            self.models[currency]['xgb'] = joblib.load(
                f"{load_dir}/xgboost.pkl"
            )
        
        # Load LSTM
        if os.path.exists(f"{load_dir}/lstm_model.keras"):
            self.models[currency]['lstm'] = keras.models.load_model(
                f"{load_dir}/lstm_model.keras"
            )
        
        # Load scaler
        self.scalers[currency] = joblib.load(f"{load_dir}/scaler.pkl")
        
        print(f"✅ Loaded models for {currency} from {load_dir}")


if __name__ == "__main__":
    # Test ML ensemble
    print("Testing MLEnsemble...")
    
    # Create dummy data
    n_samples = 1000
    n_features = 50
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Target: simple linear combination + noise
    y = pd.Series(
        X.iloc[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.1
    )
    
    # Train ensemble
    ensemble = MLEnsemble()
    results = ensemble.train(X, y, currency='EUR', validation_split=0.2)
    
    print(f"\n✅ Test complete")
    print(f"   Best model: Ensemble R² = {results['ensemble']['r2']:.4f}")
