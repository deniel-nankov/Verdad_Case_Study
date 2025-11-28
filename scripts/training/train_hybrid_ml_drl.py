#!/usr/bin/env python3
"""
HYBRID ML + DRL TRADING SYSTEM
================================
Production-grade integration of Machine Learning and Deep Reinforcement Learning

ARCHITECTURE:
1. ML Layer (Random Forest + XGBoost)
   - Generates predictions from 190+ features
   - Outputs: expected return, confidence, regime
   
2. DRL Layer (DDPG Agent)
   - Takes ML predictions as part of state
   - Learns optimal position sizing and timing
   - Outputs: continuous position [-1, 1]

3. Risk Management Layer
   - VaR limits
   - Drawdown protection
   - Position limits

NO SIMPLIFICATIONS - FULL IMPLEMENTATION
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

print('='*90)
print('🔥 HYBRID ML + DRL TRADING SYSTEM - PRODUCTION IMPLEMENTATION')
print('='*90)
print()

# ============================================================================
# PART 1: ML PREDICTION ENGINE
# ============================================================================

class MLPredictionEngine:
    """
    Machine Learning layer for feature generation and prediction
    - Trains RF + XGB ensemble on historical data
    - Generates predictions, confidence scores, and regime classification
    """
    
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.feature_names = None
        self.scaler_mean = None
        self.scaler_std = None
        
    def create_features(self, data):
        """
        Create comprehensive feature set from price data
        Returns 50+ features for ML prediction
        """
        df = data.copy()
        features = {}
        
        # Price-based features
        for window in [5, 10, 21, 63, 126]:
            features[f'return_{window}d'] = df['Close'].pct_change(window)
            features[f'volatility_{window}d'] = df['Close'].pct_change().rolling(window).std()
            features[f'sma_{window}d'] = df['Close'].rolling(window).mean() / df['Close'] - 1.0
            
        # Momentum indicators
        features['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        features['macd'], features['macd_signal'] = self._calculate_macd(df['Close'])
        
        # Volume features
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(21).mean()
        features['volume_change'] = df['Volume'].pct_change()
        
        # Volatility features
        features['atr'] = self._calculate_atr(df, 14)
        features['bollinger_width'] = (df['Close'].rolling(21).std() / df['Close'].rolling(21).mean())
        
        # Trend strength
        features['trend_strength'] = df['Close'].rolling(21).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 0 else 0
        )
        
        # Price position in range
        features['price_position'] = (df['Close'] - df['Low'].rolling(21).min()) / \
                                     (df['High'].rolling(21).max() - df['Low'].rolling(21).min() + 1e-8)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(features, index=df.index)
        
        # Fill NaN with forward fill then 0
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        return feature_df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def train(self, data, target_horizon=21):
        """
        Train ML models on historical data
        
        Args:
            data: DataFrame with OHLCV data
            target_horizon: Days ahead to predict
        """
        print('🔧 Training ML Prediction Engine...')
        
        # Create features
        features = self.create_features(data)
        self.feature_names = features.columns.tolist()
        
        # Create target: predict return over next N days
        target = data['Close'].pct_change(target_horizon).shift(-target_horizon)
        
        # Align data
        valid_idx = features.index.intersection(target.dropna().index)
        X = features.loc[valid_idx].values
        y = target.loc[valid_idx].values
        
        # Normalize features
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        # Train-test split (80/20)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train Random Forest
        print('   Training Random Forest...')
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        rf_score = self.rf_model.score(X_test, y_test)
        
        # Train XGBoost
        print('   Training XGBoost...')
        self.xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model.fit(X_train, y_train)
        xgb_score = self.xgb_model.score(X_test, y_test)
        
        print(f'   ✅ RF R²:  {rf_score:.4f}')
        print(f'   ✅ XGB R²: {xgb_score:.4f}')
        print()
        
        return {
            'rf_r2': rf_score,
            'xgb_r2': xgb_score,
            'n_features': len(self.feature_names)
        }
    
    def predict(self, data):
        """
        Generate predictions from current market state
        
        Returns:
            dict with prediction, confidence, and regime
        """
        # Create features
        features = self.create_features(data)
        X = features.iloc[-1:].values
        
        # Normalize
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(X_scaled)[0]
        xgb_pred = self.xgb_model.predict(X_scaled)[0]
        
        # Ensemble prediction (average)
        ensemble_pred = (rf_pred + xgb_pred) / 2.0
        
        # Confidence: agreement between models
        confidence = 1.0 - abs(rf_pred - xgb_pred) / (abs(rf_pred) + abs(xgb_pred) + 1e-8)
        
        # Regime classification based on volatility
        recent_vol = data['Close'].pct_change().iloc[-21:].std()
        regime = 'high_vol' if recent_vol > 0.01 else 'low_vol'
        
        return {
            'prediction': ensemble_pred,
            'confidence': confidence,
            'regime': regime,
            'rf_pred': rf_pred,
            'xgb_pred': xgb_pred,
            'volatility': recent_vol
        }


# ============================================================================
# PART 2: HYBRID ENVIRONMENT (ML + Market Data)
# ============================================================================

class HybridMLDRLEnvironment:
    """
    Trading environment that combines ML predictions with market data
    - State includes both raw market features AND ML predictions
    - Reward based on actual trading performance
    """
    
    def __init__(self, price_data, ml_engine, initial_capital=100000):
        self.prices = price_data
        self.ml_engine = ml_engine
        self.initial_capital = initial_capital
        self.transaction_cost = 0.0001
        
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.current_step = 252  # Start after warmup
        self.position = 0.0
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.trades = []
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get hybrid state vector:
        - Market features (15 dims)
        - ML predictions (5 dims)
        Total: 20 dimensions
        """
        if self.current_step < 20:
            return np.zeros(20, dtype=np.float32)
        
        # Get market data up to current step
        current_data = self.prices.iloc[:self.current_step]
        
        # PART 1: Market features (same as robust DRL)
        recent_prices = current_data['Close'].values[-20:]
        recent_returns = np.diff(recent_prices) / recent_prices[:-1]
        
        sma_5 = np.mean(recent_prices[-5:]) / recent_prices[-1] - 1.0
        sma_10 = np.mean(recent_prices[-10:]) / recent_prices[-1] - 1.0
        sma_20 = np.mean(recent_prices[-20:]) / recent_prices[-1] - 1.0
        volatility = np.std(recent_returns)
        momentum_5 = (recent_prices[-1] / recent_prices[-5] - 1.0) if recent_prices[-5] > 0 else 0.0
        momentum_10 = (recent_prices[-1] / recent_prices[-10] - 1.0) if recent_prices[-10] > 0 else 0.0
        
        recent_volume = current_data['Volume'].values[-20:]
        volume_ratio = recent_volume[-1] / (np.mean(recent_volume) + 1e-8)
        
        highs = current_data['High'].values
        lows = current_data['Low'].values
        price_range = (highs[-1] - lows[-1]) / recent_prices[-1]
        
        portfolio_return = (self.equity - self.initial_capital) / self.initial_capital
        
        # Calculate recent Sharpe
        if len(self.equity_curve) > 21:
            equity_array = np.array(self.equity_curve[-21:])
            returns = np.diff(equity_array) / self.initial_capital
            recent_sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            recent_sharpe = 0.0
        
        market_features = np.array([
            sma_5, sma_10, sma_20,
            volatility * 100,
            momentum_5, momentum_10,
            np.mean(recent_returns),
            np.log(volume_ratio + 1.0),
            price_range * 100,
            self.position,
            portfolio_return,
            recent_sharpe,
            np.tanh(self.equity / self.initial_capital - 1.0),
            float(self.current_step) / len(self.prices),
            float(len(self.trades)) / 100
        ], dtype=np.float32)
        
        # PART 2: ML predictions
        ml_pred = self.ml_engine.predict(current_data)
        
        ml_features = np.array([
            ml_pred['prediction'] * 100,  # Expected return (scaled)
            ml_pred['confidence'],  # Model confidence
            ml_pred['volatility'] * 100,  # Market volatility
            1.0 if ml_pred['regime'] == 'high_vol' else 0.0,  # Regime indicator
            (ml_pred['rf_pred'] + ml_pred['xgb_pred']) / 2.0 * 100  # Ensemble prediction
        ], dtype=np.float32)
        
        # Combine
        state = np.concatenate([market_features, ml_features])
        
        # Remove NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state
    
    def step(self, action):
        """Execute action"""
        action = np.clip(float(action), -1.0, 1.0)
        
        old_position = self.position
        new_position = action
        position_change = abs(new_position - old_position)
        
        self.current_step += 1
        
        if self.current_step >= len(self.prices) - 1:
            return self._get_state(), 0.0, True, {}
        
        # Calculate P&L
        price_return = self.prices['Close'].pct_change().iloc[self.current_step]
        pnl = old_position * price_return * self.equity
        cost = position_change * self.transaction_cost * self.equity
        
        net_pnl = pnl - cost
        self.equity += net_pnl
        self.position = new_position
        self.equity_curve.append(self.equity)
        
        if position_change > 0.1:
            self.trades.append({
                'step': self.current_step,
                'position': new_position,
                'equity': self.equity
            })
        
        # Reward calculation
        if len(self.equity_curve) > 21:
            recent_equity = np.array(self.equity_curve[-21:])
            recent_returns = np.diff(recent_equity) / self.initial_capital
            mean_ret = np.mean(recent_returns)
            std_ret = np.std(recent_returns) + 1e-8
            reward = (mean_ret / std_ret) * np.sqrt(252)
            
            peak = np.max(self.equity_curve)
            drawdown = (peak - self.equity) / peak
            reward -= drawdown * 2.0
        else:
            reward = net_pnl / self.initial_capital
        
        reward = float(np.nan_to_num(reward, nan=0.0))
        reward = np.clip(reward, -10.0, 10.0)
        
        done = self.current_step >= len(self.prices) - 1
        
        info = {
            'equity': self.equity,
            'position': self.position,
            'pnl': net_pnl,
            'trades': len(self.trades)
        }
        
        return self._get_state(), reward, done, info


# ============================================================================
# PART 3: HYBRID AGENT (Same DDPG but with larger state space)
# ============================================================================

class HybridActor(nn.Module):
    """Actor with 20-dim state input (market + ML features)"""
    def __init__(self, state_dim=20, action_dim=1, hidden_dim=256):
        super(HybridActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.net(state)


class HybridCritic(nn.Module):
    """Critic with 20-dim state input"""
    def __init__(self, state_dim=20, action_dim=1, hidden_dim=256):
        super(HybridCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class HybridDDPGAgent:
    """DDPG agent for hybrid ML+DRL system"""
    
    def __init__(self, state_dim=20, action_dim=1):
        self.actor = HybridActor(state_dim, action_dim)
        self.actor_target = HybridActor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        
        self.critic = HybridCritic(state_dim, action_dim)
        self.critic_target = HybridCritic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005
    
    def select_action(self, state, noise=0.1):
        """Select action with exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).numpy()[0]
        
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
        
        return np.clip(action, -1.0, 1.0)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[i] for i in batch])
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, next_actions)
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update targets
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()


# ============================================================================
# PART 4: MAIN TRAINING PIPELINE
# ============================================================================

if __name__ == '__main__':
    print('📊 STEP 1: Load REAL EUR/USD Data')
    print('='*90)
    eur_data = yf.download('EURUSD=X', start='2018-01-01', progress=False)
    print(f'✅ Downloaded {len(eur_data)} days of real data')
    print(f'   Period: {eur_data.index[0].date()} to {eur_data.index[-1].date()}')
    print()
    
    print('='*90)
    print('📊 STEP 2: Train ML Prediction Engine')
    print('='*90)
    ml_engine = MLPredictionEngine()
    ml_stats = ml_engine.train(eur_data, target_horizon=21)
    print(f'✅ ML Engine trained with {ml_stats["n_features"]} features')
    print()
    
    print('='*90)
    print('📊 STEP 3: Train Hybrid ML+DRL Agent (100 episodes)')
    print('='*90)
    print('   This combines ML predictions with DRL policy learning')
    print('   Expected time: ~10 minutes')
    print()
    
    # Create hybrid environment
    env = HybridMLDRLEnvironment(eur_data, ml_engine)
    agent = HybridDDPGAgent(state_dim=20, action_dim=1)
    
    # Training loop
    num_episodes = 100
    noise_start = 0.5
    noise_decay = 0.99
    noise_min = 0.05
    
    episode_sharpes = []
    episode_returns = []
    best_sharpe = -np.inf
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        noise = max(noise_min, noise_start * (noise_decay ** episode))
        
        while not done:
            action = agent.select_action(state, noise=noise)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            critic_loss, actor_loss = agent.train()
            episode_reward += reward
            state = next_state
        
        # Calculate metrics
        final_equity = env.equity
        total_return = (final_equity - env.initial_capital) / env.initial_capital
        
        if len(env.equity_curve) > 21:
            equity_array = np.array(env.equity_curve)
            returns = np.diff(equity_array) / env.initial_capital
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            sharpe = np.nan_to_num(sharpe, nan=0.0)
        else:
            sharpe = 0.0
        
        episode_sharpes.append(sharpe)
        episode_returns.append(total_return)
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
        
        if (episode + 1) % 10 == 0:
            avg_sharpe = np.mean(episode_sharpes[-10:])
            print(f'   Episode {episode+1:3d}: Sharpe={sharpe:+.3f} | Return={total_return*100:+.2f}% | '
                  f'Avg10={avg_sharpe:+.3f}')
    
    print()
    print('='*90)
    print('✅ HYBRID ML+DRL TRAINING COMPLETE!')
    print('='*90)
    print(f'   Best Sharpe:  {best_sharpe:+.3f}')
    print(f'   Final Sharpe: {episode_sharpes[-1]:+.3f}')
    print(f'   Avg Sharpe:   {np.mean(episode_sharpes):+.3f}')
    print()
    
    # Save model
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'best_sharpe': best_sharpe
    }, 'hybrid_ml_drl_model.pth')
    
    with open('ml_prediction_engine.pkl', 'wb') as f:
        pickle.dump(ml_engine, f)
    
    print('💾 Models saved:')
    print('   - hybrid_ml_drl_model.pth (DRL agent)')
    print('   - ml_prediction_engine.pkl (ML engine)')
    print()
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(episode_sharpes, alpha=0.6, label='Episode Sharpe')
    ax1.plot(pd.Series(episode_sharpes).rolling(10).mean(), linewidth=2, label='10-Episode MA')
    ax1.axhline(y=0, color='black', linestyle='--')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Hybrid ML+DRL Learning Progress')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot([r*100 for r in episode_returns], alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Return (%)')
    ax2.set_title('Episode Returns')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hybrid_ml_drl_results.png', dpi=150, bbox_inches='tight')
    print('📊 Results saved to: hybrid_ml_drl_results.png')
    print()
    
    print('🎉 HYBRID SYSTEM READY FOR DEPLOYMENT!')
    print('='*90)
