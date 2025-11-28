#!/usr/bin/env python3
"""
TRAIN PROPER DRL AGENT FOR BACKTESTING
========================================
Trains a robust DDPG agent specifically for the backtesting system
Uses the same data and features as the backtest for consistency
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from sklearn.preprocessing import StandardScaler

print('='*80)
print('🤖 TRAINING DRL AGENT FOR BACKTESTING')
print('='*80)
print()

# Load data (same as backtest)
print('📊 Loading EUR/USD data...')
eur_data = yf.download('EURUSD=X', start='2018-01-01', progress=False)
eur_data['returns'] = eur_data['Close'].pct_change()
eur_data = eur_data.dropna()

# Split (same as backtest)
split_date = '2024-01-01'
train_data = eur_data[eur_data.index < split_date].copy()
print(f'✅ Training data: {len(train_data)} days')
print()

# Feature engineering
def create_state(data, idx, lookback=20):
    """Create state vector from market data"""
    if idx < lookback:
        return np.zeros(15)
    
    recent_prices = data['Close'].iloc[max(0, idx-lookback):idx].values
    recent_returns = data['returns'].iloc[max(0, idx-lookback):idx].values
    recent_volume = data['Volume'].iloc[max(0, idx-lookback):idx].values
    
    # Handle 2D arrays
    if recent_prices.ndim > 1:
        recent_prices = recent_prices.ravel()
    if recent_returns.ndim > 1:
        recent_returns = recent_returns.ravel()
    if recent_volume.ndim > 1:
        recent_volume = recent_volume.ravel()
    
    # Returns statistics
    mean_ret = float(np.mean(recent_returns))
    std_ret = float(np.std(recent_returns))
    skew_ret = float(np.mean(((recent_returns - mean_ret) / (std_ret + 1e-8)) ** 3))
    
    # Momentum features
    mom_1 = float(recent_prices[-1] / recent_prices[-2] - 1.0) if len(recent_prices) >= 2 else 0.0
    mom_5 = float(recent_prices[-1] / recent_prices[-5] - 1.0) if len(recent_prices) >= 5 else 0.0
    mom_10 = float(recent_prices[-1] / recent_prices[-10] - 1.0) if len(recent_prices) >= 10 else 0.0
    
    # Moving averages
    sma_5 = np.mean(recent_prices[-5:]) if len(recent_prices) >= 5 else recent_prices[-1]
    sma_10 = np.mean(recent_prices[-10:]) if len(recent_prices) >= 10 else recent_prices[-1]
    if isinstance(sma_5, np.ndarray):
        sma_5 = sma_5[0]
    if isinstance(sma_10, np.ndarray):
        sma_10 = sma_10[0]
    
    ma_cross = float(sma_5 / sma_10 - 1.0)
    dist_sma5 = float(recent_prices[-1] / sma_5 - 1.0)
    dist_sma10 = float(recent_prices[-1] / sma_10 - 1.0)
    
    # Volatility
    vol_5 = float(np.std(recent_returns[-5:])) if len(recent_returns) >= 5 else 0.0
    vol_10 = float(np.std(recent_returns[-10:])) if len(recent_returns) >= 10 else 0.0
    
    # Volume
    vol_ratio = float(recent_volume[-1] / (np.mean(recent_volume) + 1e-8))
    
    # Range
    high = float(np.max(recent_prices))
    low = float(np.min(recent_prices))
    price_position = (recent_prices[-1] - low) / (high - low + 1e-8)
    
    state = np.array([
        mean_ret, std_ret, skew_ret,
        mom_1, mom_5, mom_10,
        ma_cross, dist_sma5, dist_sma10,
        vol_5, vol_10, vol_ratio,
        price_position,
        0.0,  # Current position (placeholder)
        0.0   # P&L (placeholder)
    ], dtype=np.float64)
    
    return np.nan_to_num(state, nan=0.0)

# Simple policy gradient agent
class PolicyGradientAgent:
    """Simple but effective policy gradient agent"""
    def __init__(self, state_dim=15, learning_rate=0.001):
        self.state_dim = state_dim
        self.lr = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(state_dim, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 16) * 0.1
        self.b2 = np.zeros(16)
        self.W3 = np.random.randn(16, 1) * 0.1
        self.b3 = np.zeros(1)
        
        # Scaler for states
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
    def forward(self, state):
        """Forward pass"""
        # Scale state
        if self.scaler_fitted:
            state = self.scaler.transform(state.reshape(1, -1)).ravel()
        
        # Layer 1
        h1 = np.maximum(0, np.dot(state, self.W1) + self.b1)  # ReLU
        
        # Layer 2
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)  # ReLU
        
        # Output
        output = np.dot(h2, self.W3) + self.b3
        
        # Tanh activation for bounded output
        action = np.tanh(output[0])
        
        return np.clip(action, -1.0, 1.0), h1, h2
    
    def backward(self, state, action, reward, h1, h2):
        """Backward pass - policy gradient"""
        # Scale state
        if self.scaler_fitted:
            state = self.scaler.transform(state.reshape(1, -1)).ravel()
        
        # Gradient of tanh
        dtanh = 1 - action ** 2
        
        # Output layer gradients
        dW3 = reward * dtanh * h2.reshape(-1, 1)
        db3 = reward * dtanh
        
        # Hidden layer 2 gradients
        dh2 = reward * dtanh * self.W3.ravel()
        dh2[h2 <= 0] = 0  # ReLU gradient
        dW2 = np.outer(h1, dh2)
        db2 = dh2
        
        # Hidden layer 1 gradients
        dh1 = np.dot(dh2, self.W2.T)
        dh1[h1 <= 0] = 0  # ReLU gradient
        dW1 = np.outer(state, dh1)
        db1 = dh1
        
        # Update weights
        self.W3 += self.lr * dW3
        self.b3 += self.lr * db3
        self.W2 += self.lr * dW2
        self.b2 += self.lr * db2
        self.W1 += self.lr * dW1
        self.b1 += self.lr * db1
    
    def predict(self, state):
        """Predict action"""
        action, _, _ = self.forward(state)
        return action

# Training
print('🔧 Training DRL agent...')
print()

agent = PolicyGradientAgent(state_dim=15, learning_rate=0.0005)

# Fit scaler on all training states
print('📊 Fitting state scaler...')
all_states = []
for i in range(50, len(train_data)):
    state = create_state(train_data, i)
    all_states.append(state)
agent.scaler.fit(np.array(all_states))
agent.scaler_fitted = True
print(f'   ✅ Scaler fitted on {len(all_states)} states')
print()

# Training loop
num_episodes = 200
transaction_cost = 0.0001

print(f'🚀 Training for {num_episodes} episodes...')
print()

best_sharpe = -999
sharpes = []

for episode in range(num_episodes):
    # Reset
    position = 0.0
    equity = 1.0
    equity_curve = [1.0]
    returns_list = []
    
    # Episode
    for i in range(50, len(train_data), 2):  # Skip every other day for speed
        state = create_state(train_data, i)
        action, h1, h2 = agent.forward(state)
        
        # Execute action
        new_position = action * 0.5  # Scale down position
        position_change = abs(new_position - position)
        
        # Get return
        if i + 1 < len(train_data):
            ret = float(train_data['returns'].iloc[i+1])
            if isinstance(ret, pd.Series):
                ret = ret.values[0]
            
            # Calculate P&L
            pnl = position * ret - position_change * transaction_cost
            equity *= (1 + pnl)
            equity_curve.append(equity)
            returns_list.append(pnl)
            
            # Calculate reward (Sharpe-based)
            if len(returns_list) >= 10:
                recent_rets = returns_list[-10:]
                mean_r = np.mean(recent_rets)
                std_r = np.std(recent_rets) + 1e-8
                sharpe = mean_r / std_r * np.sqrt(252)
                reward = sharpe * 0.1  # Scale reward
            else:
                reward = pnl * 100
            
            # Update policy
            agent.backward(state, action, reward, h1, h2)
            
            position = new_position
    
    # Episode metrics
    total_return = (equity - 1.0) * 100
    if len(returns_list) > 0:
        mean_ret = np.mean(returns_list)
        std_ret = np.std(returns_list) + 1e-8
        sharpe = mean_ret / std_ret * np.sqrt(252)
        sharpes.append(sharpe)
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
    else:
        sharpe = 0.0
    
    if episode % 20 == 0:
        avg_sharpe = np.mean(sharpes[-20:]) if len(sharpes) >= 20 else np.mean(sharpes)
        print(f'Episode {episode:3d}: Return={total_return:+.2f}% | Sharpe={sharpe:+.3f} | Avg20={avg_sharpe:+.3f}')

print()
print(f'✅ Training complete!')
print(f'   Best Sharpe: {best_sharpe:.3f}')
print(f'   Final Avg Sharpe: {np.mean(sharpes[-50:]):.3f}')
print()

# Save agent
print('💾 Saving trained agent...')
agent_data = {
    'W1': agent.W1,
    'b1': agent.b1,
    'W2': agent.W2,
    'b2': agent.b2,
    'W3': agent.W3,
    'b3': agent.b3,
    'scaler': agent.scaler,
    'best_sharpe': best_sharpe
}

with open('drl_agent_backtest.pkl', 'wb') as f:
    pickle.dump(agent_data, f)

print('✅ Agent saved to: drl_agent_backtest.pkl')
print()

print('='*80)
print('✅ DRL AGENT READY FOR BACKTESTING!')
print('='*80)
print()
print('📝 To use in backtest:')
print('   1. Load with: pickle.load(open("drl_agent_backtest.pkl", "rb"))')
print('   2. Create agent and load weights')
print('   3. Generate signals on test data')
print()
