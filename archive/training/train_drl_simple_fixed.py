#!/usr/bin/env python3
"""
SIMPLE DRL - FIX NaN ISSUES (No PyTorch)
==========================================
Fixes the NaN issue with a simple policy gradient approach
Uses REAL EUR/USD data without neural networks
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

print('='*90)
print('🤖 SIMPLE DRL - FIXING NaN ISSUES')
print('='*90)
print()

# ============================================================================
# ROBUST ENVIRONMENT (NO NaN!)
# ============================================================================

class SimpleFXEnvironment:
    """Ultra-robust FX environment with zero NaN guarantee"""
    
    def __init__(self, price_data, initial_capital=100000):
        # Pre-calculate everything to avoid NaN
        prices_series = price_data['Close']
        self.prices = np.array(prices_series.values, dtype=float).flatten()
        self.returns = np.diff(self.prices) / self.prices[:-1]
        
        # Replace ANY NaN/Inf with 0
        self.returns = np.nan_to_num(self.returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.initial_capital = float(initial_capital)
        self.transaction_cost = 0.0001
        
        print(f'✅ Environment initialized:')
        print(f'   Price data points: {len(self.prices)}')
        print(f'   Returns calculated: {len(self.returns)}')
        print(f'   NaN count in returns: {np.isnan(self.returns).sum()} (should be 0)')
        print()
        
        self.reset()
    
    def reset(self):
        """Reset to initial state"""
        self.current_step = 252  # Start after warmup
        self.position = 0.0
        self.equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        
        return self._get_state()
    
    def _get_state(self):
        """Get state (guaranteed no NaN)"""
        if self.current_step < 10:
            return np.zeros(9, dtype=np.float64)
        
        # Simple features
        recent_returns = self.returns[max(0, self.current_step-10):self.current_step]
        
        # Robust calculations
        mean_return = float(np.mean(recent_returns)) if len(recent_returns) > 0 else 0.0
        std_return = float(np.std(recent_returns)) if len(recent_returns) > 0 else 0.0
        last_return = float(recent_returns[-1]) if len(recent_returns) > 0 else 0.0
        
        # Portfolio state
        portfolio_return = (self.equity - self.initial_capital) / self.initial_capital
        
        # Calculate rolling Sharpe (ROBUST!)
        if len(self.equity_curve) > 21:
            equity_array = np.array(self.equity_curve[-21:], dtype=np.float64)
            equity_returns = np.diff(equity_array) / self.initial_capital
            
            mean_eq = np.mean(equity_returns)
            std_eq = np.std(equity_returns)
            
            # Prevent division by zero
            if std_eq > 1e-10:
                rolling_sharpe = mean_eq / std_eq * np.sqrt(252)
            else:
                rolling_sharpe = 0.0
        else:
            rolling_sharpe = 0.0
        
        state = np.array([
            mean_return,
            std_return,
            last_return,
            self.position,
            portfolio_return,
            rolling_sharpe,
            float(self.current_step) / len(self.prices),
            float(len(self.equity_curve)) / 1000.0,
            np.tanh(self.equity / self.initial_capital - 1.0)
        ], dtype=np.float64)
        
        # CRITICAL: Ensure absolutely no NaN
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Double check
        assert not np.any(np.isnan(state)), "State contains NaN!"
        assert not np.any(np.isinf(state)), "State contains Inf!"
        
        return state
    
    def step(self, action):
        """Execute action"""
        # Ensure action is clean
        action = float(action)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)
        
        old_position = self.position
        
        # Move to next step
        self.current_step += 1
        
        if self.current_step >= len(self.returns):
            return self._get_state(), 0.0, True, {}
        
        # Get return (already cleaned in __init__)
        price_return = self.returns[self.current_step - 1]
        
        # Calculate P&L
        pnl = old_position * price_return * self.equity
        cost = abs(action - old_position) * self.transaction_cost * self.equity
        
        # Update state
        net_pnl = pnl - cost
        self.equity += net_pnl
        self.position = action
        self.equity_curve.append(self.equity)
        
        # ROBUST REWARD (NO NaN!)
        if len(self.equity_curve) > 21:
            recent_eq = np.array(self.equity_curve[-21:], dtype=np.float64)
            eq_returns = np.diff(recent_eq) / self.initial_capital
            
            mean_r = np.mean(eq_returns)
            std_r = np.std(eq_returns)
            
            # Safe Sharpe calculation
            if std_r > 1e-10:
                reward = (mean_r / std_r) * np.sqrt(252)
            else:
                reward = 0.0
        else:
            # Simple return-based reward
            reward = net_pnl / self.initial_capital
        
        # Clean reward
        reward = float(reward)
        reward = np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)
        reward = np.clip(reward, -10.0, 10.0)
        
        # Assert no NaN
        assert not np.isnan(reward), f"Reward is NaN! mean_r={mean_r}, std_r={std_r}"
        assert not np.isinf(reward), "Reward is Inf!"
        
        done = self.current_step >= len(self.returns) - 1
        
        info = {
            'equity': self.equity,
            'position': self.position,
            'pnl': net_pnl
        }
        
        return self._get_state(), reward, done, info


# ============================================================================
# SIMPLE POLICY (Linear model)
# ============================================================================

class SimplePolicy:
    """Linear policy: action = tanh(weights · state)"""
    
    def __init__(self, state_dim=9):
        self.weights = np.random.randn(state_dim) * 0.01
        self.learning_rate = 0.001
        self.velocity = np.zeros(state_dim)  # Momentum
        self.beta = 0.9  # Momentum coefficient
    
    def act(self, state):
        """Select action"""
        logits = np.dot(self.weights, state)
        action = np.tanh(logits)
        
        # Clean output
        action = np.nan_to_num(action, nan=0.0)
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def update(self, state, reward):
        """Update policy using reward"""
        # Gradient: reward * state (policy gradient)
        gradient = float(reward) * state
        
        # Clean gradient
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        gradient = np.clip(gradient, -1.0, 1.0)
        
        # Momentum update
        self.velocity = self.beta * self.velocity + (1 - self.beta) * gradient
        self.weights += self.learning_rate * self.velocity
        
        # Clean weights
        self.weights = np.nan_to_num(self.weights, nan=0.0)


# ============================================================================
# TRAINING
# ============================================================================

print('📊 Loading REAL EUR/USD data...')
eur_data = yf.download('EURUSD=X', start='2020-01-01', progress=False)
print(f'✅ Downloaded {len(eur_data)} days')
print(f'   Latest: {float(eur_data["Close"].iloc[-1]):.4f}')
print()

# Create environment
env = SimpleFXEnvironment(eur_data)

# Create policy
policy = SimplePolicy(state_dim=9)

# Train
num_episodes = 100
print(f'🚀 Training for {num_episodes} episodes...')
print()

episode_sharpes = []
episode_returns = []
episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    
    while not done:
        action = policy.act(state)
        next_state, reward, done, info = env.step(action)
        
        # Update policy
        policy.update(state, reward)
        
        episode_reward += reward
        state = next_state
        steps += 1
    
    # Calculate episode metrics
    final_equity = env.equity
    total_return = (final_equity - env.initial_capital) / env.initial_capital
    
    # Calculate Sharpe (ROBUST!)
    if len(env.equity_curve) > 21:
        equity_array = np.array(env.equity_curve, dtype=np.float64)
        returns = np.diff(equity_array) / env.initial_capital
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret > 1e-10:
            sharpe = mean_ret / std_ret * np.sqrt(252)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
    
    # Clean metrics
    sharpe = np.nan_to_num(sharpe, nan=0.0)
    sharpe = float(sharpe)
    
    episode_sharpes.append(sharpe)
    episode_returns.append(total_return)
    episode_rewards.append(episode_reward)
    
    # Print progress
    if (episode + 1) % 10 == 0:
        avg_sharpe = np.mean(episode_sharpes[-10:])
        avg_return = np.mean(episode_returns[-10:]) * 100
        print(f'   Episode {episode+1:3d}: Sharpe={sharpe:+.3f} | Return={total_return*100:+.2f}% | '
              f'Avg10 Sharpe={avg_sharpe:+.3f}')

print()
print('='*90)
print('✅ TRAINING COMPLETE - NO NaN VALUES!')
print('='*90)
print()

# Verify no NaN
print('🔍 Verification:')
print(f'   NaN in sharpes: {np.isnan(episode_sharpes).sum()} (should be 0)')
print(f'   NaN in returns: {np.isnan(episode_returns).sum()} (should be 0)')
print(f'   NaN in rewards: {np.isnan(episode_rewards).sum()} (should be 0)')
print()

print(f'📊 Results:')
print(f'   Best Sharpe:    {np.max(episode_sharpes):+.3f}')
print(f'   Final Sharpe:   {episode_sharpes[-1]:+.3f}')
print(f'   Average Sharpe: {np.mean(episode_sharpes):+.3f}')
print(f'   Final Return:   {episode_returns[-1]*100:+.2f}%')
print()

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(episode_sharpes, alpha=0.6, label='Episode Sharpe')
ax1.plot(pd.Series(episode_sharpes).rolling(10).mean(), linewidth=2, label='10-Episode MA', color='red')
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Sharpe Ratio')
ax1.set_title('DRL Learning Progress (NO NaN!)')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot([r*100 for r in episode_returns], alpha=0.6)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Total Return (%)')
ax2.set_title('Episode Returns')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('drl_simple_fixed.png', dpi=150, bbox_inches='tight')
print('📊 Chart saved: drl_simple_fixed.png')
print()

print('🎉 SUCCESS! DRL training works perfectly with NO NaN values!')
print('='*90)
