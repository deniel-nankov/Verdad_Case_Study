#!/usr/bin/env python3
"""
ROBUST DRL TRAINING - FIX NaN ISSUES
=====================================
Proper implementation of Deep Reinforcement Learning for FX trading
- Fix NaN reward calculation
- Use DDPG with experience replay
- Train on REAL EUR/USD data
- 200 episodes with proper convergence tracking

NO SHORTCUTS - PRODUCTION QUALITY
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

print('='*90)
print('🤖 ROBUST DRL TRAINING - FIXING NaN ISSUES')
print('='*90)
print()

# ============================================================================
# PART 1: FIX THE ENVIRONMENT (No more NaN!)
# ============================================================================

class RobustFXEnvironment:
    """
    Production-grade FX trading environment
    - Handles edge cases (no NaN)
    - Realistic transaction costs
    - Proper reward normalization
    """
    
    def __init__(self, price_data, initial_capital=100000, transaction_cost=0.0001):
        self.prices = price_data['Close'].values
        self.volumes = price_data['Volume'].values
        self.highs = price_data['High'].values
        self.lows = price_data['Low'].values
        
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Calculate returns ahead of time (avoid NaN)
        self.returns = np.diff(self.prices) / self.prices[:-1]
        self.returns = np.nan_to_num(self.returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 252  # Start after 1 year warmup
        self.position = 0.0
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.trades = []
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state (NO NaN values!)
        Returns 15-dimensional state vector with proper normalization
        """
        if self.current_step < 20:
            return np.zeros(15, dtype=np.float32)
        
        # Price features (last 20 days)
        recent_prices = self.prices[self.current_step-20:self.current_step]
        recent_returns = self.returns[self.current_step-20:self.current_step]
        
        # Technical indicators
        sma_5 = np.mean(recent_prices[-5:]) / recent_prices[-1] - 1.0
        sma_10 = np.mean(recent_prices[-10:]) / recent_prices[-1] - 1.0
        sma_20 = np.mean(recent_prices[-20:]) / recent_prices[-1] - 1.0
        
        # Volatility (robust calculation)
        volatility = np.std(recent_returns) if len(recent_returns) > 0 else 0.0
        
        # Momentum
        momentum_5 = (recent_prices[-1] / recent_prices[-5] - 1.0) if recent_prices[-5] > 0 else 0.0
        momentum_10 = (recent_prices[-1] / recent_prices[-10] - 1.0) if recent_prices[-10] > 0 else 0.0
        
        # Volume features
        recent_volume = self.volumes[self.current_step-20:self.current_step]
        volume_ratio = recent_volume[-1] / (np.mean(recent_volume) + 1e-8)
        
        # Price range
        price_range = (self.highs[self.current_step-1] - self.lows[self.current_step-1]) / self.prices[self.current_step-1]
        
        # Portfolio state
        portfolio_return = (self.equity - self.initial_capital) / self.initial_capital
        
        # Recent performance (last 21 days)
        if len(self.equity_curve) > 21:
            recent_equity = np.array(self.equity_curve[-21:])
            recent_pnl = np.diff(recent_equity) / self.initial_capital
            recent_sharpe = np.mean(recent_pnl) / (np.std(recent_pnl) + 1e-8) * np.sqrt(252)
        else:
            recent_sharpe = 0.0
        
        # Construct state vector
        state = np.array([
            sma_5,
            sma_10,
            sma_20,
            volatility * 100,  # Scale up
            momentum_5,
            momentum_10,
            np.mean(recent_returns),
            np.log(volume_ratio + 1.0),  # Log scale
            price_range * 100,
            self.position,
            portfolio_return,
            recent_sharpe,
            np.tanh(self.equity / self.initial_capital - 1.0),  # Bounded
            float(self.current_step) / len(self.prices),  # Time progress
            float(len(self.trades)) / 100  # Trade count (normalized)
        ], dtype=np.float32)
        
        # CRITICAL: Remove any NaN/Inf values
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info)
        
        Args:
            action: Float in [-1, 1] representing position size
        
        Returns:
            next_state, reward, done, info
        """
        # Clip action to valid range
        action = np.clip(float(action), -1.0, 1.0)
        
        # Calculate position change
        old_position = self.position
        new_position = action
        position_change = abs(new_position - old_position)
        
        # Get next price
        self.current_step += 1
        
        if self.current_step >= len(self.prices) - 1:
            # Episode done
            return self._get_state(), 0.0, True, {}
        
        # Calculate P&L from price movement
        price_return = self.returns[self.current_step - 1]
        pnl = old_position * price_return * self.equity
        
        # Transaction cost
        cost = position_change * self.transaction_cost * self.equity
        
        # Update equity
        net_pnl = pnl - cost
        self.equity += net_pnl
        self.position = new_position
        self.equity_curve.append(self.equity)
        
        # Record trade if position changed significantly
        if position_change > 0.1:
            self.trades.append({
                'step': self.current_step,
                'position': new_position,
                'price': self.prices[self.current_step],
                'equity': self.equity
            })
        
        # ROBUST REWARD CALCULATION (NO NaN!)
        # Use instantaneous Sharpe ratio over last 21 days
        if len(self.equity_curve) > 21:
            recent_equity = np.array(self.equity_curve[-21:])
            recent_returns = np.diff(recent_equity) / self.initial_capital
            
            mean_ret = np.mean(recent_returns)
            std_ret = np.std(recent_returns) + 1e-8  # Prevent division by zero
            
            # Annualized Sharpe
            reward = (mean_ret / std_ret) * np.sqrt(252)
            
            # Add penalty for drawdown
            peak = np.max(self.equity_curve)
            drawdown = (peak - self.equity) / peak
            reward -= drawdown * 2.0  # Penalty for being in drawdown
            
        else:
            # Early in episode: use simple return
            reward = net_pnl / self.initial_capital
        
        # CRITICAL: Ensure reward is not NaN
        reward = float(reward)
        reward = np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip reward to reasonable range
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
# PART 2: DDPG AGENT (Actor-Critic with Experience Replay)
# ============================================================================

class Actor(nn.Module):
    """Actor network: maps state → action"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """Critic network: maps (state, action) → Q-value"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient Agent
    - Actor-Critic architecture
    - Experience replay buffer
    - Target networks for stability
    """
    
    def __init__(self, state_dim, action_dim, lr_actor=0.0001, lr_critic=0.001, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic networks
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Experience replay
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 64
        
    def select_action(self, state, noise=0.1):
        """Select action with exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        
        # Add Ornstein-Uhlenbeck noise for exploration
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
        
        return np.clip(action, -1.0, 1.0)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train agent using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # Sample batch
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
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()


# ============================================================================
# PART 3: TRAINING LOOP (200 EPISODES)
# ============================================================================

print('📊 Loading REAL EUR/USD data from Yahoo Finance...')
eur_data = yf.download('EURUSD=X', start='2020-01-01', progress=False)
print(f'✅ Downloaded {len(eur_data)} days of real EUR/USD prices')
print(f'   Latest: {float(eur_data["Close"].iloc[-1]):.4f} (as of {eur_data.index[-1].date()})')
print()

# Create environment
env = RobustFXEnvironment(eur_data)
state_dim = 15
action_dim = 1

# Create agent
agent = DDPGAgent(state_dim, action_dim)

# Training parameters
num_episodes = 200
noise_start = 0.5
noise_decay = 0.995
noise_min = 0.05

print('🚀 Starting DDPG training (200 episodes)...')
print('   This will take ~10-15 minutes')
print()

# Training metrics
episode_rewards = []
episode_sharpes = []
episode_returns = []
actor_losses = []
critic_losses = []

best_sharpe = -np.inf
best_model = None

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    noise = max(noise_min, noise_start * (noise_decay ** episode))
    
    while not done:
        # Select action
        action = agent.select_action(state, noise=noise)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        # Train agent
        critic_loss, actor_loss = agent.train()
        
        episode_reward += reward
        state = next_state
    
    # Calculate episode metrics
    final_equity = env.equity
    total_return = (final_equity - env.initial_capital) / env.initial_capital
    
    # Calculate Sharpe (ROBUST - NO NaN!)
    if len(env.equity_curve) > 21:
        equity_array = np.array(env.equity_curve)
        returns = np.diff(equity_array) / env.initial_capital
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        sharpe = np.nan_to_num(sharpe, nan=0.0)
    else:
        sharpe = 0.0
    
    episode_rewards.append(episode_reward)
    episode_sharpes.append(sharpe)
    episode_returns.append(total_return)
    
    # Track best model
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_model = {
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'episode': episode,
            'sharpe': sharpe
        }
    
    # Print progress
    if (episode + 1) % 20 == 0:
        avg_sharpe = np.mean(episode_sharpes[-20:])
        avg_return = np.mean(episode_returns[-20:]) * 100
        print(f'   Episode {episode+1:3d}: Sharpe={sharpe:+.3f} | Return={total_return*100:+.2f}% | '
              f'Avg20={avg_sharpe:+.3f} | Noise={noise:.3f}')

print()
print('='*90)
print('✅ TRAINING COMPLETE!')
print('='*90)
print()
print(f'📊 FINAL RESULTS:')
print(f'   Best Sharpe Ratio:  {best_sharpe:+.3f} (Episode {best_model["episode"]+1})')
print(f'   Final Sharpe:       {episode_sharpes[-1]:+.3f}')
print(f'   Average Sharpe:     {np.mean(episode_sharpes):+.3f}')
print(f'   Final Return:       {episode_returns[-1]*100:+.2f}%')
print()

# Save best model
torch.save(best_model, 'drl_best_model.pth')
print('💾 Best model saved to: drl_best_model.pth')
print()

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Sharpe over episodes
axes[0, 0].plot(episode_sharpes, alpha=0.6, label='Episode Sharpe')
axes[0, 0].plot(pd.Series(episode_sharpes).rolling(20).mean(), linewidth=2, label='20-Episode MA')
axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Sharpe Ratio')
axes[0, 0].set_title('DRL Learning Progress (Sharpe Ratio)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Returns over episodes
axes[0, 1].plot([r*100 for r in episode_returns], alpha=0.6)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Total Return (%)')
axes[0, 1].set_title('Episode Returns')
axes[0, 1].grid(alpha=0.3)

# Cumulative reward
axes[1, 0].plot(np.cumsum(episode_rewards))
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Cumulative Reward')
axes[1, 0].set_title('Cumulative Learning Reward')
axes[1, 0].grid(alpha=0.3)

# Distribution of Sharpe ratios
axes[1, 1].hist(episode_sharpes, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=best_sharpe, color='red', linestyle='--', linewidth=2, label=f'Best: {best_sharpe:.3f}')
axes[1, 1].axvline(x=np.mean(episode_sharpes), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_sharpes):.3f}')
axes[1, 1].set_xlabel('Sharpe Ratio')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Episode Sharpe Ratios')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('drl_training_results.png', dpi=150, bbox_inches='tight')
print('📊 Training charts saved to: drl_training_results.png')
print()

print('🚀 DRL agent is now trained and ready for Hybrid ML+DRL integration!')
print('='*90)
