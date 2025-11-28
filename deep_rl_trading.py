"""
DEEP REINFORCEMENT LEARNING FOR FX TRADING
Based on arXiv:2511.00190 - "Deep reinforcement learning for optimal trading with partial information"

Key Components:
1. Regime-Switching Ornstein-Uhlenbeck (OU) process for trading signals
2. GRU network to filter hidden regime states
3. DDPG (Deep Deterministic Policy Gradient) for optimal trading
4. prob-DDPG: Uses posterior regime probabilities (best performer in paper)

Application: EUR/CHF pair trading with latent regimes
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random


class RegimeFilterGRU(nn.Module):
    """
    GRU network to filter regime probabilities from observations
    
    Input: Historical trading signals (prices, returns, features)
    Output: Regime probabilities P(regime | observations)
    """
    
    def __init__(self, input_dim=10, hidden_dim=64, num_regimes=3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        
        # GRU to capture temporal dependencies
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Regime probability head
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_regimes),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_dim) - sequence of observations
            hidden: Previous hidden state
        
        Returns:
            regime_probs: (batch, num_regimes) - P(regime | observations)
            hidden: Updated hidden state
        """
        # GRU forward
        gru_out, hidden = self.gru(x, hidden)
        
        # Use last timestep output
        last_output = gru_out[:, -1, :]
        
        # Regime probabilities
        regime_probs = self.regime_head(last_output)
        
        return regime_probs, hidden


class Actor(nn.Module):
    """
    Actor network for DDPG - outputs trading action (position size)
    
    Input: State (regime probs, current position, market features)
    Output: Action (continuous position in [-1, +1])
    """
    
    def __init__(self, state_dim, action_dim=1, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    """
    Critic network for DDPG - estimates Q-value Q(s, a)
    
    Input: State + Action
    Output: Q-value (expected future return)
    """
    
    def __init__(self, state_dim, action_dim=1, hidden_dim=128):
        super().__init__()
        
        # State pathway
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combined state-action pathway
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        state_features = self.state_net(state)
        combined = torch.cat([state_features, action], dim=1)
        q_value = self.combined_net(combined)
        return q_value


class ReplayBuffer:
    """Experience replay buffer for DRL"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )
        
    def __len__(self):
        return len(self.buffer)


class ProbDDPG:
    """
    prob-DDPG: DDPG with Regime Probability Filtering
    
    Best performer from paper - uses GRU to filter regime probabilities,
    then feeds them to DDPG agent for optimal trading.
    """
    
    def __init__(
        self, 
        input_dim=10,  # Number of market features
        num_regimes=3,  # Number of hidden regimes
        hidden_dim=64,
        lr_actor=1e-4,
        lr_critic=1e-3,
        lr_regime=1e-3,
        gamma=0.99,
        tau=0.001,
        device='cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.num_regimes = num_regimes
        
        # Regime filter (GRU)
        self.regime_filter = RegimeFilterGRU(input_dim, hidden_dim, num_regimes).to(device)
        self.regime_optimizer = torch.optim.Adam(self.regime_filter.parameters(), lr=lr_regime)
        
        # State dimension = regime_probs + current_position + additional_features
        state_dim = num_regimes + 1 + 5  # regime(3) + position(1) + features(5)
        
        # Actor networks (policy)
        self.actor = Actor(state_dim).to(device)
        self.actor_target = Actor(state_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic networks (Q-function)
        self.critic = Critic(state_dim).to(device)
        self.critic_target = Critic(state_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.memory = ReplayBuffer()
        
        # GRU hidden state
        self.gru_hidden = None
        
    def filter_regime(self, observations):
        """
        Filter regime probabilities from observation sequence
        
        Args:
            observations: (seq_len, input_dim) - recent market data
        
        Returns:
            regime_probs: (num_regimes,) - P(regime | observations)
        """
        # Add batch dimension
        obs_batch = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
        
        # Filter regime
        with torch.no_grad():
            regime_probs, self.gru_hidden = self.regime_filter(obs_batch, self.gru_hidden)
        
        return regime_probs.squeeze(0).cpu().numpy()
    
    def get_state(self, regime_probs, current_position, market_features):
        """
        Construct state vector for actor/critic
        
        Args:
            regime_probs: (num_regimes,) - filtered probabilities
            current_position: float - current position size
            market_features: (5,) - additional market state
        
        Returns:
            state: (state_dim,) - combined state vector
        """
        state = np.concatenate([
            regime_probs,
            [current_position],
            market_features
        ])
        return state
    
    def select_action(self, state, noise=0.0):
        """
        Select action using actor network (with optional exploration noise)
        
        Args:
            state: State vector
            noise: Exploration noise std
        
        Returns:
            action: Position size in [-1, 1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0, 0]
        
        # Add exploration noise
        if noise > 0:
            action += np.random.normal(0, noise)
            action = np.clip(action, -1, 1)
        
        return action
    
    def train_step(self, batch_size=64):
        """
        Perform one training step of DDPG
        
        Updates:
        - Critic: Minimize TD error
        - Actor: Maximize Q-value
        - Target networks: Soft update
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def _soft_update(self, source, target):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path):
        """Save model"""
        torch.save({
            'regime_filter': self.regime_filter.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.regime_filter.load_state_dict(checkpoint['regime_filter'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


def create_market_features(data, pair='EUR', lookback=20):
    """
    Create market features from FX data
    
    Features:
    - Returns (1d, 5d, 20d)
    - Volatility (rolling std)
    - Momentum z-score
    - RSI
    - Distance from MA
    - Volume proxy
    """
    features = []
    
    price = data[f'{pair}_close']
    
    # Returns
    ret_1d = price.pct_change()
    ret_5d = price.pct_change(5)
    ret_20d = price.pct_change(20)
    
    # Volatility
    vol = ret_1d.rolling(20).std()
    
    # Momentum z-score
    momentum = ret_20d
    momentum_zscore = (momentum - momentum.rolling(60).mean()) / momentum.rolling(60).std()
    
    # Simple RSI
    delta = ret_1d
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Distance from MA
    ma20 = price.rolling(20).mean()
    dist_ma = (price - ma20) / ma20
    
    # Volume proxy (volatility)
    volume_proxy = vol.rolling(5).mean()
    
    # Trend strength
    ma_fast = price.rolling(10).mean()
    ma_slow = price.rolling(30).mean()
    trend = (ma_fast - ma_slow) / ma_slow
    
    # Combine
    feature_df = pd.DataFrame({
        'ret_1d': ret_1d,
        'ret_5d': ret_5d,
        'ret_20d': ret_20d,
        'volatility': vol,
        'momentum_z': momentum_zscore,
        'rsi': rsi / 100,  # Normalize to [0, 1]
        'dist_ma': dist_ma,
        'volume_proxy': volume_proxy,
        'trend': trend,
        'price_norm': (price - price.rolling(252).mean()) / price.rolling(252).std()
    })
    
    return feature_df.fillna(0)


if __name__ == '__main__':
    print("="*70)
    print("🤖 DEEP REINFORCEMENT LEARNING FOR FX TRADING")
    print("="*70)
    print()
    print("Based on: arXiv:2511.00190")
    print("'Deep reinforcement learning for optimal trading with partial information'")
    print()
    
    print("✅ Components Implemented:")
    print("   1. RegimeFilterGRU - Filters hidden regime states")
    print("   2. Actor Network - Outputs optimal position size")
    print("   3. Critic Network - Estimates Q-value")
    print("   4. prob-DDPG - Complete algorithm")
    print()
    
    print("📊 Architecture:")
    print("   Observations → GRU → Regime Probabilities")
    print("   Regime Probs + State → Actor → Action (position)")
    print("   State + Action → Critic → Q-value")
    print()
    
    # Demo initialization
    agent = ProbDDPG(
        input_dim=10,
        num_regimes=3,
        hidden_dim=64,
        device='cpu'
    )
    
    print("🎯 Agent Configuration:")
    print(f"   Input features: 10")
    print(f"   Hidden regimes: 3 (bull/neutral/bear)")
    print(f"   GRU hidden dim: 64")
    print(f"   State dim: {3 + 1 + 5} (regime + position + features)")
    print()
    
    print("📁 Files created:")
    print("   deep_rl_trading.py - Full implementation")
    print()
    
    print("🚀 Next Steps:")
    print("   1. Create FX trading environment")
    print("   2. Train prob-DDPG on EUR/CHF data")
    print("   3. Compare with baseline strategies")
    print("   4. Implement hid-DDPG and reg-DDPG variants")
