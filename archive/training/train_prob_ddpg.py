"""
FX TRADING ENVIRONMENT FOR DRL

Gym-style environment for training DRL agents on FX pair trading
Based on EUR/CHF real data with regime-switching dynamics
"""

import numpy as np
import pandas as pd
import yfinance as yf
from deep_rl_trading import ProbDDPG, create_market_features


class FXTradingEnv:
    """
    FX Trading Environment with regime-switching dynamics
    
    State: Market features + regime probabilities + current position
    Action: Position size in [-1, +1]
    Reward: PnL + transaction costs + risk penalty
    """
    
    def __init__(
        self,
        pair='EUR',
        start_date='2020-01-01',
        end_date='2025-01-01',
        initial_capital=100000,
        transaction_cost=0.0001,  # 1 bp
        lookback_window=20
    ):
        self.pair = pair
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.lookback = lookback_window
        
        # Download data
        print(f"📥 Loading {pair} data...")
        ticker = f'{pair}USD=X'
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if isinstance(data['Close'], pd.DataFrame):
            self.price = data['Close'].iloc[:, 0]
        else:
            self.price = data['Close']
        
        # Create features
        print(f"🔧 Creating features...")
        feature_df = pd.DataFrame({
            f'{pair}_close': self.price
        })
        self.features = create_market_features(feature_df, pair)
        
        # Align price and features
        common_idx = self.price.index.intersection(self.features.index)
        self.price = self.price.loc[common_idx]
        self.features = self.features.loc[common_idx]
        
        self.num_steps = len(self.price) - self.lookback - 1
        
        print(f"   ✅ {len(self.price)} days loaded")
        print(f"   ✅ {self.num_steps} tradeable steps")
        
        # State tracking
        self.reset()
        
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        self.position = 0.0
        self.capital = self.initial_capital
        self.equity = self.initial_capital
        self.trades = []
        
        return self._get_state()
    
    def _get_observations(self):
        """
        Get observation window for GRU
        
        Returns: (lookback, num_features) array
        """
        start_idx = self.current_step
        end_idx = self.current_step + self.lookback
        
        obs = self.features.iloc[start_idx:end_idx].values
        return obs
    
    def _get_state(self):
        """
        Get current state
        
        NOTE: Agent will filter regime probs from observations
        This just returns raw observations + current position + market state
        """
        # Recent observations for regime filtering
        observations = self._get_observations()
        
        # Current market features (last row)
        current_features = observations[-1, :]
        
        # Additional state info
        current_price = self.price.iloc[self.current_step + self.lookback]
        price_norm = (current_price - self.price.iloc[:self.current_step + self.lookback].mean()) / \
                     (self.price.iloc[:self.current_step + self.lookback].std() + 1e-8)
        
        # Market state (5 features)
        market_state = np.array([
            price_norm,
            current_features[0],  # 1d return
            current_features[3],  # volatility
            current_features[5],  # rsi
            self.equity / self.initial_capital - 1  # pnl ratio
        ])
        
        return {
            'observations': observations,
            'current_position': self.position,
            'market_features': market_state
        }
    
    def step(self, action):
        """
        Execute one trading step
        
        Args:
            action: float in [-1, +1] - target position size
        
        Returns:
            next_state, reward, done, info
        """
        # Current and next price
        current_price = self.price.iloc[self.current_step + self.lookback]
        next_price = self.price.iloc[self.current_step + self.lookback + 1]
        
        # Price change
        price_return = (next_price - current_price) / current_price
        
        # Old position
        old_position = self.position
        
        # New position (action is target)
        new_position = np.clip(action, -1, 1)
        
        # Position change (for transaction costs)
        position_change = abs(new_position - old_position)
        
        # PnL from holding position
        pnl = old_position * price_return * self.capital
        
        # Transaction costs
        transaction_costs = position_change * self.transaction_cost * abs(self.capital)
        
        # Update capital
        self.capital += pnl - transaction_costs
        self.equity = self.capital * (1 + new_position * price_return)
        
        # Reward = PnL - costs - risk penalty
        # Risk penalty: penalize large positions in high volatility
        volatility = self.features.iloc[self.current_step + self.lookback]['volatility']
        risk_penalty = 0.001 * abs(new_position) * volatility * self.capital
        
        reward = pnl - transaction_costs - risk_penalty
        
        # Normalize reward
        reward = reward / self.initial_capital
        
        # Update position
        self.position = new_position
        
        # Record trade
        self.trades.append({
            'step': self.current_step,
            'price': current_price,
            'action': action,
            'position': new_position,
            'pnl': pnl,
            'reward': reward,
            'equity': self.equity
        })
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= self.num_steps - 1
        
        # Info
        info = {
            'pnl': pnl,
            'transaction_costs': transaction_costs,
            'equity': self.equity,
            'sharpe': self._calculate_sharpe() if len(self.trades) > 30 else 0
        }
        
        # Next state
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done, info
    
    def _calculate_sharpe(self):
        """Calculate Sharpe ratio from recent trades"""
        if len(self.trades) < 30:
            return 0
        
        recent_returns = [t['reward'] for t in self.trades[-30:]]
        return np.mean(recent_returns) / (np.std(recent_returns) + 1e-8) * np.sqrt(252)
    
    def get_performance(self):
        """Get final performance metrics"""
        if len(self.trades) == 0:
            return {}
        
        trade_df = pd.DataFrame(self.trades)
        
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        rewards = trade_df['reward'].values
        sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252)
        
        max_equity = trade_df['equity'].cummax()
        drawdown = (trade_df['equity'] - max_equity) / max_equity
        max_dd = drawdown.min()
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(self.trades),
            'final_equity': self.equity
        }


def train_prob_ddpg(
    env,
    agent,
    num_episodes=100,
    noise_start=0.5,
    noise_decay=0.995,
    noise_min=0.05,
    batch_size=64,
    save_path='prob_ddpg_model.pth'
):
    """
    Train prob-DDPG agent on FX environment
    
    Args:
        env: FXTradingEnv
        agent: ProbDDPG agent
        num_episodes: Number of training episodes
        noise_start: Initial exploration noise
        noise_decay: Noise decay rate
        noise_min: Minimum noise
        batch_size: Training batch size
        save_path: Path to save best model
    
    Returns:
        training_history: Dict with metrics
    """
    print("="*70)
    print("🚀 TRAINING prob-DDPG AGENT")
    print("="*70)
    print()
    
    history = {
        'episode_rewards': [],
        'sharpe_ratios': [],
        'total_returns': []
    }
    
    best_sharpe = -np.inf
    noise = noise_start
    
    for episode in range(num_episodes):
        state_dict = env.reset()
        agent.gru_hidden = None  # Reset GRU hidden state
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Filter regime from observations
            regime_probs = agent.filter_regime(state_dict['observations'])
            
            # Construct state for actor
            state = agent.get_state(
                regime_probs,
                state_dict['current_position'],
                state_dict['market_features']
            )
            
            # Select action (with exploration noise)
            action = agent.select_action(state, noise=noise)
            
            # Environment step
            next_state_dict, reward, done, info = env.step(action)
            
            # Store transition
            if next_state_dict is not None:
                next_regime_probs = agent.filter_regime(next_state_dict['observations'])
                next_state = agent.get_state(
                    next_regime_probs,
                    next_state_dict['current_position'],
                    next_state_dict['market_features']
                )
                
                agent.memory.push(state, [action], reward, next_state, done)
            
            # Train
            if len(agent.memory) > batch_size:
                agent.train_step(batch_size)
            
            episode_reward += reward
            state_dict = next_state_dict
            step += 1
        
        # Episode complete
        perf = env.get_performance()
        history['episode_rewards'].append(episode_reward)
        history['sharpe_ratios'].append(perf.get('sharpe', 0))
        history['total_returns'].append(perf.get('total_return', 0))
        
        # Decay noise
        noise = max(noise_min, noise * noise_decay)
        
        # Save best model
        if perf.get('sharpe', -np.inf) > best_sharpe:
            best_sharpe = perf.get('sharpe', -np.inf)
            agent.save(save_path)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(history['episode_rewards'][-10:])
            avg_sharpe = np.mean(history['sharpe_ratios'][-10:])
            avg_return = np.mean(history['total_returns'][-10:])
            
            print(f"Episode {episode+1:3d}/{num_episodes} | "
                  f"Reward: {avg_reward:+.4f} | "
                  f"Sharpe: {avg_sharpe:+.3f} | "
                  f"Return: {avg_return*100:+.1f}% | "
                  f"Noise: {noise:.3f}")
    
    print()
    print("="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"   Best Sharpe: {best_sharpe:+.3f}")
    print(f"   Model saved: {save_path}")
    
    return history


if __name__ == '__main__':
    print("="*70)
    print("🎯 DEMO: Training prob-DDPG on EUR/USD")
    print("="*70)
    print()
    
    # Create environment
    env = FXTradingEnv(
        pair='EUR',
        start_date='2020-01-01',
        end_date='2024-01-01',
        initial_capital=100000
    )
    print()
    
    # Create agent
    agent = ProbDDPG(
        input_dim=10,
        num_regimes=3,
        hidden_dim=64,
        device='cpu'
    )
    print("✅ Agent initialized")
    print()
    
    # Train (small demo - 20 episodes)
    print("🚀 Starting training (20 episodes demo)...")
    print("   (For real training, use 100-500 episodes)")
    print()
    
    history = train_prob_ddpg(
        env,
        agent,
        num_episodes=20,
        save_path='prob_ddpg_eur.pth'
    )
    
    print()
    print("📊 Final Performance:")
    final_perf = env.get_performance()
    for key, value in final_perf.items():
        if 'return' in key or 'drawdown' in key:
            print(f"   {key:20s}: {value*100:+.1f}%")
        else:
            print(f"   {key:20s}: {value:+.3f}")
    
    print()
    print("🎊 Demo complete! For full training, increase num_episodes to 100-500")
