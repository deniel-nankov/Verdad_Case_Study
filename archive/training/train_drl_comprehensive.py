#!/usr/bin/env python3
"""
Comprehensive DRL Training - Research Mode
Train prob-DDPG with multiple configurations and compare results
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
from deep_rl_trading import ProbDDPG, RegimeFilterGRU, ReplayBuffer
import warnings
warnings.filterwarnings('ignore')

# Setup
sns.set_style('whitegrid')
device = torch.device('cpu')

print('='*80)
print('🧠 COMPREHENSIVE DRL TRADING RESEARCH')
print('='*80)
print(f'Device: {device}')
print()

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print('📊 Loading FX data...')
print('-'*80)

def load_fx_data(pair='EURUSD=X', start='2020-01-01'):
    """Load FX data with features"""
    data = yf.download(pair, start=start, progress=False)
    
    # Price features
    data['returns'] = data['Close'].pct_change()
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Volatility
    data['vol_5d'] = data['returns'].rolling(5).std()
    data['vol_21d'] = data['returns'].rolling(21).std()
    
    # Momentum
    data['mom_5d'] = data['Close'].pct_change(5)
    data['mom_21d'] = data['Close'].pct_change(21)
    
    # Moving averages
    data['ma_5'] = data['Close'].rolling(5).mean()
    data['ma_21'] = data['Close'].rolling(21).mean()
    data['ma_cross'] = (data['ma_5'] - data['ma_21']) / data['ma_21']
    
    # Volume (proxy - using range)
    data['range'] = (data['High'] - data['Low']) / data['Close']
    
    return data.dropna()

# Load EUR/USD
eur_data = load_fx_data('EURUSD=X', start='2020-01-01')
print(f"✅ Loaded {len(eur_data)} days of EUR/USD data")
print(f"   Date range: {eur_data.index[0].date()} to {eur_data.index[-1].date()}")
print()

# ============================================================================
# 2. CREATE TRADING ENVIRONMENT
# ============================================================================
print('🏗️  Creating trading environment...')
print('-'*80)

class FXTradingEnv:
    """Simplified FX trading environment for research"""
    
    def __init__(self, data, initial_capital=100000, transaction_cost=0.0001):
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reset()
        
    def reset(self):
        self.position = 0.0
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.current_step = 21  # Start after warmup
        self.equity_curve = [self.initial_capital]
        return self._get_state()
    
    def _get_state(self):
        """Get current state (last 10 days of features)"""
        if self.current_step < 10:
            return np.zeros((10, 10))
        
        idx = self.current_step
        window = self.data.iloc[idx-10:idx]
        
        features = np.column_stack([
            window['returns'].values,
            window['log_returns'].values,
            window['vol_5d'].values,
            window['vol_21d'].values,
            window['mom_5d'].values,
            window['mom_21d'].values,
            window['ma_cross'].values,
            window['range'].values,
            np.full(10, self.position),
            np.full(10, self.equity / self.initial_capital - 1)
        ])
        
        return features
    
    def step(self, action):
        """Execute trading action"""
        # Get current price
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Calculate position change
        target_position = np.clip(action, -1, 1)
        position_change = target_position - self.position
        
        # Transaction costs
        cost = abs(position_change) * current_price * self.transaction_cost * self.initial_capital
        
        # Update position
        old_position = self.position
        self.position = target_position
        
        # Move to next step
        self.current_step += 1
        
        if self.current_step >= len(self.data):
            return self._get_state(), 0, True
        
        # Calculate returns
        next_price = self.data.iloc[self.current_step]['Close']
        price_change = (next_price - current_price) / current_price
        
        # PnL from position
        pnl = old_position * price_change * self.initial_capital
        
        # Update equity
        self.equity = self.equity + pnl - cost
        self.equity_curve.append(self.equity)
        
        # Reward = sharpe-like metric
        if len(self.equity_curve) > 21:
            returns = np.diff(self.equity_curve[-21:]) / self.initial_capital
            reward = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            reward = pnl / self.initial_capital
        
        done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, done

print("✅ Environment created")
print()

# ============================================================================
# 3. TRAIN MULTIPLE CONFIGURATIONS
# ============================================================================
print('🚀 Training DRL agents with different configurations...')
print('-'*80)
print()

configs = [
    {
        'name': 'Baseline',
        'hidden_size': 64,
        'num_layers': 2,
        'lr_actor': 0.0001,
        'lr_critic': 0.001,
        'episodes': 100,
        'batch_size': 64
    },
    {
        'name': 'Deep',
        'hidden_size': 128,
        'num_layers': 3,
        'lr_actor': 0.0001,
        'lr_critic': 0.001,
        'episodes': 100,
        'batch_size': 64
    },
    {
        'name': 'Conservative',
        'hidden_size': 32,
        'num_layers': 1,
        'lr_actor': 0.00005,
        'lr_critic': 0.0005,
        'episodes': 100,
        'batch_size': 32
    },
    {
        'name': 'Aggressive',
        'hidden_size': 64,
        'num_layers': 2,
        'lr_actor': 0.0003,
        'lr_critic': 0.003,
        'episodes': 100,
        'batch_size': 128
    }
]

all_results = {}

for config in configs:
    print(f"\n{'='*80}")
    print(f"Training: {config['name']}")
    print('='*80)
    
    # Create environment
    env = FXTradingEnv(eur_data)
    
    # Create agent
    agent = ProbDDPG(
        input_dim=10,  # Number of market features
        num_regimes=3,  # Number of hidden regimes
        hidden_dim=config['hidden_size'],
        lr_actor=config['lr_actor'],
        lr_critic=config['lr_critic']
    )
    
    # Training loop
    episode_rewards = []
    episode_sharpes = []
    episode_returns = []
    best_sharpe = -np.inf
    
    for episode in range(config['episodes']):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = agent.select_action(state_tensor)
            
            # Add exploration noise
            noise = np.random.normal(0, 0.1 * (0.995 ** episode))
            action = np.clip(action + noise, -1, 1)
            
            # Execute action
            next_state, reward, done = env.step(action[0])
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train
            if len(agent.replay_buffer) > config['batch_size']:
                agent.train(batch_size=config['batch_size'])
            
            state = next_state
            episode_reward += reward
        
        # Episode metrics
        equity_curve = np.array(env.equity_curve)
        returns = np.diff(equity_curve) / env.initial_capital
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        
        episode_rewards.append(episode_reward)
        episode_sharpes.append(sharpe)
        episode_returns.append(total_return)
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_episode = episode
        
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode+1:3d}: Sharpe={sharpe:+.3f}, Return={total_return:+.2f}%, Best={best_sharpe:+.3f}")
    
    # Store results
    all_results[config['name']] = {
        'config': config,
        'episode_rewards': episode_rewards,
        'episode_sharpes': episode_sharpes,
        'episode_returns': episode_returns,
        'best_sharpe': best_sharpe,
        'best_episode': best_episode,
        'final_sharpe': episode_sharpes[-1],
        'final_return': episode_returns[-1]
    }
    
    print(f"\n✅ {config['name']} Complete:")
    print(f"   Best Sharpe: {best_sharpe:+.3f} (Episode {best_episode})")
    print(f"   Final Sharpe: {episode_sharpes[-1]:+.3f}")
    print(f"   Final Return: {episode_returns[-1]:+.2f}%")

print()
print('='*80)
print('📊 TRAINING COMPLETE - COMPARING RESULTS')
print('='*80)
print()

# ============================================================================
# 4. COMPARE RESULTS
# ============================================================================

# Create comparison dataframe
comparison = pd.DataFrame([
    {
        'Configuration': name,
        'Best_Sharpe': res['best_sharpe'],
        'Final_Sharpe': res['final_sharpe'],
        'Final_Return': res['final_return'],
        'Hidden_Size': res['config']['hidden_size'],
        'Num_Layers': res['config']['num_layers'],
        'Learning_Rate': res['config']['lr_actor']
    }
    for name, res in all_results.items()
])

print("Performance Comparison:")
print('-'*80)
print(comparison.to_string(index=False))
print()

# Find best configuration
best_config = comparison.loc[comparison['Best_Sharpe'].idxmax(), 'Configuration']
print(f"🏆 Best Configuration: {best_config}")
print(f"   Best Sharpe: {comparison.loc[comparison['Best_Sharpe'].idxmax(), 'Best_Sharpe']:+.3f}")
print()

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print('📊 Creating visualizations...')
print('-'*80)

# Figure 1: Learning curves
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for idx, (name, res) in enumerate(all_results.items()):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    episodes = range(1, len(res['episode_sharpes']) + 1)
    
    # Plot sharpe ratio
    ax.plot(episodes, res['episode_sharpes'], label='Sharpe Ratio', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.3, label='Target')
    
    # Mark best episode
    best_ep = res['best_episode']
    best_sharpe = res['best_sharpe']
    ax.scatter([best_ep + 1], [best_sharpe], color='red', s=100, zorder=5, 
               label=f'Best: {best_sharpe:+.3f}')
    
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontweight='bold')
    ax.set_title(f'{name} - Learning Curve', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('drl_learning_curves.png', dpi=150, bbox_inches='tight')
print('✅ Saved: drl_learning_curves.png')
plt.close()

# Figure 2: Performance comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Best Sharpe
ax = axes[0]
sorted_comp = comparison.sort_values('Best_Sharpe', ascending=True)
colors = ['green' if x > 0.5 else 'orange' if x > 0 else 'red' for x in sorted_comp['Best_Sharpe']]
ax.barh(sorted_comp['Configuration'], sorted_comp['Best_Sharpe'], color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.3, label='Good threshold')
ax.set_xlabel('Best Sharpe Ratio', fontweight='bold')
ax.set_title('Best Sharpe by Configuration', fontweight='bold', fontsize=12)
ax.grid(axis='x', alpha=0.3)

# Final Sharpe
ax = axes[1]
sorted_comp = comparison.sort_values('Final_Sharpe', ascending=True)
colors = ['green' if x > 0.5 else 'orange' if x > 0 else 'red' for x in sorted_comp['Final_Sharpe']]
ax.barh(sorted_comp['Configuration'], sorted_comp['Final_Sharpe'], color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.3)
ax.set_xlabel('Final Sharpe Ratio', fontweight='bold')
ax.set_title('Final Sharpe by Configuration', fontweight='bold', fontsize=12)
ax.grid(axis='x', alpha=0.3)

# Final Return
ax = axes[2]
sorted_comp = comparison.sort_values('Final_Return', ascending=True)
colors = ['green' if x > 0 else 'red' for x in sorted_comp['Final_Return']]
ax.barh(sorted_comp['Configuration'], sorted_comp['Final_Return'], color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Final Return (%)', fontweight='bold')
ax.set_title('Final Return by Configuration', fontweight='bold', fontsize=12)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('drl_performance_comparison.png', dpi=150, bbox_inches='tight')
print('✅ Saved: drl_performance_comparison.png')
plt.close()

# Figure 3: All learning curves together
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

for name, res in all_results.items():
    episodes = range(1, len(res['episode_sharpes']) + 1)
    ax1.plot(episodes, res['episode_sharpes'], label=name, linewidth=2, alpha=0.7)
    ax2.plot(episodes, res['episode_returns'], label=name, linewidth=2, alpha=0.7)

ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.3, label='Target')
ax1.set_xlabel('Episode', fontweight='bold')
ax1.set_ylabel('Sharpe Ratio', fontweight='bold')
ax1.set_title('Sharpe Ratio - All Configurations', fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax2.set_xlabel('Episode', fontweight='bold')
ax2.set_ylabel('Return (%)', fontweight='bold')
ax2.set_title('Returns - All Configurations', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('drl_all_configurations.png', dpi=150, bbox_inches='tight')
print('✅ Saved: drl_all_configurations.png')
plt.close()

print()

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print('💾 Saving results...')
print('-'*80)

# Save comparison CSV
comparison.to_csv('drl_comparison_results.csv', index=False)
print('✅ Saved: drl_comparison_results.csv')

# Save detailed results
with open('drl_detailed_results.txt', 'w') as f:
    f.write('='*80 + '\n')
    f.write('COMPREHENSIVE DRL TRAINING RESULTS\n')
    f.write('='*80 + '\n\n')
    
    f.write(f'Training Date: {datetime.now()}\n')
    f.write(f'Data Period: {eur_data.index[0].date()} to {eur_data.index[-1].date()}\n')
    f.write(f'Total Days: {len(eur_data)}\n\n')
    
    f.write('='*80 + '\n')
    f.write('PERFORMANCE COMPARISON\n')
    f.write('='*80 + '\n\n')
    f.write(comparison.to_string(index=False))
    f.write('\n\n')
    
    f.write('='*80 + '\n')
    f.write('DETAILED RESULTS BY CONFIGURATION\n')
    f.write('='*80 + '\n\n')
    
    for name, res in all_results.items():
        f.write(f"\n{name}:\n")
        f.write('-'*80 + '\n')
        f.write(f"Configuration:\n")
        for key, val in res['config'].items():
            if key != 'name':
                f.write(f"  {key}: {val}\n")
        f.write(f"\nPerformance:\n")
        f.write(f"  Best Sharpe: {res['best_sharpe']:+.4f} (Episode {res['best_episode'] + 1})\n")
        f.write(f"  Final Sharpe: {res['final_sharpe']:+.4f}\n")
        f.write(f"  Final Return: {res['final_return']:+.2f}%\n")

print('✅ Saved: drl_detailed_results.txt')
print()

# ============================================================================
# 7. SUMMARY
# ============================================================================
print('='*80)
print('✅ COMPREHENSIVE DRL RESEARCH COMPLETE')
print('='*80)
print()
print(f'📊 Summary:')
print(f'   • Configurations tested: {len(configs)}')
print(f'   • Episodes per config: 100')
print(f'   • Best configuration: {best_config}')
print(f'   • Best Sharpe achieved: {comparison["Best_Sharpe"].max():+.3f}')
print(f'   • Training data: {len(eur_data)} days of EUR/USD')
print()
print(f'📁 Files created:')
print(f'   • drl_learning_curves.png - Learning curves for each config')
print(f'   • drl_performance_comparison.png - Performance comparison charts')
print(f'   • drl_all_configurations.png - All configs on same plot')
print(f'   • drl_comparison_results.csv - Performance metrics')
print(f'   • drl_detailed_results.txt - Full detailed results')
print()
print(f'🔬 Research Insights:')
print(f'   • Test different architectures to find best for FX trading')
print(f'   • Compare learning stability vs final performance')
print(f'   • Identify trade-offs between model complexity and results')
print()
print(f'🚀 Next Steps:')
print(f'   1. Analyze which configuration works best')
print(f'   2. Train winning config for more episodes (200-500)')
print(f'   3. Test on other currency pairs (GBP, JPY, CHF)')
print(f'   4. Implement ensemble of best configs')
print(f'   5. Combine DRL with ML factor models')
print()
print('='*80)
