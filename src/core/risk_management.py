"""
Risk Management System for FX Trading
======================================

Provides real-time risk monitoring and enforcement:
- Position size limits
- Total exposure limits
- Drawdown monitoring
- Stop-loss triggers
- Daily loss limits

Author: Deniel Nankov
Date: November 27, 2025
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk management parameters"""
    max_position_size: float = 0.5  # Max position as % of capital
    max_total_exposure: float = 2.0  # Max gross exposure
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    stop_loss_pct: float = 0.05  # 5% stop loss per position
    daily_loss_limit: float = 0.03  # 3% daily loss limit
    position_timeout_days: int = 30  # Max holding period


@dataclass
class Position:
    """Represents a currency position"""
    currency: str
    size: float  # Position size (positive = long, negative = short)
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_price(self, current_price: float):
        """Update current price and calculate P&L"""
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * self.size


class RiskManager:
    """Real-time risk management system"""
    
    def __init__(self, limits: RiskLimits, initial_capital: float):
        """Initialize risk manager
        
        Args:
            limits: Risk limit parameters
            initial_capital: Starting capital
        """
        self.limits = limits
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_pnl = 0.0
        self.daily_start_capital = initial_capital
        logger.info(f"Risk manager initialized with ${initial_capital:,.0f} capital")
    
    def check_position_limit(self, currency: str, size: float, current_price: float) -> bool:
        """Check if position size is within limits
        
        Args:
            currency: Currency code
            size: Position size
            current_price: Current FX rate
            
        Returns:
            True if within limits, False otherwise
        """
        position_value = abs(size * current_price)
        max_position_value = self.initial_capital * self.limits.max_position_size
        
        if position_value > max_position_value:
            logger.warning(
                f"Position size ${position_value:,.0f} exceeds limit "
                f"${max_position_value:,.0f} for {currency}"
            )
            return False
        return True
    
    def check_total_exposure(self, positions: List[Position]) -> bool:
        """Check total exposure across all positions
        
        Args:
            positions: List of current positions
            
        Returns:
            True if within limits, False otherwise
        """
        total_exposure = sum(abs(pos.size * pos.current_price) for pos in positions)
        max_exposure = self.initial_capital * self.limits.max_total_exposure
        
        if total_exposure > max_exposure:
            logger.warning(
                f"Total exposure ${total_exposure:,.0f} exceeds limit "
                f"${max_exposure:,.0f}"
            )
            return False
        return True
    
    def check_drawdown(self, current_capital: float) -> bool:
        """Check if drawdown limit is breached
        
        Args:
            current_capital: Current account value
            
        Returns:
            True if within limits, False if breached
        """
        self.peak_capital = max(self.peak_capital, current_capital)
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        
        if drawdown > self.limits.max_drawdown_pct:
            logger.error(f"DRAWDOWN LIMIT BREACHED: {drawdown*100:.2f}%")
            return False
        
        if drawdown > self.limits.max_drawdown_pct * 0.8:
            logger.warning(f"Approaching drawdown limit: {drawdown*100:.2f}%")
        
        return True
    
    def check_stop_loss(self, position: Position) -> bool:
        """Check if stop loss is triggered for a position
        
        Args:
            position: Position to check
            
        Returns:
            True if stop loss triggered, False otherwise
        """
        if abs(position.size) < 0.001:
            return False
            
        pnl_pct = position.unrealized_pnl / (position.entry_price * abs(position.size))
        
        if pnl_pct < -self.limits.stop_loss_pct:
            logger.warning(
                f"Stop loss triggered for {position.currency}: {pnl_pct*100:.2f}%"
            )
            return True
        return False
    
    def check_daily_loss_limit(self, current_capital: float) -> bool:
        """Check daily loss limit
        
        Args:
            current_capital: Current account value
            
        Returns:
            True if within limits, False if breached
        """
        daily_loss = (self.daily_start_capital - current_capital) / self.daily_start_capital
        
        if daily_loss > self.limits.daily_loss_limit:
            logger.error(f"DAILY LOSS LIMIT BREACHED: {daily_loss*100:.2f}%")
            return False
        
        if daily_loss > self.limits.daily_loss_limit * 0.8:
            logger.warning(f"Approaching daily loss limit: {daily_loss*100:.2f}%")
        
        return True
    
    def check_position_timeout(self, position: Position) -> bool:
        """Check if position has exceeded maximum holding period
        
        Args:
            position: Position to check
            
        Returns:
            True if timeout exceeded, False otherwise
        """
        days_held = (datetime.now() - position.entry_time).days
        
        if days_held > self.limits.position_timeout_days:
            logger.warning(
                f"Position timeout for {position.currency}: {days_held} days"
            )
            return True
        return False
    
    def reset_daily_limits(self, current_capital: float):
        """Reset daily counters (call at start of each trading day)
        
        Args:
            current_capital: Current account value
        """
        self.daily_start_capital = current_capital
        self.daily_pnl = 0.0
        logger.info(f"Daily risk limits reset. Starting capital: ${current_capital:,.0f}")
    
    def get_risk_metrics(self, current_capital: float, positions: List[Position]) -> dict:
        """Get current risk metrics
        
        Args:
            current_capital: Current account value
            positions: List of current positions
            
        Returns:
            Dict with risk metrics
        """
        total_exposure = sum(abs(pos.size * pos.current_price) for pos in positions)
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        daily_loss = (self.daily_start_capital - current_capital) / self.daily_start_capital
        
        return {
            'current_capital': current_capital,
            'peak_capital': self.peak_capital,
            'drawdown_pct': drawdown,
            'daily_loss_pct': daily_loss,
            'total_exposure': total_exposure,
            'exposure_pct': total_exposure / self.initial_capital,
            'num_positions': len(positions),
            'within_limits': all([
                drawdown <= self.limits.max_drawdown_pct,
                daily_loss <= self.limits.daily_loss_limit,
                total_exposure <= self.initial_capital * self.limits.max_total_exposure
            ])
        }
