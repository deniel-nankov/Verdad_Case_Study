"""
FX Carry Live Trading System
=============================

Production-ready trading system with:
- Real-time data feeds
- Broker API integration
- Risk management
- Paper trading mode
- Performance monitoring

Author: Deniel Nankov
Date: November 5, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode: paper or live"""
    PAPER = "paper"
    LIVE = "live"


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


@dataclass
class Order:
    """Represents a trading order"""
    currency: str
    size: float
    order_type: str  # 'market' or 'limit'
    side: str  # 'buy' or 'sell'
    price: Optional[float] = None
    status: str = 'pending'  # 'pending', 'filled', 'cancelled'
    fill_price: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RiskLimits:
    """Risk management parameters"""
    max_position_size: float = 0.5  # Max position as % of capital
    max_total_exposure: float = 2.0  # Max gross exposure
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    stop_loss_pct: float = 0.05  # 5% stop loss per position
    daily_loss_limit: float = 0.03  # 3% daily loss limit
    position_timeout_days: int = 30  # Max holding period


class DataFeed(ABC):
    """Abstract base class for data feeds"""
    
    @abstractmethod
    def get_fx_rate(self, currency_pair: str) -> Dict:
        """Get current FX rate"""
        pass
    
    @abstractmethod
    def get_interest_rate(self, currency: str) -> float:
        """Get current interest rate"""
        pass


class CachedYahooDataFeed(DataFeed):
    """
    Yahoo Finance cached data feed - No API spam, real data
    Uses persistent cache updated from Yahoo Finance
    """
    
    def __init__(self, cache_file: str = 'fx_data_cache.json', 
                 auto_refresh_hours: int = 6):
        self.cache_file = cache_file
        self.auto_refresh_hours = auto_refresh_hours
        self.disk_cache = {}
        self.last_refresh = None
        
        # Load cache from disk
        self._load_disk_cache()
        logger.info(f"✅ Loaded {len(self.disk_cache)} cached FX rates from {cache_file}")
        
    def _load_disk_cache(self):
        """Load cached FX rates from disk"""
        try:
            import os
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.disk_cache = json.load(f)
                    
                # Check if cache needs refresh
                if self.disk_cache:
                    first_key = list(self.disk_cache.keys())[0]
                    timestamp_str = self.disk_cache[first_key].get('timestamp')
                    if timestamp_str:
                        from dateutil import parser
                        try:
                            self.last_refresh = parser.parse(timestamp_str)
                        except:
                            self.last_refresh = datetime.now()
        except Exception as e:
            logger.warning(f"Could not load disk cache: {e}")
            self.disk_cache = {}
    
    def _should_refresh(self) -> bool:
        """Check if cache should be refreshed"""
        if not self.last_refresh:
            return True
        hours_old = (datetime.now() - self.last_refresh).total_seconds() / 3600
        return hours_old > self.auto_refresh_hours
    
    def _refresh_cache(self):
        """Refresh cache from Yahoo Finance"""
        try:
            import subprocess
            logger.info("🔄 Refreshing FX cache from Yahoo Finance...")
            result = subprocess.run(['python', 'populate_fx_cache.py'], 
                                  capture_output=True, timeout=30)
            if result.returncode == 0:
                self._load_disk_cache()
                logger.info("✅ Cache refreshed successfully")
            else:
                logger.warning("⚠️  Cache refresh failed, using existing cache")
        except Exception as e:
            logger.warning(f"Could not refresh cache: {e}")
    
    def get_fx_rate(self, currency_pair: str) -> Dict:
        """Get FX rate from cache"""
        # Auto-refresh if needed
        if self._should_refresh():
            self._refresh_cache()
        
        cache_key = f"fx_{currency_pair}"
        
        if cache_key in self.disk_cache:
            data = self.disk_cache[cache_key].copy()
            return data
        
        # Fallback to default rates
        logger.warning(f"⚠️  No cached data for {currency_pair}, using default")
        default_rates = {
            'USDEUR': 0.92, 'USDGBP': 0.79, 'USDJPY': 149.50,
            'USDCAD': 1.38, 'USDAUD': 0.65, 'USDNZD': 0.60,
            'USDCHF': 0.87, 'USDSEK': 10.50, 'USDBRL': 0.20,
            'USDMXN': 0.058
        }
        
        if currency_pair in default_rates:
            return {
                'rate': default_rates[currency_pair],
                'bid': default_rates[currency_pair] * 0.9995,
                'ask': default_rates[currency_pair] * 1.0005,
                'timestamp': datetime.now().isoformat(),
                'source': 'DEFAULT_ESTIMATE'
            }
        
        return None
    
    def get_interest_rate(self, currency: str) -> float:
        """Get interest rate"""
        rate_map = {
            'USD': 5.50, 'EUR': 4.00, 'GBP': 5.25, 'JPY': 0.25,
            'AUD': 4.35, 'CAD': 5.00, 'CHF': 1.75, 'BRL': 11.75,
            'MXN': 11.25, 'SEK': 3.75, 'NZD': 5.50
        }
        return rate_map.get(currency, 0.0)
    
    def is_connected(self) -> bool:
        """Always return True if we have cache"""
        return len(self.disk_cache) > 0


class OANDADataFeed(DataFeed):
    def is_connected(self) -> bool:
        """Check connection status"""
        pass


class AlphaVantageDataFeed(DataFeed):
    """Alpha Vantage API data feed"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = {}
        self.cache_duration = 60  # Cache for 60 seconds
        logger.info("Alpha Vantage data feed initialized")
    
    def get_fx_rate(self, currency_pair: str) -> Dict:
        """Get real-time FX rate from Alpha Vantage"""
        cache_key = f"fx_{currency_pair}"
        
        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_duration:
                return cached_data
        
        try:
            # Example: USD to EUR
            from_currency = currency_pair[:3]
            to_currency = currency_pair[3:]
            
            params = {
                'function': 'CURRENCY_EXCHANGE_RATE',
                'from_currency': from_currency,
                'to_currency': to_currency,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'Realtime Currency Exchange Rate' in data:
                exchange_data = data['Realtime Currency Exchange Rate']
                result = {
                    'rate': float(exchange_data['5. Exchange Rate']),
                    'bid': float(exchange_data['8. Bid Price']),
                    'ask': float(exchange_data['9. Ask Price']),
                    'timestamp': exchange_data['6. Last Refreshed']
                }
                
                # Cache result
                self.cache[cache_key] = (datetime.now(), result)
                return result
            else:
                logger.error(f"Error fetching FX rate: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Error in get_fx_rate: {e}")
            return None
    
    def get_interest_rate(self, currency: str) -> float:
        """Get interest rate (would need treasury/central bank API)"""
        # Placeholder - would integrate with FRED, ECB, BOJ APIs
        # For now, return cached rates
        rate_map = {
            'USD': 5.50,
            'EUR': 4.00,
            'GBP': 5.25,
            'JPY': 0.25,
            'AUD': 4.35,
            'CAD': 5.00,
            'CHF': 1.75,
            'BRL': 11.75,
            'MXN': 11.25
        }
        return rate_map.get(currency, 0.0)
    
    def is_connected(self) -> bool:
        """Check API connection"""
        try:
            test_pair = 'USDEUR'
            result = self.get_fx_rate(test_pair)
            return result is not None
        except:
            return False


class OANDADataFeed(DataFeed):
    """OANDA API data feed - Real-time FX rates with intelligent caching"""
    
    def __init__(self, api_key: str, account_id: str, practice: bool = True, 
                 cache_file: str = 'fx_data_cache.json'):
        try:
            from oandapyV20 import API
            from oandapyV20.endpoints import pricing, accounts
            
            self.api_key = api_key
            self.account_id = account_id
            self.practice = practice
            self.cache_file = cache_file
            
            # Initialize OANDA API
            environment = 'practice' if practice else 'live'
            self.client = API(access_token=api_key, environment=environment)
            
            # Map our currency pairs to OANDA instrument format
            self.instrument_map = {
                'USDEUR': 'EUR_USD',
                'USDGBP': 'GBP_USD',
                'USDJPY': 'USD_JPY',
                'USDCAD': 'USD_CAD',
                'USDAUD': 'AUD_USD',
                'USDNZD': 'NZD_USD',
                'USDCHF': 'USD_CHF',
                'USDSEK': 'USD_SEK',
                'USDBRL': 'USD_BRL',
                'USDMXN': 'USD_MXN'
            }
            
            self.pricing = pricing
            self.memory_cache = {}  # In-memory cache (short-term)
            self.cache_duration = 10  # Memory cache: 10 seconds
            self.disk_cache = {}  # Persistent cache (loaded from disk)
            self.api_connected = False
            
            # Load persistent cache from disk
            self._load_disk_cache()
            
            logger.info(f"OANDA data feed initialized ({environment} mode)")
            logger.info(f"Loaded {len(self.disk_cache)} cached rates from disk")
            
        except ImportError:
            logger.error("oandapyV20 not installed. Run: pip install oandapyV20")
            raise
    
    def _load_disk_cache(self):
        """Load cached FX rates from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.disk_cache = json.load(f)
                logger.info(f"Loaded disk cache with {len(self.disk_cache)} entries")
        except Exception as e:
            logger.warning(f"Could not load disk cache: {e}")
            self.disk_cache = {}
    
    def _save_disk_cache(self):
        """Save cached FX rates to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.disk_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save disk cache: {e}")
    
    def get_fx_rate(self, currency_pair: str) -> Dict:
        """
        Get FX rate with intelligent fallback strategy:
        1. Try in-memory cache (10 seconds)
        2. Try OANDA API (real-time data)
        3. Fall back to disk cache (persistent)
        4. Fall back to default/last known rate
        """
        cache_key = f"fx_{currency_pair}"
        
        # Strategy 1: Check in-memory cache (fastest, recent data)
        if cache_key in self.memory_cache:
            cached_time, cached_data = self.memory_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_duration:
                logger.debug(f"Using memory cache for {currency_pair}")
                return cached_data
        
        # Strategy 2: Try OANDA API (real-time data - preferred)
        try:
            instrument = self.instrument_map.get(currency_pair)
            if not instrument:
                logger.error(f"Unknown currency pair: {currency_pair}")
                return self._get_fallback_rate(currency_pair)
            
            # Get pricing from OANDA
            params = {"instruments": instrument}
            r = self.pricing.PricingInfo(accountID=self.account_id, params=params)
            response = self.client.request(r)
            
            if 'prices' in response and len(response['prices']) > 0:
                price_data = response['prices'][0]
                
                # Extract bid and ask
                bid = float(price_data['bids'][0]['price']) if price_data.get('bids') else 0
                ask = float(price_data['asks'][0]['price']) if price_data.get('asks') else 0
                mid = (bid + ask) / 2
                
                result = {
                    'rate': mid,
                    'bid': bid,
                    'ask': ask,
                    'timestamp': price_data.get('time', datetime.now().isoformat()),
                    'source': 'OANDA_REALTIME'
                }
                
                # Update both caches
                self.memory_cache[cache_key] = (datetime.now(), result)
                self.disk_cache[cache_key] = result
                self._save_disk_cache()
                
                self.api_connected = True
                logger.info(f"✅ OANDA real-time rate for {currency_pair}: {mid:.4f}")
                return result
                
        except Exception as e:
            logger.warning(f"OANDA API unavailable: {e}")
            self.api_connected = False
        
        # Strategy 3: Fall back to disk cache (persistent, possibly stale)
        return self._get_fallback_rate(currency_pair)
    
    def _get_fallback_rate(self, currency_pair: str) -> Dict:
        """Get rate from disk cache or defaults"""
        cache_key = f"fx_{currency_pair}"
        
        # Try disk cache first
        if cache_key in self.disk_cache:
            cached_data = self.disk_cache[cache_key].copy()
            cached_data['source'] = 'DISK_CACHE'
            logger.info(f"📦 Using cached rate for {currency_pair}: {cached_data['rate']:.4f}")
            return cached_data
        
        # Last resort: use default/estimated rates
        default_rates = {
            'USDEUR': 0.92,
            'USDGBP': 0.79,
            'USDJPY': 149.50,
            'USDCAD': 1.38,
            'USDAUD': 0.65,
            'USDNZD': 0.60,
            'USDCHF': 0.87,
            'USDSEK': 10.50,
            'USDBRL': 0.20,
            'USDMXN': 0.058
        }
        
        if currency_pair in default_rates:
            result = {
                'rate': default_rates[currency_pair],
                'bid': default_rates[currency_pair] * 0.9995,
                'ask': default_rates[currency_pair] * 1.0005,
                'timestamp': datetime.now().isoformat(),
                'source': 'DEFAULT_ESTIMATE'
            }
            logger.warning(f"⚠️  Using default rate for {currency_pair}: {result['rate']:.4f}")
            return result
        
        logger.error(f"❌ No rate available for {currency_pair}")
        return None
    
    def get_interest_rate(self, currency: str) -> float:
        """Get interest rate (from central bank rates or OANDA swap rates)"""
        rate_map = {
            'USD': 5.50,
            'EUR': 4.00,
            'GBP': 5.25,
            'JPY': 0.25,
            'AUD': 4.35,
            'CAD': 5.00,
            'CHF': 1.75,
            'BRL': 11.75,
            'MXN': 11.25,
            'SEK': 3.75,
            'NZD': 5.50
        }
        return rate_map.get(currency, 0.0)
    
    def is_connected(self) -> bool:
        """Check if OANDA API is accessible"""
        try:
            from oandapyV20.endpoints import accounts
            r = accounts.AccountSummary(accountID=self.account_id)
            self.client.request(r)
            self.api_connected = True
            return True
        except Exception as e:
            logger.warning(f"OANDA API not connected: {e}")
            self.api_connected = False
            # Still return True if we have cached data
            return len(self.disk_cache) > 0


class BrokerAPI(ABC):
    """Abstract base class for broker integration"""
    
    @abstractmethod
    def place_order(self, order: Order) -> bool:
        """Place an order"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> float:
        """Get account balance"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass


class PaperTradingBroker(BrokerAPI):
    """Paper trading broker (simulation)"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trade_history: List[Dict] = []
        logger.info(f"Paper trading broker initialized with ${initial_capital:,.2f}")
    
    def place_order(self, order: Order) -> bool:
        """Simulate order execution"""
        try:
            # Simulate slippage (0.5 bps)
            slippage = 0.00005
            
            if order.order_type == 'market':
                # Execute immediately with slippage
                if order.side == 'buy':
                    fill_price = order.price * (1 + slippage)
                else:
                    fill_price = order.price * (1 - slippage)
                
                order.fill_price = fill_price
                order.status = 'filled'
                
                # Update positions
                if order.currency in self.positions:
                    # Modify existing position
                    pos = self.positions[order.currency]
                    new_size = pos.size + order.size
                    if abs(new_size) < 0.001:  # Close position
                        # Realize P&L
                        realized_pnl = (fill_price - pos.entry_price) * pos.size
                        self.cash += realized_pnl
                        del self.positions[order.currency]
                        logger.info(f"Closed {order.currency} position. P&L: ${realized_pnl:,.2f}")
                    else:
                        # Update position
                        pos.size = new_size
                        pos.entry_price = fill_price
                else:
                    # New position
                    self.positions[order.currency] = Position(
                        currency=order.currency,
                        size=order.size,
                        entry_price=fill_price,
                        entry_time=datetime.now()
                    )
                
                # Record trade
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'currency': order.currency,
                    'side': order.side,
                    'size': order.size,
                    'price': fill_price
                })
                
                logger.info(f"Order filled: {order.side.upper()} {abs(order.size):.2f} {order.currency} @ {fill_price:.5f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False
    
    def get_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self.positions.values())
    
    def get_account_balance(self) -> float:
        """Get total account value"""
        total = self.cash
        for pos in self.positions.values():
            total += pos.unrealized_pnl
        return total
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order (not implemented for paper trading)"""
        return True


class RiskManager:
    """Real-time risk management system"""
    
    def __init__(self, limits: RiskLimits, initial_capital: float):
        self.limits = limits
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_pnl = 0.0
        self.daily_start_capital = initial_capital
        logger.info("Risk manager initialized")
    
    def check_position_limit(self, currency: str, size: float, current_price: float) -> bool:
        """Check if position size is within limits"""
        position_value = abs(size * current_price)
        max_position_value = self.initial_capital * self.limits.max_position_size
        
        if position_value > max_position_value:
            logger.warning(f"Position size ${position_value:,.0f} exceeds limit ${max_position_value:,.0f}")
            return False
        return True
    
    def check_total_exposure(self, positions: List[Position]) -> bool:
        """Check total exposure across all positions"""
        total_exposure = sum(abs(pos.size * pos.current_price) for pos in positions)
        max_exposure = self.initial_capital * self.limits.max_total_exposure
        
        if total_exposure > max_exposure:
            logger.warning(f"Total exposure ${total_exposure:,.0f} exceeds limit ${max_exposure:,.0f}")
            return False
        return True
    
    def check_drawdown(self, current_capital: float) -> bool:
        """Check if drawdown limit is breached"""
        self.peak_capital = max(self.peak_capital, current_capital)
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        
        if drawdown > self.limits.max_drawdown_pct:
            logger.error(f"DRAWDOWN LIMIT BREACHED: {drawdown*100:.2f}%")
            return False
        return True
    
    def check_stop_loss(self, position: Position) -> bool:
        """Check if stop loss is triggered"""
        pnl_pct = position.unrealized_pnl / (position.entry_price * abs(position.size))
        
        if pnl_pct < -self.limits.stop_loss_pct:
            logger.warning(f"Stop loss triggered for {position.currency}: {pnl_pct*100:.2f}%")
            return True
        return False
    
    def check_daily_loss_limit(self, current_capital: float) -> bool:
        """Check daily loss limit"""
        daily_loss = (self.daily_start_capital - current_capital) / self.daily_start_capital
        
        if daily_loss > self.limits.daily_loss_limit:
            logger.error(f"DAILY LOSS LIMIT BREACHED: {daily_loss*100:.2f}%")
            return False
        return True
    
    def reset_daily_limits(self, current_capital: float):
        """Reset daily counters"""
        self.daily_start_capital = current_capital
        self.daily_pnl = 0.0
        logger.info("Daily risk limits reset")


class LiveTradingSystem:
    """Complete live trading system"""
    
    def __init__(
        self,
        data_feed: DataFeed,
        broker: BrokerAPI,
        risk_manager: RiskManager,
        mode: TradingMode = TradingMode.PAPER
    ):
        self.data_feed = data_feed
        self.broker = broker
        self.risk_manager = risk_manager
        self.mode = mode
        self.is_running = False
        self.currencies = ['AUD', 'BRL', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'MXN']
        self.performance_log = []
        
        logger.info(f"Live trading system initialized in {mode.value.upper()} mode")
    
    def calculate_target_weights(self) -> Dict[str, float]:
        """
        Calculate target position weights using optimized strategy
        In production, this would use the ML/optimization models
        """
        # Get current interest rates
        interest_rates = {curr: self.data_feed.get_interest_rate(curr) for curr in self.currencies}
        usd_rate = self.data_feed.get_interest_rate('USD')
        
        # Calculate interest rate differentials
        rate_diffs = {curr: rate - usd_rate for curr, rate in interest_rates.items()}
        
        # Rank currencies by rate differential
        sorted_currencies = sorted(rate_diffs.items(), key=lambda x: x[1], reverse=True)
        
        # Simple 3x3 strategy for demo (would use optimized weights in production)
        weights = {}
        long_currencies = [curr for curr, _ in sorted_currencies[:3]]
        short_currencies = [curr for curr, _ in sorted_currencies[-3:]]
        
        for curr in self.currencies:
            if curr in long_currencies:
                weights[curr] = 1/3  # Long position
            elif curr in short_currencies:
                weights[curr] = -1/3  # Short position
            else:
                weights[curr] = 0.0
        
        return weights
    
    def rebalance_portfolio(self):
        """Rebalance portfolio to target weights"""
        try:
            logger.info("Starting portfolio rebalance...")
            
            # Get target weights
            target_weights = self.calculate_target_weights()
            
            # Get current positions
            current_positions = {pos.currency: pos.size for pos in self.broker.get_positions()}
            
            # Get account balance
            capital = self.broker.get_account_balance()
            
            # Calculate required trades
            for currency in self.currencies:
                target_size = target_weights[currency] * capital
                current_size = current_positions.get(currency, 0.0)
                trade_size = target_size - current_size
                
                if abs(trade_size) > 0.01 * capital:  # Only trade if > 1% of capital
                    # Get current FX rate
                    fx_data = self.data_feed.get_fx_rate(f'USD{currency}')
                    
                    if fx_data:
                        current_price = fx_data['rate']
                        
                        # Check risk limits
                        if not self.risk_manager.check_position_limit(currency, trade_size, current_price):
                            logger.warning(f"Skipping {currency} - position limit exceeded")
                            continue
                        
                        # Create order
                        order_size = trade_size
                        
                        # Place market order directly
                        order_id = self.broker.place_market_order(currency, order_size)
                        if order_id:
                            logger.info(f"Rebalanced {currency}: {trade_size:+.2f} (Order: {order_id})")
                        else:
                            logger.error(f"Failed to place order for {currency}")
            
            logger.info("Portfolio rebalance complete")
            
        except Exception as e:
            logger.error(f"Error in rebalance_portfolio: {e}")
    
    def monitor_risk(self):
        """Monitor and enforce risk limits"""
        try:
            positions = self.broker.get_positions()
            capital = self.broker.get_account_balance()
            
            # Skip if no positions
            if not positions or len(positions) == 0:
                return
            
            # Convert dict positions to list of Position objects if needed
            position_list = []
            if isinstance(positions, dict):
                for currency, size in positions.items():
                    if isinstance(size, (int, float)) and abs(size) > 0:
                        # Create Position object for risk checking
                        fx_data = self.data_feed.get_fx_rate(f"USD{currency}")
                        if fx_data:
                            position_list.append(Position(
                                currency=currency,
                                size=size,
                                entry_price=fx_data['rate'],
                                entry_time=datetime.now(),
                                current_price=fx_data['rate']
                            ))
            else:
                position_list = positions
            
            # Check total exposure
            if not self.risk_manager.check_total_exposure(position_list):
                logger.error("Total exposure limit breached - liquidating positions")
                self.liquidate_all_positions()
                return
            
            # Check drawdown
            if not self.risk_manager.check_drawdown(capital):
                logger.error("Drawdown limit breached - stopping trading")
                self.stop()
                return
            
            # Check daily loss limit
            if not self.risk_manager.check_daily_loss_limit(capital):
                logger.error("Daily loss limit breached - stopping trading for today")
                self.stop()
                return
            
            # Check individual stop losses
            for position in position_list:
                if hasattr(position, 'currency') and self.risk_manager.check_stop_loss(position):
                    logger.warning(f"Stop loss triggered for {position.currency} - closing position")
                    self.close_position(position.currency)
            
        except Exception as e:
            logger.error(f"Error in monitor_risk: {e}")
    
    def close_position(self, currency: str):
        """Close a specific position"""
        positions = self.broker.get_positions()
        if currency in positions:
            position_size = positions[currency]
            if abs(position_size) > 0:
                # Close position with market order
                close_size = -position_size
                order_id = self.broker.place_market_order(currency, close_size)
                if order_id:
                    logger.info(f"Closed position {currency}: {close_size}")
                else:
                    logger.error(f"Failed to close position {currency}")
    
    def liquidate_all_positions(self):
        """Close all positions"""
        logger.warning("Liquidating all positions...")
        success = self.broker.close_all_positions()
        if success:
            logger.info("All positions liquidated")
        else:
            logger.error("Failed to liquidate all positions")
    
    def update_performance_metrics(self):
        """Update and log performance metrics"""
        capital = self.broker.get_account_balance()
        positions = self.broker.get_positions()
        
        total_pnl = capital - self.risk_manager.initial_capital
        pnl_pct = (total_pnl / self.risk_manager.initial_capital) * 100
        
        metrics = {
            'timestamp': datetime.now(),
            'capital': capital,
            'total_pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'num_positions': len(positions),
            'cash': getattr(self.broker, 'cash', 0)
        }
        
        self.performance_log.append(metrics)
        
        logger.info(f"Performance: Capital=${capital:,.2f} | P&L={pnl_pct:+.2f}% | Positions={len(positions)}")
        
        return metrics
    
    def run(self, rebalance_frequency_hours: int = 24):
        """Main trading loop"""
        self.is_running = True
        last_rebalance = datetime.now() - timedelta(hours=rebalance_frequency_hours)
        last_risk_check = datetime.now()
        
        logger.info(f"Trading system started - {self.mode.value.upper()} mode")
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Check data feed connection
                if not self.data_feed.is_connected():
                    logger.error("Data feed disconnected - waiting...")
                    time.sleep(60)
                    continue
                
                # Rebalance portfolio (daily or as configured)
                if (current_time - last_rebalance).total_seconds() > rebalance_frequency_hours * 3600:
                    self.rebalance_portfolio()
                    last_rebalance = current_time
                
                # Monitor risk (every 5 minutes)
                if (current_time - last_risk_check).total_seconds() > 300:
                    self.monitor_risk()
                    last_risk_check = current_time
                
                # Update performance metrics (every 15 minutes)
                if len(self.performance_log) == 0 or \
                   (current_time - self.performance_log[-1]['timestamp']).total_seconds() > 900:
                    self.update_performance_metrics()
                
                # Sleep for a short interval
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Trading system interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading system"""
        self.is_running = False
        logger.info("Trading system stopped")
        
        # Save performance log
        if self.performance_log:
            df = pd.DataFrame(self.performance_log)
            df.to_csv('performance_log.csv', index=False)
            logger.info("Performance log saved to performance_log.csv")


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    print("FX Carry Live Trading System")
    print("=" * 60)
    print()
    print("IMPORTANT: This is a demonstration framework.")
    print("For live trading, you need:")
    print("  1. Valid API keys (Alpha Vantage, broker)")
    print("  2. Broker account setup")
    print("  3. Thorough testing in paper mode")
    print("  4. Regulatory compliance")
    print()
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    oanda_api_key = os.getenv('BROKER_API_KEY')
    oanda_account_id = os.getenv('BROKER_ACCOUNT_ID')
    broker_type = os.getenv('BROKER_TYPE', 'paper')
    initial_capital = float(os.getenv('INITIAL_CAPITAL', '100000'))
    rebalance_hours = int(os.getenv('REBALANCE_FREQUENCY', '24'))
    
    # Check for OANDA credentials
    if not oanda_api_key or not oanda_account_id:
        print("\n❌ ERROR: OANDA credentials not found in .env file")
        print("   Required: BROKER_API_KEY and BROKER_ACCOUNT_ID")
        print("   Please check your .env file")
        exit(1)
    
    print(f"\n🚀 Starting in {broker_type.upper()} mode...")
    print(f"   Initial Capital: ${initial_capital:,.0f}")
    print(f"   Rebalance: Every {rebalance_hours} hours")
    print(f"   Data Feed: OANDA Real-Time (Practice Environment)")
    print()
    
    # Data feed - Use OANDA with working credentials
    try:
        data_feed = OANDADataFeed(
            api_key=oanda_api_key,
            account_id=oanda_account_id,
            practice=True
        )
        if data_feed.is_connected():
            print("✅ OANDA real-time data feed connected!")
            print(f"   Account: {oanda_account_id}")
        else:
            raise Exception("OANDA connection failed")
    except Exception as e:
        print(f"⚠️  OANDA unavailable: {e}")
        print("   Falling back to cached Yahoo Finance data...")
        data_feed = CachedYahooDataFeed(
            cache_file='fx_data_cache.json',
            auto_refresh_hours=6
        )
        if data_feed.is_connected():
            print("✅ Yahoo Finance cache connected")
        else:
            print("❌ No data feed available")
            exit(1)
    
    # Broker setup
    if broker_type.lower() == 'paper':
        from broker_integrations import PaperTradingBroker, BrokerConfig
        broker_config = BrokerConfig(
            broker_name='paper',
            api_key='',
            practice=True
        )
        broker = PaperTradingBroker(broker_config)
        # Override initial balance
        broker.balance = initial_capital
    else:
        print(f"⚠️  Broker type '{broker_type}' requires additional setup")
        print("   Defaulting to paper trading for safety")
        from broker_integrations import PaperTradingBroker, BrokerConfig
        broker_config = BrokerConfig(
            broker_name='paper',
            api_key='',
            practice=True
        )
        broker = PaperTradingBroker(broker_config)
        broker.balance = initial_capital
    
    # Risk manager
    risk_limits = RiskLimits(
        max_position_size=float(os.getenv('MAX_POSITION_PCT', '0.30')),
        max_total_exposure=float(os.getenv('MAX_EXPOSURE', '2.0')),
        max_drawdown_pct=float(os.getenv('MAX_DRAWDOWN_PCT', '0.15')),
        stop_loss_pct=float(os.getenv('STOP_LOSS_PCT', '0.05')),
        daily_loss_limit=float(os.getenv('DAILY_LOSS_LIMIT_PCT', '0.03'))
    )
    risk_manager = RiskManager(risk_limits, initial_capital=initial_capital)
    
    # Trading system
    trading_mode = TradingMode.PAPER if broker_type.lower() == 'paper' else TradingMode.LIVE
    trading_system = LiveTradingSystem(
        data_feed=data_feed,
        broker=broker,
        risk_manager=risk_manager,
        mode=trading_mode
    )
    
    # Store strategy type for logging
    strategy_type = os.getenv('STRATEGY_TYPE', 'optimized')
    
    print("✅ System initialized successfully!")
    print(f"\n📊 Strategy: {strategy_type}")
    print(f"🛡️  Risk Limits:")
    print(f"   - Max position: {risk_limits.max_position_size*100:.0f}%")
    print(f"   - Max drawdown: {risk_limits.max_drawdown_pct*100:.0f}%")
    print(f"   - Stop-loss: {risk_limits.stop_loss_pct*100:.0f}%")
    print()
    print("🚀 Starting trading loop...")
    print("   Press Ctrl+C to stop")
    print("   Monitor: python monitoring_dashboard.py (in another terminal)")
    print("   Logs: tail -f trading_system.log")
    print()
    
    # Start trading!
    try:
        trading_system.run(rebalance_frequency_hours=rebalance_hours)
    except KeyboardInterrupt:
        print("\n\n⚠️  Trading stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        print("\n" + "="*60)
        print("Trading session ended")
        print("="*60)
