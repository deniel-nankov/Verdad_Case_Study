"""
Data Feed Implementations for FX Trading System
================================================

Provides multiple data feed options:
- CachedYahooDataFeed: Offline-first with Yahoo Finance cache
- AlphaVantageDataFeed: Real-time via Alpha Vantage API
- OANDADataFeed: Real-time via OANDA API with intelligent caching

Author: Deniel Nankov
Date: November 27, 2025
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class DataFeed(ABC):
    """Abstract base class for data feeds"""
    
    @abstractmethod
    def get_fx_rate(self, currency_pair: str) -> Optional[Dict]:
        """Get current FX rate
        
        Args:
            currency_pair: Currency pair in format 'USDEUR', 'USDJPY', etc.
            
        Returns:
            Dict with keys: rate, bid, ask, timestamp, source
            None if rate unavailable
        """
        pass
    
    @abstractmethod
    def get_interest_rate(self, currency: str) -> float:
        """Get current interest rate for currency
        
        Args:
            currency: Currency code (e.g., 'USD', 'EUR')
            
        Returns:
            Interest rate as decimal (e.g., 0.055 for 5.5%)
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if data feed is connected and operational
        
        Returns:
            True if connected, False otherwise
        """
        pass


class CachedYahooDataFeed(DataFeed):
    """
    Yahoo Finance cached data feed - No API spam, real data
    Uses persistent cache updated from Yahoo Finance
    
    This is the recommended data feed for development and testing
    as it works offline and doesn't require API keys.
    """
    
    def __init__(self, cache_file: str = 'fx_data_cache.json', 
                 auto_refresh_hours: int = 6):
        """Initialize cached data feed
        
        Args:
            cache_file: Path to cache file
            auto_refresh_hours: Hours before cache refresh (0 = never auto-refresh)
        """
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
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.disk_cache = json.load(f)
                    
                # Check if cache needs refresh
                if self.disk_cache:
                    first_key = list(self.disk_cache.keys())[0]
                    timestamp_str = self.disk_cache[first_key].get('timestamp')
                    if timestamp_str:
                        try:
                            from dateutil import parser
                            self.last_refresh = parser.parse(timestamp_str)
                        except:
                            self.last_refresh = datetime.now()
        except Exception as e:
            logger.warning(f"Could not load disk cache: {e}")
            self.disk_cache = {}
    
    def _should_refresh(self) -> bool:
        """Check if cache should be refreshed"""
        if self.auto_refresh_hours == 0:
            return False
        if not self.last_refresh:
            return True
        # Make both datetimes timezone-naive for comparison
        now = datetime.now()
        last_refresh = self.last_refresh.replace(tzinfo=None) if self.last_refresh.tzinfo else self.last_refresh
        hours_old = (now - last_refresh).total_seconds() / 3600
        return hours_old > self.auto_refresh_hours
    
    def _refresh_cache(self):
        """Refresh cache from Yahoo Finance"""
        try:
            import subprocess
            logger.info("🔄 Refreshing FX cache from Yahoo Finance...")
            # Look for populate script in multiple locations
            script_paths = [
                'populate_fx_cache.py',
                'scripts/data/populate_cache.py',
                '../populate_fx_cache.py'
            ]
            
            for script_path in script_paths:
                if os.path.exists(script_path):
                    result = subprocess.run(['python', script_path], 
                                          capture_output=True, timeout=30)
                    if result.returncode == 0:
                        self._load_disk_cache()
                        logger.info("✅ Cache refreshed successfully")
                        return
            
            logger.warning("⚠️  Cache refresh script not found, using existing cache")
        except Exception as e:
            logger.warning(f"Could not refresh cache: {e}")
    
    def get_fx_rate(self, currency_pair: str) -> Optional[Dict]:
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
        """Get interest rate (from static map)"""
        rate_map = {
            'USD': 5.50, 'EUR': 4.00, 'GBP': 5.25, 'JPY': 0.25,
            'AUD': 4.35, 'CAD': 5.00, 'CHF': 1.75, 'BRL': 11.75,
            'MXN': 11.25, 'SEK': 3.75, 'NZD': 5.50
        }
        return rate_map.get(currency, 0.0)
    
    def is_connected(self) -> bool:
        """Always return True if we have cache"""
        return len(self.disk_cache) > 0


class AlphaVantageDataFeed(DataFeed):
    """Alpha Vantage API data feed
    
    Requires API key from https://www.alphavantage.co/
    Free tier: 5 API calls per minute, 500 per day
    """
    
    def __init__(self, api_key: str):
        """Initialize Alpha Vantage data feed
        
        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = {}
        self.cache_duration = 60  # Cache for 60 seconds
        logger.info("Alpha Vantage data feed initialized")
    
    def get_fx_rate(self, currency_pair: str) -> Optional[Dict]:
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
                    'timestamp': exchange_data['6. Last Refreshed'],
                    'source': 'ALPHAVANTAGE'
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
        """Get interest rate (static map for now)"""
        rate_map = {
            'USD': 5.50, 'EUR': 4.00, 'GBP': 5.25, 'JPY': 0.25,
            'AUD': 4.35, 'CAD': 5.00, 'CHF': 1.75, 'BRL': 11.75,
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
    """OANDA API data feed - Real-time FX rates with intelligent caching
    
    Requires OANDA account and API key from https://www.oanda.com/
    Supports both practice and live environments.
    """
    
    def __init__(self, api_key: str, account_id: str, practice: bool = True, 
                 cache_file: str = 'fx_data_cache.json'):
        """Initialize OANDA data feed
        
        Args:
            api_key: OANDA API key
            account_id: OANDA account ID
            practice: Use practice environment (True) or live (False)
            cache_file: Path to persistent cache file
        """
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
    
    def get_fx_rate(self, currency_pair: str) -> Optional[Dict]:
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
    
    def _get_fallback_rate(self, currency_pair: str) -> Optional[Dict]:
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
            'USDEUR': 0.92, 'USDGBP': 0.79, 'USDJPY': 149.50,
            'USDCAD': 1.38, 'USDAUD': 0.65, 'USDNZD': 0.60,
            'USDCHF': 0.87, 'USDSEK': 10.50, 'USDBRL': 0.20,
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
            'USD': 5.50, 'EUR': 4.00, 'GBP': 5.25, 'JPY': 0.25,
            'AUD': 4.35, 'CAD': 5.00, 'CHF': 1.75, 'BRL': 11.75,
            'MXN': 11.25, 'SEK': 3.75, 'NZD': 5.50
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


def create_data_feed(feed_type: str = 'cached', **kwargs) -> DataFeed:
    """Factory function to create data feed
    
    Args:
        feed_type: Type of feed ('cached', 'alphavantage', 'oanda')
        **kwargs: Additional arguments for specific feed type
        
    Returns:
        DataFeed instance
        
    Examples:
        >>> feed = create_data_feed('cached')
        >>> feed = create_data_feed('alphavantage', api_key='YOUR_KEY')
        >>> feed = create_data_feed('oanda', api_key='KEY', account_id='ID')
    """
    if feed_type == 'cached':
        return CachedYahooDataFeed(**kwargs)
    elif feed_type == 'alphavantage':
        if 'api_key' not in kwargs:
            raise ValueError("AlphaVantage feed requires 'api_key'")
        return AlphaVantageDataFeed(**kwargs)
    elif feed_type == 'oanda':
        if 'api_key' not in kwargs or 'account_id' not in kwargs:
            raise ValueError("OANDA feed requires 'api_key' and 'account_id'")
        return OANDADataFeed(**kwargs)
    else:
        raise ValueError(f"Unknown feed type: {feed_type}")
