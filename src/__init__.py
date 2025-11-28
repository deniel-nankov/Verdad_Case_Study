"""
FX Trading System Package
==========================

A comprehensive FX carry trading system with ML, DRL, and live trading.

Modules:
    - core: Core trading system components
    - ml: Machine learning models and strategies
    - factors: Factor implementations
    - monitoring: Monitoring and alerting
    - utils: Utility functions
"""

__version__ = "2.0.0"
__author__ = "Deniel Nankov"

# Make key classes easily importable
from src.core.data_feeds import create_data_feed, DataFeed
from src.core.risk_management import RiskManager, RiskLimits, Position
from src.utils.config_loader import load_config, get_api_key

__all__ = [
    'create_data_feed',
    'DataFeed',
    'RiskManager',
    'RiskLimits',
    'Position',
    'load_config',
    'get_api_key',
]
