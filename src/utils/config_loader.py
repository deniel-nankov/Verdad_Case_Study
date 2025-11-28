"""
Configuration Loader for FX Trading System
===========================================

Safely loads configuration from:
1. Environment variables (for secrets)
2. JSON config files (for non-sensitive settings)

Author: Deniel Nankov
Date: November 27, 2025
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate trading system configuration"""
    
    def __init__(self, config_dir: str = 'config'):
        """Initialize config loader
        
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self.config = {}
        
    def load(self) -> Dict[str, Any]:
        """Load complete configuration
        
        Returns:
            Dict with all configuration settings
        """
        # Load from JSON files
        self._load_json_configs()
        
        # Override with environment variables (for secrets)
        self._load_env_vars()
        
        # Validate required settings
        self._validate_config()
        
        return self.config
    
    def _load_json_configs(self):
        """Load configuration from JSON files"""
        config_files = [
            'system_config.json',
            'strategy_config.json',
            'risk_config.json'
        ]
        
        for filename in config_files:
            filepath = self.config_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        self.config.update(data)
                    logger.info(f"Loaded config from {filename}")
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
            else:
                logger.debug(f"Config file not found: {filename}")
        
        # Fallback: try loading old trading_config.json
        old_config = Path('trading_config.json')
        if old_config.exists() and not self.config:
            try:
                with open(old_config, 'r') as f:
                    self.config = json.load(f)
                logger.info("Loaded config from trading_config.json (legacy)")
            except Exception as e:
                logger.warning(f"Could not load trading_config.json: {e}")
    
    def _load_env_vars(self):
        """Load sensitive settings from environment variables"""
        env_mappings = {
            'ALPHAVANTAGE_API_KEY': ['api_keys', 'alpha_vantage'],
            'FRED_API_KEY': ['api_keys', 'fred'],
            'OANDA_API_KEY': ['api_keys', 'broker_api_key'],
            'OANDA_ACCOUNT_ID': ['api_keys', 'broker_account_id'],
            'BROKER_TYPE': ['system', 'broker_type'],
            'TRADING_MODE': ['system', 'mode'],
            'INITIAL_CAPITAL': ['system', 'initial_capital'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                # Navigate nested dict
                current = self.config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set value (convert to appropriate type)
                final_key = config_path[-1]
                if final_key in ['initial_capital']:
                    current[final_key] = float(value)
                else:
                    current[final_key] = value
                
                logger.debug(f"Loaded {env_var} from environment")
    
    def _validate_config(self):
        """Validate required configuration settings"""
        required_paths = [
            ['system', 'mode'],
            ['system', 'initial_capital'],
        ]
        
        for path in required_paths:
            current = self.config
            for key in path:
                if key not in current:
                    logger.warning(f"Missing required config: {'.'.join(path)}")
                    break
                current = current[key]
    
    def get(self, *keys, default=None) -> Any:
        """Get configuration value
        
        Args:
            *keys: Nested keys to navigate (e.g., 'system', 'mode')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config = ConfigLoader().load()
            >>> mode = config.get('system', 'mode', default='paper')
        """
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current


def load_config(config_dir: str = 'config') -> Dict[str, Any]:
    """Convenience function to load configuration
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        Complete configuration dict
        
    Example:
        >>> config = load_config()
        >>> mode = config.get('system', {}).get('mode', 'paper')
    """
    loader = ConfigLoader(config_dir)
    return loader.load()


def get_api_key(service: str) -> Optional[str]:
    """Get API key for a service
    
    Args:
        service: Service name ('alphavantage', 'fred', 'oanda')
        
    Returns:
        API key or None if not found
    """
    env_vars = {
        'alphavantage': 'ALPHAVANTAGE_API_KEY',
        'fred': 'FRED_API_KEY',
        'oanda': 'OANDA_API_KEY',
    }
    
    env_var = env_vars.get(service.lower())
    if env_var:
        key = os.getenv(env_var)
        if key:
            return key
    
    # Fallback to config file (not recommended for production)
    try:
        config = load_config()
        api_keys = config.get('api_keys', {})
        
        key_mappings = {
            'alphavantage': 'alpha_vantage',
            'fred': 'fred',
            'oanda': 'broker_api_key',
        }
        
        key_name = key_mappings.get(service.lower())
        if key_name and key_name in api_keys:
            logger.warning(f"Using API key from config file for {service} - consider using environment variables")
            return api_keys[key_name]
    except Exception as e:
        logger.error(f"Error loading API key for {service}: {e}")
    
    return None


if __name__ == '__main__':
    # Test configuration loading
    logging.basicConfig(level=logging.INFO)
    
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Trading mode: {config.get('system', {}).get('mode', 'unknown')}")
    print(f"Initial capital: ${config.get('system', {}).get('initial_capital', 0):,.0f}")
    
    # Test API key loading
    for service in ['alphavantage', 'fred', 'oanda']:
        key = get_api_key(service)
        if key:
            print(f"{service.upper()} API key: {'*' * 10}{key[-4:]}")
        else:
            print(f"{service.upper()} API key: Not found")
