"""
Broker Integration Modules for FX Live Trading
Supports: OANDA, Interactive Brokers, Alpaca, and Paper Trading
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

# Import broker-specific libraries (install as needed)
try:
    import oandapyV20
    from oandapyV20 import API
    from oandapyV20.endpoints import accounts, orders, trades, pricing
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    logging.warning("oandapyV20 not installed. OANDA integration unavailable.")

try:
    from ib_insync import IB, Forex, MarketOrder, LimitOrder
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    # Create dummy classes for type hints
    class Forex:
        pass
    class MarketOrder:
        pass
    class LimitOrder:
        pass
    logging.warning("ib_insync not installed. Interactive Brokers integration unavailable.")

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("alpaca_trade_api not installed. Alpaca integration unavailable.")


@dataclass
class BrokerConfig:
    """Configuration for broker connection"""
    broker_name: str
    api_key: str
    api_secret: Optional[str] = None
    account_id: Optional[str] = None
    base_url: Optional[str] = None
    practice: bool = True  # Paper trading vs live
    

class BrokerInterface(ABC):
    """Abstract base class for all broker integrations"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker API"""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> float:
        """Get current account cash balance"""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Get current positions {currency_pair: quantity}"""
        pass
    
    @abstractmethod
    def place_market_order(self, currency_pair: str, quantity: float) -> str:
        """Place market order, returns order ID"""
        pass
    
    @abstractmethod
    def place_limit_order(self, currency_pair: str, quantity: float, limit_price: float) -> str:
        """Place limit order, returns order ID"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        """Get order status: pending/filled/cancelled"""
        pass
    
    @abstractmethod
    def close_all_positions(self) -> bool:
        """Emergency liquidation of all positions"""
        pass


# ============================================================================
# OANDA BROKER INTEGRATION
# ============================================================================

class OANDABroker(BrokerInterface):
    """OANDA FX broker integration (Recommended for FX)"""
    
    def __init__(self, config: BrokerConfig):
        if not OANDA_AVAILABLE:
            raise ImportError("oandapyV20 library not installed. Run: pip install oandapyV20")
        
        self.config = config
        self.client = API(
            access_token=config.api_key,
            environment='practice' if config.practice else 'live'
        )
        self.account_id = config.account_id
        self.logger = logging.getLogger(__name__)
        
        # Map our currency pairs to OANDA format
        self.currency_map = {
            'USDEUR': 'EUR_USD',
            'USDGBP': 'GBP_USD',
            'USDJPY': 'USD_JPY',
            'USDCAD': 'USD_CAD',
            'USDAUD': 'AUD_USD',
            'USDNZD': 'NZD_USD',
            'USDCHF': 'USD_CHF',
            'USDSEK': 'USD_SEK'
        }
        
    def connect(self) -> bool:
        """Test connection by fetching account info"""
        try:
            r = accounts.AccountSummary(accountID=self.account_id)
            self.client.request(r)
            self.logger.info(f"Connected to OANDA account: {self.account_id}")
            return True
        except Exception as e:
            self.logger.error(f"OANDA connection failed: {e}")
            return False
    
    def get_account_balance(self) -> float:
        """Get account balance in USD"""
        try:
            r = accounts.AccountSummary(accountID=self.account_id)
            response = self.client.request(r)
            balance = float(response['account']['balance'])
            return balance
        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    def get_positions(self) -> Dict[str, float]:
        """Get open positions"""
        try:
            r = trades.OpenTrades(accountID=self.account_id)
            response = self.client.request(r)
            
            positions = {}
            for trade in response.get('trades', []):
                instrument = trade['instrument']
                units = float(trade['currentUnits'])
                
                # Convert OANDA format back to our format
                for our_pair, oanda_pair in self.currency_map.items():
                    if instrument == oanda_pair:
                        positions[our_pair] = units
                        break
            
            return positions
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return {}
    
    def _convert_currency_pair(self, currency_pair: str) -> str:
        """Convert our format to OANDA format"""
        return self.currency_map.get(currency_pair, currency_pair)
    
    def place_market_order(self, currency_pair: str, quantity: float) -> str:
        """Place market order on OANDA"""
        try:
            instrument = self._convert_currency_pair(currency_pair)
            
            order_data = {
                "order": {
                    "instrument": instrument,
                    "units": str(int(quantity)),
                    "type": "MARKET",
                    "positionFill": "DEFAULT"
                }
            }
            
            r = orders.OrderCreate(accountID=self.account_id, data=order_data)
            response = self.client.request(r)
            
            order_id = response['orderCreateTransaction']['id']
            self.logger.info(f"Market order placed: {instrument} {quantity} units, ID: {order_id}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Failed to place market order: {e}")
            return ""
    
    def place_limit_order(self, currency_pair: str, quantity: float, limit_price: float) -> str:
        """Place limit order on OANDA"""
        try:
            instrument = self._convert_currency_pair(currency_pair)
            
            order_data = {
                "order": {
                    "instrument": instrument,
                    "units": str(int(quantity)),
                    "price": str(round(limit_price, 5)),
                    "type": "LIMIT",
                    "positionFill": "DEFAULT"
                }
            }
            
            r = orders.OrderCreate(accountID=self.account_id, data=order_data)
            response = self.client.request(r)
            
            order_id = response['orderCreateTransaction']['id']
            self.logger.info(f"Limit order placed: {instrument} {quantity} @ {limit_price}, ID: {order_id}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Failed to place limit order: {e}")
            return ""
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            r = orders.OrderCancel(accountID=self.account_id, orderID=order_id)
            self.client.request(r)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> str:
        """Get order status"""
        try:
            r = orders.OrderDetails(accountID=self.account_id, orderID=order_id)
            response = self.client.request(r)
            state = response['order']['state']
            
            # Map OANDA states to our standard states
            state_map = {
                'PENDING': 'pending',
                'FILLED': 'filled',
                'CANCELLED': 'cancelled',
                'TRIGGERED': 'filled'
            }
            return state_map.get(state, 'unknown')
            
        except Exception as e:
            self.logger.error(f"Failed to get order status: {e}")
            return 'unknown'
    
    def close_all_positions(self) -> bool:
        """Close all open positions"""
        try:
            positions = self.get_positions()
            
            for currency_pair, quantity in positions.items():
                # Close position by placing opposite order
                close_quantity = -quantity
                self.place_market_order(currency_pair, close_quantity)
                time.sleep(0.1)  # Small delay between orders
            
            self.logger.warning("All positions closed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to close all positions: {e}")
            return False


# ============================================================================
# INTERACTIVE BROKERS INTEGRATION
# ============================================================================

class InteractiveBrokersBroker(BrokerInterface):
    """Interactive Brokers integration"""
    
    def __init__(self, config: BrokerConfig):
        if not IB_AVAILABLE:
            raise ImportError("ib_insync library not installed. Run: pip install ib_insync")
        
        self.config = config
        self.ib = IB()
        self.logger = logging.getLogger(__name__)
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to IB TWS or Gateway"""
        try:
            # Default ports: 7497 for TWS paper, 7496 for TWS live, 4002 for Gateway paper, 4001 for Gateway live
            port = 7497 if self.config.practice else 7496
            self.ib.connect('127.0.0.1', port, clientId=1)
            self.connected = True
            self.logger.info(f"Connected to Interactive Brokers (port {port})")
            return True
        except Exception as e:
            self.logger.error(f"IB connection failed: {e}")
            return False
    
    def get_account_balance(self) -> float:
        """Get account balance"""
        try:
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'TotalCashValue' and av.currency == 'USD':
                    return float(av.value)
            return 0.0
        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    def get_positions(self) -> Dict[str, float]:
        """Get open positions"""
        try:
            positions = self.ib.positions()
            position_dict = {}
            
            for pos in positions:
                if isinstance(pos.contract, Forex):
                    # Convert IB forex format to our format
                    pair = f"USD{pos.contract.symbol}" if pos.contract.currency == 'USD' else f"{pos.contract.symbol}USD"
                    position_dict[pair] = float(pos.position)
            
            return position_dict
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return {}
    
    def _create_forex_contract(self, currency_pair: str) -> Forex:
        """Create IB Forex contract"""
        # Example: USDEUR -> EUR.USD
        base = currency_pair[:3]
        quote = currency_pair[3:]
        return Forex(pair=f"{quote}{base}")
    
    def place_market_order(self, currency_pair: str, quantity: float) -> str:
        """Place market order"""
        try:
            contract = self._create_forex_contract(currency_pair)
            order = MarketOrder('BUY' if quantity > 0 else 'SELL', abs(quantity))
            
            trade = self.ib.placeOrder(contract, order)
            self.logger.info(f"Market order placed: {currency_pair} {quantity}")
            return str(trade.order.orderId)
            
        except Exception as e:
            self.logger.error(f"Failed to place market order: {e}")
            return ""
    
    def place_limit_order(self, currency_pair: str, quantity: float, limit_price: float) -> str:
        """Place limit order"""
        try:
            contract = self._create_forex_contract(currency_pair)
            order = LimitOrder('BUY' if quantity > 0 else 'SELL', abs(quantity), limit_price)
            
            trade = self.ib.placeOrder(contract, order)
            self.logger.info(f"Limit order placed: {currency_pair} {quantity} @ {limit_price}")
            return str(trade.order.orderId)
            
        except Exception as e:
            self.logger.error(f"Failed to place limit order: {e}")
            return ""
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        try:
            self.ib.cancelOrder(int(order_id))
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> str:
        """Get order status"""
        try:
            trades = self.ib.trades()
            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    status = trade.orderStatus.status
                    # Map IB status to our standard
                    if status in ['Submitted', 'PreSubmitted']:
                        return 'pending'
                    elif status == 'Filled':
                        return 'filled'
                    elif status == 'Cancelled':
                        return 'cancelled'
            return 'unknown'
        except Exception as e:
            self.logger.error(f"Failed to get order status: {e}")
            return 'unknown'
    
    def close_all_positions(self) -> bool:
        """Close all positions"""
        try:
            positions = self.get_positions()
            
            for currency_pair, quantity in positions.items():
                self.place_market_order(currency_pair, -quantity)
                time.sleep(0.1)
            
            self.logger.warning("All positions closed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to close all positions: {e}")
            return False


# ============================================================================
# PAPER TRADING SIMULATOR
# ============================================================================

class PaperTradingBroker(BrokerInterface):
    """Paper trading simulator (no real money)"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.balance = 100000.0  # Start with $100k
        self.positions: Dict[str, float] = {}
        self.orders: Dict[str, dict] = {}
        self.order_counter = 0
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Always succeeds for paper trading"""
        self.logger.info("Paper trading mode activated")
        return True
    
    def get_account_balance(self) -> float:
        """Return simulated balance"""
        return self.balance
    
    def get_positions(self) -> Dict[str, float]:
        """Return simulated positions"""
        return self.positions.copy()
    
    def place_market_order(self, currency_pair: str, quantity: float) -> str:
        """Simulate market order with immediate fill"""
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter}"
        
        # Update positions
        current_position = self.positions.get(currency_pair, 0.0)
        self.positions[currency_pair] = current_position + quantity
        
        # Simulate transaction costs (2 bps)
        cost = abs(quantity) * 0.0002
        self.balance -= cost
        
        self.logger.info(f"Paper market order: {currency_pair} {quantity:+.0f}, Cost: ${cost:.2f}")
        
        # Store order
        self.orders[order_id] = {
            'currency_pair': currency_pair,
            'quantity': quantity,
            'type': 'market',
            'status': 'filled',
            'timestamp': datetime.now()
        }
        
        return order_id
    
    def place_limit_order(self, currency_pair: str, quantity: float, limit_price: float) -> str:
        """Simulate limit order (stays pending)"""
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter}"
        
        self.orders[order_id] = {
            'currency_pair': currency_pair,
            'quantity': quantity,
            'limit_price': limit_price,
            'type': 'limit',
            'status': 'pending',
            'timestamp': datetime.now()
        }
        
        self.logger.info(f"Paper limit order: {currency_pair} {quantity} @ {limit_price}")
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'cancelled'
            self.logger.info(f"Paper order cancelled: {order_id}")
            return True
        return False
    
    def get_order_status(self, order_id: str) -> str:
        """Get order status"""
        if order_id in self.orders:
            return self.orders[order_id]['status']
        return 'unknown'
    
    def close_all_positions(self) -> bool:
        """Close all positions"""
        for currency_pair, quantity in self.positions.items():
            if abs(quantity) > 0:
                self.place_market_order(currency_pair, -quantity)
        
        self.positions.clear()
        self.logger.warning("All paper positions closed")
        return True


# ============================================================================
# BROKER FACTORY
# ============================================================================

def create_broker(config: BrokerConfig) -> BrokerInterface:
    """Factory function to create appropriate broker instance"""
    
    broker_classes = {
        'oanda': OANDABroker,
        'ib': InteractiveBrokersBroker,
        'interactive_brokers': InteractiveBrokersBroker,
        'paper': PaperTradingBroker
    }
    
    broker_name = config.broker_name.lower()
    
    if broker_name not in broker_classes:
        raise ValueError(f"Unknown broker: {config.broker_name}. Available: {list(broker_classes.keys())}")
    
    broker_class = broker_classes[broker_name]
    return broker_class(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Paper Trading
    print("\n=== PAPER TRADING ===")
    paper_config = BrokerConfig(
        broker_name='paper',
        api_key='',
        practice=True
    )
    paper_broker = create_broker(paper_config)
    paper_broker.connect()
    print(f"Balance: ${paper_broker.get_account_balance():,.2f}")
    
    # Place some trades
    paper_broker.place_market_order('USDEUR', 10000)
    paper_broker.place_market_order('USDJPY', -5000)
    print(f"Positions: {paper_broker.get_positions()}")
    
    # Example 2: OANDA (requires valid credentials)
    if OANDA_AVAILABLE:
        print("\n=== OANDA (DEMO) ===")
        oanda_config = BrokerConfig(
            broker_name='oanda',
            api_key='your_oanda_api_key',
            account_id='your_account_id',
            practice=True
        )
        # Uncomment to test with real credentials:
        # oanda_broker = create_broker(oanda_config)
        # oanda_broker.connect()
        # print(f"Balance: ${oanda_broker.get_account_balance():,.2f}")
    
    print("\n✓ Broker integrations ready!")
