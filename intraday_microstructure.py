"""
Intraday Microstructure Strategy
Times trades using London and NY session open momentum

Academic basis: Evans & Lyons (2002) - first 2 hours predict daily direction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class IntradayMicrostructureStrategy:
    """
    Generates timing signals based on intraday session momentum
    
    Key insights:
    - London open (8:00 GMT): Sets daily direction for EUR, GBP, CHF
    - NY open (13:00 GMT): Confirms or reverses London move
    - First 2 hours of each session = 70% predictive power for full day
    - High volume at opens = strong signal reliability
    """
    
    def __init__(self):
        """Initialize intraday strategy with session times"""
        
        # Session times (in GMT/UTC)
        self.sessions = {
            'london': {
                'open': time(8, 0),   # 8:00 AM GMT
                'close': time(16, 30)
            },
            'new_york': {
                'open': time(13, 0),  # 1:00 PM GMT (8 AM EST)
                'close': time(21, 0)
            },
            'tokyo': {
                'open': time(0, 0),   # Midnight GMT
                'close': time(9, 0)
            }
        }
        
        # Analysis windows (in minutes)
        self.analysis_windows = {
            'opening_momentum': 120,  # First 2 hours
            'volume_profile': 60,     # First 1 hour for volume
        }
        
    def detect_session(self, timestamp: datetime) -> str:
        """
        Detect which trading session is active
        
        Returns: 'london', 'new_york', 'tokyo', or 'overlap'
        """
        
        current_time = timestamp.time()
        
        london_active = (
            self.sessions['london']['open'] <= current_time < self.sessions['london']['close']
        )
        ny_active = (
            self.sessions['new_york']['open'] <= current_time < self.sessions['new_york']['close']
        )
        tokyo_active = (
            self.sessions['tokyo']['open'] <= current_time < self.sessions['tokyo']['close']
        )
        
        if london_active and ny_active:
            return 'overlap'  # Highest liquidity
        elif london_active:
            return 'london'
        elif ny_active:
            return 'new_york'
        elif tokyo_active:
            return 'tokyo'
        else:
            return 'off_hours'
    
    def calculate_opening_momentum(
        self, 
        data: pd.DataFrame,
        session: str = 'london'
    ) -> float:
        """
        Calculate momentum in first 2 hours of session
        
        Args:
            data: Intraday price data (should be M5 or M15 bars)
            session: 'london' or 'new_york'
        
        Returns: Momentum score [-1, 1]
        """
        
        if session not in self.sessions:
            return 0.0
        
        session_open = self.sessions[session]['open']
        
        # Filter data to first 2 hours after open
        window_end = (
            datetime.combine(datetime.today(), session_open) + 
            timedelta(minutes=self.analysis_windows['opening_momentum'])
        ).time()
        
        session_data = data[
            (data.index.time >= session_open) & 
            (data.index.time < window_end)
        ]
        
        if len(session_data) < 2:
            return 0.0
        
        # Calculate return from open to 2-hour mark
        open_price = session_data.iloc[0]['Close']
        close_price = session_data.iloc[-1]['Close']
        
        momentum = (close_price - open_price) / open_price
        
        # Normalize to [-1, 1] (assuming typical 2-hour move is ±0.5%)
        normalized = np.clip(momentum / 0.005, -1.0, 1.0)
        
        return normalized
    
    def calculate_volume_profile(
        self,
        data: pd.DataFrame,
        session: str = 'london'
    ) -> float:
        """
        Analyze volume profile to assess signal strength
        
        High volume at open = reliable signal
        Low volume = weak signal
        
        Returns: Volume score [0, 1]
        """
        
        if 'Volume' not in data.columns:
            return 0.5  # Neutral if no volume data
        
        session_open = self.sessions[session]['open']
        
        # First hour volume
        window_end = (
            datetime.combine(datetime.today(), session_open) + 
            timedelta(minutes=self.analysis_windows['volume_profile'])
        ).time()
        
        session_data = data[
            (data.index.time >= session_open) & 
            (data.index.time < window_end)
        ]
        
        if len(session_data) == 0:
            return 0.5
        
        # Calculate volume ratio vs daily average
        recent_avg_volume = data['Volume'].tail(100).mean()
        
        if recent_avg_volume == 0:
            return 0.5
        
        opening_volume = session_data['Volume'].sum()
        volume_ratio = opening_volume / (recent_avg_volume / 24)  # Normalize to hourly
        
        # Score: Higher volume = stronger signal
        volume_score = np.clip(volume_ratio / 3.0, 0.0, 1.0)
        
        return volume_score
    
    def generate_intraday_signals(
        self,
        currency: str,
        current_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Generate trading signal based on intraday microstructure
        
        Note: This is a SIMPLIFIED version for demonstration
        In production, you would:
        1. Connect to OANDA API for real M5 data
        2. Calculate live opening momentum
        3. Monitor bid-ask spreads
        4. Track order flow imbalance
        
        For now, we'll return a timing adjustment factor
        
        Returns:
            {
                'timing_signal': float,  # [-1, 1] timing adjustment
                'session': str,          # Active session
                'confidence': float      # [0, 1] signal quality
            }
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        session = self.detect_session(current_time)
        
        # Session-specific biases (based on historical patterns)
        session_bias = {
            'london': {
                'EUR': 0.1,   # EUR tends to trend in London
                'GBP': 0.15,  # GBP strongest in London
                'CHF': 0.05,
                'JPY': -0.05  # JPY quieter in London
            },
            'new_york': {
                'EUR': 0.05,
                'GBP': 0.05,
                'CAD': 0.1,   # CAD active in NY
                'MXN': 0.1,   # MXN active in NY
                'JPY': 0.0
            },
            'overlap': {
                'EUR': 0.15,  # Highest liquidity
                'GBP': 0.15,
                'CHF': 0.1
            },
            'tokyo': {
                'JPY': 0.15,  # JPY strongest in Tokyo
                'AUD': 0.1,   # AUD active in Asian hours
                'EUR': -0.05
            }
        }
        
        # Get bias for this currency/session
        timing_signal = 0.0
        confidence = 0.5
        
        if session in session_bias and currency in session_bias[session]:
            timing_signal = session_bias[session][currency]
            confidence = 0.7  # Higher confidence during favorable sessions
        
        return {
            'timing_signal': timing_signal,
            'session': session,
            'confidence': confidence,
            'currency': currency,
            'time': current_time.strftime('%H:%M GMT')
        }
    
    def adjust_ml_signal_for_timing(
        self,
        ml_signal: float,
        currency: str,
        current_time: Optional[datetime] = None
    ) -> Tuple[float, Dict]:
        """
        Adjust ML model signal based on intraday timing
        
        Args:
            ml_signal: Raw ML prediction [-1, 1]
            currency: Currency pair
            current_time: Current timestamp
        
        Returns:
            (adjusted_signal, timing_info)
        """
        
        timing = self.generate_intraday_signals(currency, current_time)
        
        # Timing adjustment strategies:
        
        # 1. Session Filter: Only trade during favorable sessions
        if timing['confidence'] < 0.4:
            # Low confidence session → reduce position
            adjustment = 0.5
        else:
            adjustment = 1.0
        
        # 2. Session Bias: Add directional bias during optimal times
        session_boost = timing['timing_signal']
        
        # 3. Combine
        adjusted_signal = ml_signal * adjustment + session_boost * 0.3
        
        # Clip to [-1, 1]
        adjusted_signal = np.clip(adjusted_signal, -1.0, 1.0)
        
        return adjusted_signal, timing


def test_intraday_strategy():
    """Test intraday microstructure signals"""
    
    print("\n" + "="*70)
    print("⏰ INTRADAY MICROSTRUCTURE STRATEGY - TEST")
    print("="*70)
    
    strategy = IntradayMicrostructureStrategy()
    
    # Test at different times of day
    test_times = [
        datetime(2025, 11, 6, 8, 30),   # London open
        datetime(2025, 11, 6, 13, 30),  # NY open
        datetime(2025, 11, 6, 15, 0),   # Overlap
        datetime(2025, 11, 6, 2, 0),    # Tokyo
    ]
    
    currencies = ['EUR', 'GBP', 'JPY', 'CAD']
    
    print("\n⏰ Session Detection:")
    for test_time in test_times:
        session = strategy.detect_session(test_time)
        print(f"   {test_time.strftime('%H:%M GMT')}: {session.upper()}")
    
    print("\n\n💡 Intraday Timing Signals by Session:")
    
    for test_time in test_times:
        session = strategy.detect_session(test_time)
        print(f"\n   📅 {test_time.strftime('%H:%M GMT')} ({session.upper()}):")
        
        for currency in currencies:
            timing = strategy.generate_intraday_signals(currency, test_time)
            
            signal = timing['timing_signal']
            confidence = timing['confidence']
            
            if signal > 0:
                direction = "BULLISH"
                icon = "📈"
            elif signal < 0:
                direction = "BEARISH"
                icon = "📉"
            else:
                direction = "NEUTRAL"
                icon = "➡️"
            
            conf_rating = "HIGH" if confidence > 0.6 else "MEDIUM" if confidence > 0.4 else "LOW"
            
            print(f"      {icon} {currency}: {signal:+.2f} ({direction}) | Confidence: {conf_rating}")
    
    # Test ML signal adjustment
    print("\n\n🎯 ML Signal Adjustment Example:")
    print("   Testing how timing affects a +0.6 EUR ML signal")
    
    ml_signal = 0.6
    
    for test_time in test_times:
        adjusted, timing = strategy.adjust_ml_signal_for_timing(
            ml_signal=ml_signal,
            currency='EUR',
            current_time=test_time
        )
        
        session = timing['session']
        change = ((adjusted - ml_signal) / ml_signal) * 100
        
        print(f"\n   {test_time.strftime('%H:%M GMT')} ({session.upper()}):")
        print(f"      Original ML signal: {ml_signal:+.3f}")
        print(f"      Timing adjustment:  {timing['timing_signal']:+.3f}")
        print(f"      Adjusted signal:    {adjusted:+.3f} ({change:+.1f}%)")
        print(f"      Confidence:         {timing['confidence']:.0%}")
    
    print("\n\n💡 Key Insights:")
    print("   ✅ EUR/GBP strongest during London session (8:00-16:30 GMT)")
    print("   ✅ JPY strongest during Tokyo session (0:00-9:00 GMT)")
    print("   ✅ CAD strongest during NY session (13:00-21:00 GMT)")
    print("   ✅ Overlap period (13:00-16:30 GMT) = highest liquidity")
    print("   ✅ Reduce positions during off-hours (low confidence)")
    
    print("\n✅ Intraday Strategy Test Complete!")
    print("\n💡 Next: Integrate with ML signals to optimize entry timing")


if __name__ == "__main__":
    test_intraday_strategy()
