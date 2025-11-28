"""
Alert System for Live Trading
Supports: Email, Slack, SMS (Twilio), and console notifications
"""

import logging
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

# Optional imports
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertSystem:
    """Unified alert system for trading notifications"""
    
    def __init__(self, config_file: str = 'trading_config.json'):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.config = config.get('monitoring', {})
        except Exception as e:
            self.logger.error(f"Failed to load alert config: {e}")
            self.config = {}
        
        # Email setup
        self.email_enabled = self.config.get('email_alerts', False)
        self.email_from = self.config.get('email_from', '')
        self.email_to = self.config.get('email_to', '')
        self.smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.smtp_password = self.config.get('smtp_password', '')
        
        # Slack setup
        self.slack_enabled = self.config.get('slack_alerts', False) and SLACK_AVAILABLE
        if self.slack_enabled:
            slack_token = self.config.get('slack_token', '')
            self.slack_client = WebClient(token=slack_token)
            self.slack_channel = self.config.get('slack_channel', '#trading-alerts')
        else:
            self.slack_client = None
        
        # SMS setup (Twilio)
        self.sms_enabled = self.config.get('sms_alerts', False) and TWILIO_AVAILABLE
        if self.sms_enabled:
            twilio_sid = self.config.get('twilio_account_sid', '')
            twilio_token = self.config.get('twilio_auth_token', '')
            self.twilio_client = Client(twilio_sid, twilio_token)
            self.twilio_from = self.config.get('twilio_from_number', '')
            self.twilio_to = self.config.get('twilio_to_number', '')
        else:
            self.twilio_client = None
        
        # Alert thresholds
        self.alert_on_trade = self.config.get('alert_on_trade', True)
        self.alert_on_error = self.config.get('alert_on_error', True)
        self.alert_on_stop_loss = self.config.get('alert_on_stop_loss', True)
    
    def send_alert(self, 
                   message: str, 
                   level: AlertLevel = AlertLevel.INFO,
                   data: Optional[Dict[str, Any]] = None):
        """Send alert through all enabled channels"""
        
        # Format message with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] [{level.value.upper()}] {message}"
        
        # Add data if provided
        if data:
            formatted_message += f"\nDetails: {json.dumps(data, indent=2)}"
        
        # Console (always enabled)
        self._send_console(formatted_message, level)
        
        # Email
        if self.email_enabled and self._should_send(level):
            self._send_email(message, level, data)
        
        # Slack
        if self.slack_enabled and self._should_send(level):
            self._send_slack(message, level, data)
        
        # SMS (only for critical alerts)
        if self.sms_enabled and level == AlertLevel.CRITICAL:
            self._send_sms(message)
    
    def _should_send(self, level: AlertLevel) -> bool:
        """Determine if alert should be sent based on level"""
        # Only send warnings and above via email/slack
        return level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
    
    def _send_console(self, message: str, level: AlertLevel):
        """Log to console"""
        if level == AlertLevel.CRITICAL:
            self.logger.critical(message)
        elif level == AlertLevel.ERROR:
            self.logger.error(message)
        elif level == AlertLevel.WARNING:
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def _send_email(self, message: str, level: AlertLevel, data: Optional[Dict] = None):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = f"Trading Alert - {level.value.upper()}: {message[:50]}"
            
            # Email body
            body = f"""
FX Carry Trading System Alert

Level: {level.value.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Message:
{message}

"""
            if data:
                body += f"\nDetails:\n{json.dumps(data, indent=2)}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.smtp_password)
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent to {self.email_to}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
    
    def _send_slack(self, message: str, level: AlertLevel, data: Optional[Dict] = None):
        """Send Slack notification"""
        try:
            # Emoji based on level
            emoji_map = {
                AlertLevel.INFO: ':information_source:',
                AlertLevel.WARNING: ':warning:',
                AlertLevel.ERROR: ':x:',
                AlertLevel.CRITICAL: ':rotating_light:'
            }
            emoji = emoji_map.get(level, ':bell:')
            
            # Format message
            slack_message = f"{emoji} *{level.value.upper()}*\n{message}"
            
            if data:
                slack_message += f"\n```{json.dumps(data, indent=2)}```"
            
            # Send to Slack
            response = self.slack_client.chat_postMessage(
                channel=self.slack_channel,
                text=slack_message
            )
            
            self.logger.info(f"Slack alert sent to {self.slack_channel}")
            
        except SlackApiError as e:
            self.logger.error(f"Failed to send Slack message: {e.response['error']}")
        except Exception as e:
            self.logger.error(f"Failed to send Slack message: {e}")
    
    def _send_sms(self, message: str):
        """Send SMS alert (critical only)"""
        try:
            # Truncate message to 160 characters for SMS
            sms_message = f"TRADING ALERT: {message[:140]}"
            
            message = self.twilio_client.messages.create(
                body=sms_message,
                from_=self.twilio_from,
                to=self.twilio_to
            )
            
            self.logger.info(f"SMS alert sent to {self.twilio_to}")
            
        except Exception as e:
            self.logger.error(f"Failed to send SMS: {e}")
    
    # Convenience methods for common alerts
    
    def alert_trade_executed(self, currency_pair: str, quantity: float, price: float):
        """Alert when trade is executed"""
        if self.alert_on_trade:
            message = f"Trade executed: {currency_pair} {quantity:+,.0f} @ {price:.5f}"
            data = {
                'currency_pair': currency_pair,
                'quantity': quantity,
                'price': price
            }
            self.send_alert(message, AlertLevel.INFO, data)
    
    def alert_position_closed(self, currency_pair: str, pnl: float):
        """Alert when position is closed"""
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING
        message = f"Position closed: {currency_pair}, P&L: ${pnl:+,.2f}"
        data = {
            'currency_pair': currency_pair,
            'pnl': pnl
        }
        self.send_alert(message, level, data)
    
    def alert_stop_loss_triggered(self, reason: str, loss: float):
        """Alert when stop-loss is triggered"""
        if self.alert_on_stop_loss:
            message = f"STOP-LOSS TRIGGERED: {reason}, Loss: ${loss:,.2f}"
            data = {
                'reason': reason,
                'loss': loss
            }
            self.send_alert(message, AlertLevel.CRITICAL, data)
    
    def alert_risk_limit_breached(self, limit_type: str, current_value: float, limit_value: float):
        """Alert when risk limit is breached"""
        message = f"Risk limit breached: {limit_type}"
        data = {
            'limit_type': limit_type,
            'current_value': current_value,
            'limit_value': limit_value
        }
        self.send_alert(message, AlertLevel.ERROR, data)
    
    def alert_data_feed_error(self, feed_name: str, error: str):
        """Alert when data feed fails"""
        message = f"Data feed error: {feed_name}"
        data = {
            'feed_name': feed_name,
            'error': str(error)
        }
        self.send_alert(message, AlertLevel.ERROR, data)
    
    def alert_system_started(self, mode: str, capital: float):
        """Alert when system starts"""
        message = f"Trading system started in {mode.upper()} mode with ${capital:,.2f}"
        data = {
            'mode': mode,
            'capital': capital
        }
        self.send_alert(message, AlertLevel.INFO, data)
    
    def alert_system_stopped(self, reason: str):
        """Alert when system stops"""
        message = f"Trading system stopped: {reason}"
        data = {'reason': reason}
        self.send_alert(message, AlertLevel.WARNING, data)
    
    def alert_performance_milestone(self, milestone: str, value: float):
        """Alert on performance milestones"""
        message = f"Performance milestone: {milestone}"
        data = {
            'milestone': milestone,
            'value': value
        }
        self.send_alert(message, AlertLevel.INFO, data)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Alert System...")
    print("="*60)
    
    # Initialize alert system
    alert_system = AlertSystem()
    
    # Test different alert types
    print("\n1. Testing INFO alert...")
    alert_system.send_alert("System initialized successfully", AlertLevel.INFO)
    
    print("\n2. Testing WARNING alert...")
    alert_system.alert_risk_limit_breached("Max Position Size", 35.5, 30.0)
    
    print("\n3. Testing trade alert...")
    alert_system.alert_trade_executed("USDEUR", 10000, 0.9234)
    
    print("\n4. Testing stop-loss alert...")
    alert_system.alert_stop_loss_triggered("Max drawdown exceeded", 15000)
    
    print("\n5. Testing system start alert...")
    alert_system.alert_system_started("paper", 100000)
    
    print("\n" + "="*60)
    print("Alert system test complete!")
    print("\nNOTE: Email/Slack/SMS alerts only work with valid credentials")
    print("      Configure in trading_config.json under 'monitoring' section")
