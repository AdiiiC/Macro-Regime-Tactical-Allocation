"""
Real-Time Signal Alerts.
Monitors regime transitions and sends notifications via email/Slack.
"""

import smtplib
import json
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class RegimeAlert:
    """Represents a regime transition alert."""
    timestamp: datetime
    previous_regime: str
    new_regime: str
    confidence: float
    allocation_changes: Dict[str, float]
    key_drivers: List[str]


class AlertManager:
    """
    Manages regime transition alerts via multiple channels.
    """

    def __init__(
        self,
        email_config: Optional[Dict] = None,
        slack_webhook: Optional[str] = None,
        alert_history_path: str = "alerts/history.json",
    ):
        self.email_config = email_config
        self.slack_webhook = slack_webhook
        self.alert_history_path = alert_history_path
        self.alert_history: List[Dict] = []
        self._load_history()

    def check_regime_transition(
        self,
        current_regime: str,
        previous_regime: str,
        confidence: float,
        allocation_changes: Dict[str, float],
        key_drivers: List[str],
    ) -> Optional[RegimeAlert]:
        """
        Check if a regime transition occurred and create alert.
        Only triggers if confidence exceeds threshold.
        """
        if current_regime == previous_regime:
            return None

        if confidence < 0.6:
            logger.info(
                f"Regime transition detected ({previous_regime} → {current_regime}) "
                f"but confidence too low ({confidence:.1%}). No alert sent."
            )
            return None

        alert = RegimeAlert(
            timestamp=datetime.now(),
            previous_regime=previous_regime,
            new_regime=current_regime,
            confidence=confidence,
            allocation_changes=allocation_changes,
            key_drivers=key_drivers,
        )

        self._dispatch_alert(alert)
        self._save_alert(alert)
        return alert

    def _dispatch_alert(self, alert: RegimeAlert):
        """Send alert through all configured channels."""
        message = self._format_alert_message(alert)

        if self.slack_webhook:
            self._send_slack(message, alert)

        if self.email_config:
            self._send_email(message, alert)

        logger.info(f"Alert dispatched: {alert.previous_regime} → {alert.new_regime}")

    def _format_alert_message(self, alert: RegimeAlert) -> str:
        """Format alert into human-readable message."""
        msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔔 REGIME TRANSITION DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M')}
Transition: {alert.previous_regime} → {alert.new_regime}
Confidence: {alert.confidence:.1%}

📊 Allocation Changes Required:
"""
        for asset, change in sorted(
            alert.allocation_changes.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            if abs(change) > 0.01:
                direction = "↑" if change > 0 else "↓"
                msg += f"  {direction} {asset}: {change:+.1%}\n"

        msg += f"\n🔑 Key Drivers:\n"
        for driver in alert.key_drivers:
            msg += f"  • {driver}\n"

        msg += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        return msg

    def _send_slack(self, message: str, alert: RegimeAlert):
        """Send alert to Slack webhook."""
        regime_emojis = {
            "Expansion": "📈",
            "Slowdown": "⚡",
            "Recession": "📉",
            "Recovery": "🌱",
        }
        emoji = regime_emojis.get(alert.new_regime, "🔔")

        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} Regime Transition: {alert.previous_regime} → {alert.new_regime}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Confidence:*\n{alert.confidence:.1%}"},
                        {"type": "mrkdwn", "text": f"*Timestamp:*\n{alert.timestamp.strftime('%Y-%m-%d')}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Key Allocation Changes:*\n" + "\n".join(
                            f"• {asset}: {change:+.1%}"
                            for asset, change in sorted(
                                alert.allocation_changes.items(),
                                key=lambda x: abs(x[1]), reverse=True
                            )[:5]
                        ),
                    },
                },
            ],
        }

        try:
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")

    def _send_email(self, message: str, alert: RegimeAlert):
        """Send alert via email."""
        if not self.email_config:
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[REGIME ALERT] {alert.previous_regime} → {alert.new_regime}"
        msg["From"] = self.email_config.get("from_addr", "")
        msg["To"] = self.email_config.get("to_addr", "")

        # Plain text
        msg.attach(MIMEText(message, "plain"))

        # HTML version
        html = self._format_html_email(alert)
        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(
                self.email_config.get("smtp_server", "smtp.gmail.com"),
                self.email_config.get("smtp_port", 587),
            ) as server:
                server.starttls()
                server.login(
                    self.email_config.get("username", ""),
                    self.email_config.get("password", ""),
                )
                server.sendmail(msg["From"], msg["To"], msg.as_string())
        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    def _format_html_email(self, alert: RegimeAlert) -> str:
        """Generate HTML email body."""
        regime_colors = {
            "Expansion": "#2ecc71",
            "Slowdown": "#f39c12",
            "Recession": "#e74c3c",
            "Recovery": "#3498db",
        }

        color = regime_colors.get(alert.new_regime, "#333")

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: {color}; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                <h1 style="margin: 0;">Regime Transition Alert</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">
                    {alert.previous_regime} → <strong>{alert.new_regime}</strong>
                </p>
            </div>
            <div style="padding: 20px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 8px 8px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Confidence</td>
                        <td style="padding: 8px;">{alert.confidence:.1%}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Timestamp</td>
                        <td style="padding: 8px;">{alert.timestamp.strftime('%B %d, %Y %H:%M')}</td>
                    </tr>
                </table>

                <h3>Allocation Changes</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background: #f5f5f5;">
                        <th style="padding: 8px; text-align: left;">Asset</th>
                        <th style="padding: 8px; text-align: right;">Change</th>
                    </tr>
        """

        for asset, change in sorted(
            alert.allocation_changes.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            if abs(change) > 0.01:
                arrow = "↑" if change > 0 else "↓"
                change_color = "#2ecc71" if change > 0 else "#e74c3c"
                html += f"""
                    <tr>
                        <td style="padding: 6px;">{asset}</td>
                        <td style="padding: 6px; text-align: right; color: {change_color};">
                            {arrow} {change:+.1%}
                        </td>
                    </tr>
                """

        html += """
                </table>
                <p style="font-size: 11px; color: #999; margin-top: 20px;">
                    This is an automated alert from the Macro Regime Detection system.
                    Not investment advice.
                </p>
            </div>
        </body>
        </html>
        """
        return html

    def _save_alert(self, alert: RegimeAlert):
        """Save alert to history."""
        record = {
            "timestamp": alert.timestamp.isoformat(),
            "previous_regime": alert.previous_regime,
            "new_regime": alert.new_regime,
            "confidence": alert.confidence,
        }
        self.alert_history.append(record)

        try:
            import os
            os.makedirs(os.path.dirname(self.alert_history_path), exist_ok=True)
            with open(self.alert_history_path, "w") as f:
                json.dump(self.alert_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alert history: {e}")

    def _load_history(self):
        """Load alert history from file."""
        try:
            with open(self.alert_history_path, "r") as f:
                self.alert_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.alert_history = []

    def get_alert_summary(self) -> pd.DataFrame:
        """Return alert history as DataFrame."""
        if not self.alert_history:
            return pd.DataFrame()
        return pd.DataFrame(self.alert_history)
