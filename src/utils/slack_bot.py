#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:23:33 2024

@author: krishnayadav
"""

import os
import sys
import json
import requests
from typing import Dict, Union, Optional

try:
    # pip install python-dotenv
    from dotenv import load_dotenv
except ImportError as e:
    raise ImportError("python-dotenv is required. Install with `pip install python-dotenv`") from e


class SlackBot:
    """
    Send notifications to Slack via Incoming Webhook.
    Reads the webhook URL from environment (loaded from .env).
    """

    def __init__(
        self,
        env_var: str = "SLACK_WEBHOOK_URL",
        dotenv_path: Optional[str] = None,
        load_override: bool = False,
        timeout: float = 10.0,
    ):
        """
        Args:
            env_var: Environment variable name that holds the webhook URL.
            dotenv_path: Optional explicit path to your .env file. If None, defaults are used.
            load_override: If True, variables from dotenv will override existing env vars.
            timeout: Requests timeout in seconds.
        """
        # Load environment variables from .env
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path, override=load_override)
        else:
            load_dotenv(override=load_override)

        self.webhook_url = os.getenv(env_var)
        if not self.webhook_url:
            raise ValueError(
                f"Slack webhook URL not found. Ensure `{env_var}` is set in your environment or .env file."
            )

        self.timeout = timeout

    @staticmethod
    def _format_message(message: Union[Dict, str]) -> str:
        """
        Convert a dict or string into a Slack-friendly text payload.
        """
        if isinstance(message, str):
            return message

        if isinstance(message, dict):
            # Stable key ordering for readability
            lines = []
            for key in sorted(message.keys()):
                val = message[key]
                try:
                    # Pretty-print nested structures
                    if isinstance(val, (dict, list)):
                        val_str = json.dumps(val, ensure_ascii=False, indent=2)
                    else:
                        val_str = str(val)
                except Exception:
                    val_str = str(val)
                lines.append(f"{key}: {val_str}")
            return "\n".join(lines)

        # Fallback
        return str(message)

    def _post(self, text: str) -> None:
        """
        Internal helper to POST to Slack with robust error handling.
        """
        payload = {"text": text}
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
            # Slack returns 200 on success; raise for anything else
            if resp.status_code != 200:
                raise requests.HTTPError(f"{resp.status_code} {resp.text}")
            print("Message posted successfully")
        except Exception as e:
            print(f"Failed to post message to Slack: {e}")

    def post_to_slack(self, message: Union[Dict, str]) -> None:
        """
        Post a formatted message (dict or str) to Slack.
        """
        self._post(self._format_message(message))

    def post_error_to_slack(self, error_message: str, context: Optional[Dict] = None) -> None:
        """
        Post an error message (optionally with context) to Slack.
        """
        prefix = ":warning: *Error* :warning:\n"
        if context:
            combined = prefix + self._format_message({"message": error_message, "context": context})
        else:
            combined = prefix + error_message
        self._post(combined)


if __name__ == "__main__":
    # Example usage:
    # .env should contain: SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
    bot = SlackBot()  # or SlackBot(dotenv_path="/path/to/.env")
    bot.post_to_slack({"event": "deploy", "status": "success", "version": "v1.2.3"})
    bot.post_error_to_slack("Database connection failed", context={"service": "api", "region": "ap-south-1"})
