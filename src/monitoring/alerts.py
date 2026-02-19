"""Simple alert hooks."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen


def send_webhook_alert(webhook_url: str, event: str, payload: dict[str, Any]) -> bool:
    """Send JSON alert to webhook endpoint."""
    body = json.dumps({"event": event, "payload": payload}).encode("utf-8")
    req = Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=5) as response:
            return 200 <= response.status < 300
    except Exception:
        return False
