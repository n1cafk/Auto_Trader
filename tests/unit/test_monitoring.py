from __future__ import annotations

import json
from pathlib import Path

from src.monitoring.alerts import send_webhook_alert
from src.monitoring.logger import configure_logger, log_event


class _DummyResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_logger_writes_json_line(tmp_path: Path) -> None:
    log_path = tmp_path / "events.log"
    logger = configure_logger(log_path)
    log_event(logger, "test_event", {"value": 42})

    content = log_path.read_text(encoding="utf-8").strip()
    payload = json.loads(content)
    assert payload["event"] == "test_event"
    assert payload["value"] == 42


def test_send_webhook_alert_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.monitoring.alerts.urlopen",
        lambda req, timeout: _DummyResponse(204),
    )
    ok = send_webhook_alert("https://example.com/hook", "risk_alert", {"x": 1})
    assert ok is True


def test_send_webhook_alert_failure(monkeypatch) -> None:
    def _raise(*_, **__):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.monitoring.alerts.urlopen", _raise)
    ok = send_webhook_alert("https://example.com/hook", "risk_alert", {"x": 1})
    assert ok is False
