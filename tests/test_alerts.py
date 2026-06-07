"""
Tests for alert logic. Uses a fake DB so no MongoDB connection is needed.
We verify: non-hazardous AQI -> no alert; hazardous -> alert written once;
the same alert on the same day is deduped (not written twice).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import aqi_predictor.alerts as alerts_mod
from aqi_predictor.alerts import SEVERITY, maybe_log_alert


class _FakeResult:
    def __init__(self, upserted):
        self.upserted_id = upserted


class _FakeAlerts:
    """Mimics a collection with unique alert_key + upsert($setOnInsert)."""
    def __init__(self):
        self.store = {}

    def update_one(self, flt, update, upsert=False):
        key = flt["alert_key"]
        if key in self.store:
            return _FakeResult(None)            # already exists -> deduped
        self.store[key] = update["$setOnInsert"]
        return _FakeResult(key)                 # newly inserted


class _FakeDB:
    def __init__(self, coll):
        self._coll = coll

    def __getitem__(self, name):
        return self._coll


def _patch_db(monkeypatch_target):
    coll = _FakeAlerts()
    alerts_mod.get_db = lambda: _FakeDB(coll)   # type: ignore[attr-defined]
    return coll


def test_no_alert_for_good_air():
    _patch_db(alerts_mod)
    assert maybe_log_alert("Karachi", "24h", 2) is None


def test_alert_for_poor_air():
    _patch_db(alerts_mod)
    now = datetime(2026, 6, 8, tzinfo=timezone.utc)
    doc = maybe_log_alert("Karachi", "24h", 4, now=now)
    assert doc is not None
    assert doc["severity"] == "poor"
    assert doc["aqi_category"] == "Poor"


def test_alert_is_deduped_same_day():
    _patch_db(alerts_mod)
    now = datetime(2026, 6, 8, tzinfo=timezone.utc)
    first = maybe_log_alert("Karachi", "24h", 5, now=now)
    second = maybe_log_alert("Karachi", "24h", 5, now=now)
    assert first is not None        # first write happens
    assert second is None           # duplicate suppressed


def test_severity_mapping():
    assert SEVERITY[4] == "poor"
    assert SEVERITY[5] == "very_poor"
