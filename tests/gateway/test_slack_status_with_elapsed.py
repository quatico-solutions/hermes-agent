"""Sticky live-status phrase + elapsed heartbeat on the Slack status line."""

import time

import pytest

pytest.importorskip("slack_bolt")

from plugins.platforms.slack.adapter import _STATUS_LINE_MAX, _status_with_elapsed


def _started(seconds_ago: float) -> float:
    return time.monotonic() - seconds_ago


def test_short_turn_shows_phrase_only():
    assert _status_with_elapsed("is running pytest…", _started(5)) == "is running pytest…"


def test_no_start_time_shows_phrase_only():
    assert _status_with_elapsed("is running pytest…", None) == "is running pytest…"


def test_long_turn_appends_elapsed_seconds():
    assert _status_with_elapsed("is running pytest…", _started(42)) == "is running pytest… (42s)"


def test_long_turn_appends_elapsed_minutes():
    assert _status_with_elapsed("is running pytest…", _started(123)) == "is running pytest… (2m03s)"


def test_long_phrase_is_trimmed_but_suffix_kept():
    phrase = "is reading " + "a/very/long/path/" * 5 + "file.md"
    out = _status_with_elapsed(phrase, _started(123))
    assert out.endswith("… (2m03s)")
    assert len(out) <= _STATUS_LINE_MAX
