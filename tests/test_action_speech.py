"""Action-result speech tests.

Most of the actions routed through ``_speak_during_act`` are reports: system
status, task and window lists, the clipboard. Their return value was discarded,
so the user heard "Checking system status..." and never the answer. These tests
lock in that the answer reaches ``speak``, and that the acknowledgement is not
duplicated in front of it.
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def spoken(monkeypatch, app_module):
    """Collect everything the app tries to say, in order, with timestamps."""
    said = []

    def _record(text):
        said.append((time.monotonic(), text))

    monkeypatch.setattr(app_module, "speak", _record, raising=False)
    return said


def _texts(spoken):
    return [text for _, text in spoken]


def _settle(seconds):
    """Wait out the grace period and the action thread."""
    time.sleep(seconds)


def test_report_result_is_spoken(silence_voice, app_module, spoken):
    app_module._speak_during_act(lambda: "CPU at 30 percent.", "Checking...")
    _settle(app_module.ACTION_ANSWER_GRACE_S + 0.4)
    assert "CPU at 30 percent." in _texts(spoken)


def test_fast_report_is_not_preceded_by_the_acknowledgement(silence_voice, app_module, spoken):
    """The acknowledgement would bury the only part that said anything."""
    app_module._speak_during_act(lambda: "CPU at 30 percent.", "Checking...")
    _settle(app_module.ACTION_ANSWER_GRACE_S + 0.4)
    assert _texts(spoken) == ["CPU at 30 percent."]


def test_slow_action_is_announced_before_its_result(silence_voice, app_module, spoken):
    def slow():
        time.sleep(app_module.ACTION_ANSWER_GRACE_S + 0.5)
        return "Opened Spotify."

    app_module._speak_during_act(slow, "Opening Spotify...")
    _settle(app_module.ACTION_ANSWER_GRACE_S * 4)
    assert _texts(spoken) == ["Opening Spotify...", "Opened Spotify."]


def test_action_without_a_result_still_acknowledges(silence_voice, app_module, spoken):
    app_module._speak_during_act(lambda: None, "Muting system volume.")
    _settle(app_module.ACTION_ANSWER_GRACE_S + 0.4)
    assert _texts(spoken) == ["Muting system volume."]


def test_acknowledgement_is_never_spoken_twice(silence_voice, app_module, spoken):
    """The grace timer and the action thread race; only one may announce."""
    app_module._speak_during_act(lambda: None, "Taking screenshot...")
    _settle(app_module.ACTION_ANSWER_GRACE_S * 3)
    assert _texts(spoken).count("Taking screenshot...") == 1


def test_long_result_is_capped_for_speech(silence_voice, app_module, spoken):
    """Thirty window titles read aloud is worse than useless."""
    long_report = "x" * (app_module.ACTION_RESULT_MAX_CHARS * 3)
    app_module._speak_during_act(lambda: long_report, "Listing windows.")
    _settle(app_module.ACTION_ANSWER_GRACE_S + 0.4)
    said = _texts(spoken)[0]
    assert len(said) < len(long_report)
    assert len(said) <= app_module.ACTION_RESULT_MAX_CHARS + 40


def test_failing_action_does_not_raise(silence_voice, app_module, spoken, monkeypatch):
    def boom():
        raise RuntimeError("device went away")

    # Self-heal would ask the LLM for a fix, which is a real network call. The
    # suite makes none, so it is stubbed out here.
    monkeypatch.setattr(app_module, "_try_self_heal", lambda *a, **kw: False,
                        raising=False)
    app_module._speak_during_act(boom, "Opening Spotify...")
    _settle(app_module.ACTION_ANSWER_GRACE_S * 3)
    # The user is told something rather than left in silence, and the error is
    # recorded for "fix the error".
    assert _texts(spoken)
    assert app_module._last_error is not None
    assert "device went away" in app_module._last_error


def test_bare_string_exception_is_logged_with_traceback(silence_voice, app_module, spoken, monkeypatch):
    def boom():
        raise "nope"

    monkeypatch.setattr(app_module, "_try_self_heal", lambda *a, **kw: False,
                        raising=False)
    app_module._last_error = None
    import logging
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.ERROR)
    app_module.log.addHandler(handler)
    try:
        app_module._speak_during_act(boom, "Opening Spotify...")
        _settle(app_module.ACTION_ANSWER_GRACE_S * 3)
    finally:
        app_module.log.removeHandler(handler)

    logged = stream.getvalue()
    assert "Action error in" in logged
    assert "nope" in logged
    assert "Traceback" in logged
    assert app_module._last_error is not None
    # The original bare string is preserved in the logged/recorded error.
    assert "nope" in app_module._last_error


