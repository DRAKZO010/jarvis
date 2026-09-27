"""Shared test fixtures.

Importing ``jarvis`` initialises every MCP, which loads the Vosk model and
creates provider clients. That is unavoidable for tests that exercise routing
or the HTTP API, so the module is imported once per session. Nothing in the
suite performs a real network call, plays audio, or writes to the user's
runtime JSON stores.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def app_module():
    """Import ``jarvis`` once, with the microphone and audio muted.

    ``voice_mcp.initialize()`` would otherwise open the real capture device
    during a test run and dispatch anything it overhears.
    """
    import logging
    import os

    os.environ["JARVIS_DISABLE_MIC"] = "1"
    logging.disable(logging.WARNING)
    try:
        import jarvis
    finally:
        logging.disable(logging.NOTSET)
    jarvis.voice_mcp.stop_listener()
    return jarvis


@pytest.fixture(autouse=True)
def reset_app_timers(request):
    """Clear the module-level cooldown/dedupe timers between tests.

    ``jarvis`` is a singleton module, so without this a test that exercises the
    speech cooldown leaves the next test inside that window and fails depending
    on execution order.
    """
    if "app_module" not in request.fixturenames:
        yield
        return

    import jarvis

    jarvis._last_spoken_at = 0.0
    jarvis._last_command = ("", 0.0)
    jarvis._last_error = None
    yield
    jarvis._last_spoken_at = 0.0
    jarvis._last_command = ("", 0.0)


@pytest.fixture
def clean_env(monkeypatch):
    """Remove API keys so offline code paths are exercised."""
    for key in ("OPENROUTER_API_KEY", "FISH_API_KEY", "STREAMELEMENTS_API_KEY",
                "GOOGLE_API_KEY", "TTS_BACKEND"):
        monkeypatch.delenv(key, raising=False)
    return os.environ


@pytest.fixture
def mic_enabled(monkeypatch):
    """Allow a test to exercise the real start/stop path without a device."""
    monkeypatch.delenv("JARVIS_DISABLE_MIC", raising=False)


@pytest.fixture
def silence_voice(monkeypatch, app_module):
    """Stop test runs from queueing real TTS audio."""
    monkeypatch.setattr(app_module.voice_mcp, "speak", lambda text: None, raising=False)
    monkeypatch.setattr(app_module.voice_mcp, "speak_stream", lambda *a, **k: iter(()), raising=False)
    return app_module.voice_mcp
