"""TTS backend chain, interruption, and broadcast-hook tests.

No real audio device is required: the backends are stubbed except for the
explicitly-marked live tests.
"""

from __future__ import annotations

import time

import pytest

from mcp.voice import VoiceMCP, set_broadcast_hook


@pytest.fixture
def voice(monkeypatch):
    """A VoiceMCP with a stubbed backend chain, so tests never hit the network."""
    v = VoiceMCP.__new__(VoiceMCP)
    VoiceMCP.__init__(v)
    v._backend_order = ["a", "b", "c"]
    v._disabled_backends = set()
    v._active_backend = None
    v._stop_playback.clear()
    v._is_speaking = False
    return v


def _stub_table(results):
    """Build a ``_TTS_BACKENDS`` dict from name -> result/exception."""

    def make(result):
        def fn(mcp, text):
            if isinstance(result, Exception):
                raise result
            return result
        return fn

    return {name: make(res) for name, res in results.items()}


# --- chain selection -------------------------------------------------------

def test_first_working_backend_wins(voice):
    voice._TTS_BACKENDS = _stub_table({"a": True, "b": True})
    assert voice._speak_with_fallback("hello") is True
    assert voice._active_backend == "a"
    assert voice._disabled_backends == set()


def test_failing_backend_is_disabled_and_next_is_used(voice):
    voice._TTS_BACKENDS = _stub_table({"a": RuntimeError("402"), "b": True})
    assert voice._speak_with_fallback("hello") is True
    assert voice._disabled_backends == {"a"}
    assert voice._active_backend == "b"


def test_disabled_backend_is_skipped(voice):
    voice._disabled_backends.add("a")
    voice._TTS_BACKENDS = _stub_table({"a": True, "b": True})
    assert voice._speak_with_fallback("hello") is True
    assert voice._active_backend == "b"


def test_exhausted_chain_returns_false_without_raising(voice):
    voice._TTS_BACKENDS = _stub_table({"a": RuntimeError("x"), "b": False, "c": RuntimeError("y")})
    assert voice._speak_with_fallback("hello") is False
    assert voice._disabled_backends == {"a", "b", "c"}


def test_backend_returning_false_is_disabled(voice):
    voice._TTS_BACKENDS = _stub_table({"a": False, "b": True})
    assert voice._speak_with_fallback("hello") is True
    assert "a" in voice._disabled_backends


# --- interruption ----------------------------------------------------------

def test_interrupt_does_not_disable_a_healthy_backend(voice):
    """A barge-in that lands mid-utterance must not look like a broken backend."""
    calls = []

    def slow(mcp, text):
        calls.append(text)
        time.sleep(1.0)
        return True

    voice._TTS_BACKENDS = {"slow": slow}
    voice._backend_order = ["slow"]

    import threading
    t = threading.Thread(target=lambda: voice._speak_with_fallback("hello"), daemon=True)
    t.start()
    time.sleep(0.15)
    voice._stop_playback.set()          # what interrupt() does
    t.join(timeout=3)

    assert voice._disabled_backends == set(), "healthy backend was disabled by a barge-in"


def test_interrupt_exception_does_not_disable_backend(voice):
    def raising(mcp, text):
        voice._stop_playback.set()
        raise RuntimeError("aborted because of barge-in")

    voice._TTS_BACKENDS = {"boom": raising}
    voice._backend_order = ["boom"]
    assert voice._speak_with_fallback("hello") is False
    assert voice._disabled_backends == set()


def test_stop_flag_short_circuits_before_any_backend_runs(voice):
    called = []

    def track(mcp, text):
        called.append(text)
        return True

    voice._TTS_BACKENDS = {"a": track}
    voice._stop_playback.set()
    assert voice._speak_with_fallback("hello") is False
    assert called == [], "backend ran despite an active stop flag"


# --- broadcast hook --------------------------------------------------------

def test_broadcast_hook_receives_speaking_state(voice):
    seen = []
    set_broadcast_hook(lambda msg: seen.append(msg))
    try:
        voice._broadcast_speaking(True)
        voice._broadcast_speaking(False)
    finally:
        set_broadcast_hook(None)

    assert seen == [
        {"type": "speaking", "speaking": True},
        {"type": "speaking", "speaking": False},
    ]


def test_broadcast_hook_failure_does_not_propagate(voice):
    def boom(msg):
        raise RuntimeError("ui is gone")

    set_broadcast_hook(boom)
    try:
        voice._broadcast_speaking(True)  # must not raise
    finally:
        set_broadcast_hook(None)


def test_voice_module_does_not_import_the_app():
    """Importing mcp.voice must not boot every other MCP as a side effect."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import mcp.voice;"
        "print('jarvis' in sys.modules)" % root
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("False"), "mcp.voice imported the app"


# --- configuration ---------------------------------------------------------

def test_edge_voice_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("EDGE_TTS_VOICE", raising=False)
    v = VoiceMCP.__new__(VoiceMCP)
    VoiceMCP.__init__(v)
    assert v.edge_voice, "a default Edge voice must be set so TTS can start"

    monkeypatch.setenv("EDGE_TTS_VOICE", "en-US-AriaNeural")
    v2 = VoiceMCP.__new__(VoiceMCP)
    VoiceMCP.__init__(v2)
    assert v2.edge_voice == "en-US-AriaNeural"


def test_interrupt_debounce_never_blocks_an_active_stop(voice):
    """Two interrupts in quick succession must both stop playback."""
    voice._is_speaking = True
    voice._last_interrupt = time.monotonic()
    voice.interrupt()
    assert voice._stop_playback.is_set()
    assert not voice._is_speaking
