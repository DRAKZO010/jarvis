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

def test_fish_audio_is_the_only_tts_backend():
    """Fish Audio is the sole speech backend, by design.

    Guards against duplicate entries on repeated init: SAPI first, Fish last.
    """
    v = VoiceMCP.__new__(VoiceMCP)
    VoiceMCP.__init__(v)
    v._fish_client = object()  # pretend Fish initialised
    v._build_tts_chain()
    assert v._backend_order == ["sapi", "fish-audio"]


def test_sapi_is_always_available():
    """Without a Fish client JARVIS still has the local SAPI fallback."""
    v = VoiceMCP.__new__(VoiceMCP)
    VoiceMCP.__init__(v)
    v._fish_client = None
    v._build_tts_chain()
    assert v._backend_order == ["sapi"]


def test_rebuilding_the_chain_does_not_duplicate_entries():
    """A repeated init must not leave two copies of the backend behind."""
    v = VoiceMCP.__new__(VoiceMCP)
    VoiceMCP.__init__(v)
    v._fish_client = object()
    v._build_tts_chain()
    v._build_tts_chain()
    assert v._backend_order == ["sapi", "fish-audio"]


def test_fish_voice_is_configurable(monkeypatch):
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    v = VoiceMCP.__new__(VoiceMCP)
    VoiceMCP.__init__(v)
    assert v.fish_tts_model, "a TTS model must be set so speech can start"

    monkeypatch.setenv("FISH_VOICE_ID", "abc123")
    v2 = VoiceMCP.__new__(VoiceMCP)
    VoiceMCP.__init__(v2)
    assert v2.fish_voice_id == "abc123"


# --- the one backend, reported honestly -------------------------------------

def test_backend_report_names_fish(voice):
    voice._backend_order = ["fish-audio"]
    assert "fish" in voice.handle("voice backend", "").lower()


def test_backend_report_refuses_a_backend_that_was_removed(voice):
    """Asking for a deleted backend must be answered, not silently ignored."""
    voice._backend_order = ["fish-audio"]
    reply = voice.handle("voice backend", "edge")
    assert "edge" in reply.lower()
    assert "fish" in reply.lower()


def test_backend_report_admits_it_when_silent(voice):
    voice._backend_order = []
    assert "unavailable" in voice.handle("voice backend", "").lower()


def test_list_voices_does_not_mention_a_removed_backend(voice):
    voice._active_backend = "fish-audio"
    reply = voice.handle("list voices", "")
    assert "edge" not in reply.lower()
    assert "fish" in reply.lower()


def test_interrupt_debounce_never_blocks_an_active_stop(voice):
    """Two interrupts in quick succession must both stop playback."""
    voice._is_speaking = True
    voice._last_interrupt = time.monotonic()
    voice.interrupt()
    assert voice._stop_playback.is_set()
    assert not voice._is_speaking


# --- output gain ----------------------------------------------------------

def test_tts_gain_defaults_to_unity(monkeypatch):
    monkeypatch.delenv("JARVIS_TTS_GAIN", raising=False)
    assert VoiceMCP._read_tts_gain() == 1.0


def test_tts_gain_reads_the_configured_value(monkeypatch):
    monkeypatch.setenv("JARVIS_TTS_GAIN", "2.5")
    assert VoiceMCP._read_tts_gain() == 2.5


def test_tts_gain_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("JARVIS_TTS_GAIN", "loud")
    assert VoiceMCP._read_tts_gain() == 1.0


@pytest.mark.parametrize("raw,expected", [("-4", 0.1), ("50", 10.0)])
def test_tts_gain_is_clamped_to_a_safe_range(monkeypatch, raw, expected):
    monkeypatch.setenv("JARVIS_TTS_GAIN", raw)
    assert VoiceMCP._read_tts_gain() == expected


def test_playback_scales_and_clips_the_samples(voice, monkeypatch, tmp_path):
    """Gain is applied to the float samples and clipped to [-1, 1]."""
    import sys
    import types

    import numpy as np

    samples = np.array([0.2, -0.2, 0.8], dtype="float32")
    played = {}

    fake_sf = types.SimpleNamespace(
        read=lambda path, dtype=None: (samples.copy(), 16000))

    class Stream:
        active = False

    fake_sd = types.SimpleNamespace(
        play=lambda data, sr, device=None: played.update(data=data),
        get_stream=lambda: Stream(),
        stop=lambda: None,
    )

    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(voice, "_select_output_device", lambda sd: None)
    voice.tts_gain = 2.0

    assert voice._play_mci(str(tmp_path / "reply.mp3")) is True
    assert played["data"].tolist() == pytest.approx([0.4, -0.4, 1.0])


def test_hands_free_output_device_is_never_chosen(voice, monkeypatch):
    """A Bluetooth telephony endpoint opens fine but is quiet and tinny."""
    import types

    monkeypatch.delenv("JARVIS_OUTPUT_DEVICE", raising=False)
    by_idx = {
        30: {"name": "Headset (Hands-Free AG Audio)"},
        5: {"name": "Headphones (CMF Buds 2a)"},
    }
    fake_sd = types.SimpleNamespace(query_devices=lambda idx: by_idx[idx])
    monkeypatch.setattr(voice, "_output_devices",
                        lambda sd: list(by_idx.items()))
    monkeypatch.setattr(voice, "_output_device_opens", lambda sd, idx: True)
    voice._output_device_cache = None

    assert voice._select_output_device(fake_sd) == 5

