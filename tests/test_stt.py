"""Speech-recognition pipeline tests.

The energy segmenter and the wake-word matcher are pure functions, so these
run without a microphone, a Vosk model, or an audio device. The listener
lifecycle is exercised with fakes.
"""

from __future__ import annotations

import struct
import sys
import threading
import time

import pytest

from mcp.voice import (
    VAD_BLOCK,
    UtteranceSegmenter,
    VoiceMCP,
    strip_wake_word,
)


def tone(amplitude: int, samples: int = VAD_BLOCK) -> bytes:
    """A 16-bit PCM block at a constant amplitude (silence is 0)."""
    return struct.pack("<" + "h" * samples, *([amplitude] * samples))


SILENT = tone(0)
SPEECH = tone(8000)


# --- segmenter -------------------------------------------------------------

def test_rms_of_silence_is_zero():
    assert UtteranceSegmenter.rms(SILENT) == 0.0


def test_rms_scales_with_amplitude():
    quiet = UtteranceSegmenter.rms(tone(2000))
    loud = UtteranceSegmenter.rms(tone(16000))
    assert 0 < quiet < loud <= 1.0


def test_rms_handles_odd_length_and_empty():
    assert UtteranceSegmenter.rms(b"") == 0.0
    assert UtteranceSegmenter.rms(b"\x01") == 0.0


def test_silence_never_produces_an_utterance():
    seg = UtteranceSegmenter()
    for _ in range(40):
        assert seg.push(SILENT) is None
    assert seg.flush() is None


def test_speech_followed_by_a_pause_produces_one_utterance():
    seg = UtteranceSegmenter(silence_ms=1000)
    for _ in range(4):
        assert seg.push(SPEECH) is None  # still talking
    emitted = [seg.push(SILENT) for _ in range(6)]
    assert any(e is not None for e in emitted), "pause did not end the utterance"


def test_utterance_contains_the_speech():
    seg = UtteranceSegmenter(silence_ms=500)
    seg.push(SPEECH)
    seg.push(SPEECH)
    emitted = [seg.push(SILENT) for _ in range(6)]
    out = next(e for e in emitted if e is not None)
    assert SPEECH in out


def test_exactly_one_utterance_per_pause():
    """A long silence must not produce a burst of empty utterances."""
    seg = UtteranceSegmenter(silence_ms=500)
    for _ in range(4):
        seg.push(SPEECH)
    emitted = [seg.push(SILENT) for _ in range(20)]
    assert sum(1 for e in emitted if e is not None) == 1


def test_two_utterances_are_reported_separately():
    seg = UtteranceSegmenter(silence_ms=500)
    emitted = []
    for _ in range(4):
        seg.push(SPEECH)
    for _ in range(6):
        if seg.push(SILENT):
            emitted.append(1)
    for _ in range(4):
        seg.push(SPEECH)
    for _ in range(6):
        if seg.push(SILENT):
            emitted.append(1)
    assert len(emitted) == 2, f"expected two utterances, got {len(emitted)}"


def test_a_click_is_too_short_to_be_an_utterance():
    """min_speech_ms exists so mouse clicks and door slams are ignored."""
    seg = UtteranceSegmenter(silence_ms=500, min_speech_ms=1000)
    seg.push(SPEECH)
    out = None
    for _ in range(6):
        out = seg.push(SILENT)
    assert out is None, "a single 250 ms block was treated as speech"


def test_flush_returns_a_pending_utterance():
    seg = UtteranceSegmenter(silence_ms=5000)
    for _ in range(4):
        seg.push(SPEECH)
    out = seg.flush()
    assert out is not None and len(out) > 0


def test_flush_is_idempotent():
    seg = UtteranceSegmenter()
    for _ in range(4):
        seg.push(SPEECH)
    assert seg.flush() is not None
    assert seg.flush() is None


def test_long_utterance_is_cut_at_the_cap():
    seg = UtteranceSegmenter(silence_ms=10000, max_utterance_s=1.0)
    out = None
    for _ in range(12):  # 3 s of continuous speech, cap is 1 s
        out = seg.push(SPEECH)
    assert out is not None, "continuous speech grew unbounded"


def test_quiet_room_does_not_trigger():
    seg = UtteranceSegmenter(threshold=0.05)
    for _ in range(20):
        assert seg.push(tone(200)) is None  # ~0.006 RMS, well under threshold


def test_custom_block_size_is_honoured():
    seg = UtteranceSegmenter(block=800, silence_ms=400, min_speech_ms=100)
    assert seg.block_ms == 50
    assert seg.silence_blocks == 8


# --- wake word -------------------------------------------------------------

@pytest.mark.parametrize("heard,expected", [
    ("jarvis what time is it", "what time is it"),
    ("jarvis, what time is it", "what time is it"),
    ("hey jarvis what time is it", "what time is it"),
    ("ok jarvis lock the computer", "lock the computer"),
    ("Jarvis! open notepad", "open notepad"),
    ("jarvis please open notepad", "open notepad"),
    ("jarvis can you open notepad", "open notepad"),
])
def test_wake_word_is_stripped(heard, expected):
    matched, command = strip_wake_word(heard, "jarvis")
    assert matched, f"wake word not detected in {heard!r}"
    assert command == expected


@pytest.mark.parametrize("heard", [
    "what time is it",
    "turn on the lights jarvis",
    "jasper open notepad",
])
def test_missing_or_mid_sentence_wake_word_is_not_matched(heard):
    matched, _ = strip_wake_word(heard, "jarvis")
    assert not matched, f"{heard!r} should not count as addressed"


def test_custom_wake_word():
    matched, command = strip_wake_word("computer lock the door", "computer")
    assert matched and command == "lock the door"


def test_empty_input():
    assert strip_wake_word("", "jarvis") == (False, "")
    # With no wake word configured, everything is treated as a command.
    assert strip_wake_word("do the thing", "") == (True, "do the thing")


# --- listener lifecycle ----------------------------------------------------

@pytest.fixture
def voice():
    v = VoiceMCP.__new__(VoiceMCP)
    VoiceMCP.__init__(v)
    return v


def test_start_without_a_model_reports_instead_of_raising(voice):
    voice._vosk_model = None
    assert voice.start_listener() is False
    assert voice.get_listen_error()
    assert voice.is_listening() is False


def test_disable_mic_env_var_prevents_capture(voice, monkeypatch):
    """Headless deployments and the test suite must not open a capture device."""
    monkeypatch.setenv("JARVIS_DISABLE_MIC", "1")
    voice._vosk_model = object()  # would otherwise start
    assert voice.start_listener() is False
    assert "JARVIS_DISABLE_MIC" in voice.get_listen_error()
    assert voice.is_listening() is False


def test_disable_mic_accepts_falsy_values(voice, monkeypatch):
    for value in ("", "0", "false", "False"):
        monkeypatch.setenv("JARVIS_DISABLE_MIC", value)
        assert voice._mic_allowed() is True, f"{value!r} should not disable the mic"


def test_mic_allowed_reflects_the_env_var(voice, monkeypatch):
    monkeypatch.setenv("JARVIS_DISABLE_MIC", "1")
    assert voice._mic_allowed() is False
    monkeypatch.setenv("JARVIS_DISABLE_MIC", "0")
    assert voice._mic_allowed() is True


def test_app_fixture_does_not_leave_the_microphone_open(app_module):
    """Importing the app during tests must not start capturing audio."""
    assert app_module.voice_mcp.is_listening() is False


def test_stop_listener_without_start_is_safe(voice):
    voice.stop_listener()
    assert voice.is_listening() is False


def test_push_to_talk_toggles(voice):
    assert voice.is_push_to_talk() is False
    voice.set_push_to_talk(True)
    assert voice.is_push_to_talk() is True
    voice.set_push_to_talk(False)
    assert voice.is_push_to_talk() is False


def test_releasing_push_to_talk_requests_a_flush(voice):
    voice.set_push_to_talk(True)
    voice._ptt_flush.clear()
    voice.set_push_to_talk(False)
    assert voice._ptt_flush.is_set(), "release did not ask for an immediate transcribe"


def test_status_reports_the_listening_mode(voice):
    status = voice.get_status()
    for key in ("listening", "push_to_talk", "wake_word", "last_transcript", "listen_error"):
        assert key in status, f"missing {key!r}"


def test_status_text_mentions_push_to_talk(voice):
    voice.set_push_to_talk(True)
    try:
        assert "push-to-talk" in voice._get_status_text().lower()
    finally:
        voice.set_push_to_talk(False)


def test_status_text_falls_back_when_no_model(voice):
    assert "typed input" in voice._get_status_text().lower()


# --- dispatch gating -------------------------------------------------------

class FakeRecognizer:
    """Returns a canned transcript, ignoring the audio it is fed."""

    def __init__(self, text):
        self.text = text
        self.fed = []
        self.finals = 0
        self.resets = 0

    def AcceptWaveform(self, data):
        self.fed.append(data)
        return True

    def FinalResult(self):
        self.finals += 1
        return '{"text": "%s"}' % self.text

    def Reset(self):
        self.resets += 1


def _capture_hook(monkeypatch):
    heard = []
    monkeypatch.setattr("mcp.voice._command_hook", lambda text: heard.append(text))
    return heard


def test_finish_utterance_does_not_refeed_audio(voice, monkeypatch):
    """The capture loop already streams audio; re-feeding doubles every command."""
    heard = _capture_hook(monkeypatch)
    rec = FakeRecognizer("jarvis what time is it")
    rec.AcceptWaveform(SPEECH)  # as the listener does, block by block
    voice._finish_utterance(rec)
    assert rec.fed == [SPEECH], "utterance audio was fed to the recognizer twice"
    assert heard == ["what time is it"]


def test_wake_word_utterance_is_dispatched(voice, monkeypatch):
    heard = _capture_hook(monkeypatch)
    voice._finish_utterance(FakeRecognizer("jarvis what time is it"))
    assert heard == ["what time is it"]


def test_utterance_without_wake_word_is_ignored(voice, monkeypatch):
    heard = _capture_hook(monkeypatch)
    voice._finish_utterance(FakeRecognizer("what time is it"))
    assert heard == [], "background speech was treated as a command"
    assert voice.get_last_transcript() == "what time is it"


def test_push_to_talk_dispatches_without_wake_word(voice, monkeypatch):
    heard = _capture_hook(monkeypatch)
    voice.set_push_to_talk(True)
    voice._finish_utterance(FakeRecognizer("what time is it"))
    assert heard == ["what time is it"]


def test_releasing_push_to_talk_still_dispatches(voice, monkeypatch):
    """On release the button is already up, so no wake word may be required."""
    heard = _capture_hook(monkeypatch)
    voice.set_push_to_talk(True)
    voice.set_push_to_talk(False)
    assert voice.is_push_to_talk() is False
    voice._finish_utterance(FakeRecognizer("what time is it"), allow_bare=True)
    assert heard == ["what time is it"]


def test_released_push_to_talk_does_not_gate_later_speech(voice, monkeypatch):
    """The flush must not leave push-to-talk latched on."""
    heard = _capture_hook(monkeypatch)
    voice.set_push_to_talk(True)
    voice.set_push_to_talk(False)
    voice._finish_utterance(FakeRecognizer("what time is it"))
    assert heard == []


def test_wake_word_while_speaking_interrupts(voice, monkeypatch):
    heard = _capture_hook(monkeypatch)
    voice._is_speaking = True
    voice._speech_idle.clear()
    voice._finish_utterance(FakeRecognizer("jarvis stop talking"))
    assert voice._stop_playback.is_set(), "wake word did not cut in over the reply"
    assert heard == ["stop talking"]


def test_wake_word_cooldown_stops_a_second_trigger(voice, monkeypatch):
    """The tail of one interrupting phrase must not fire a second command."""
    heard = _capture_hook(monkeypatch)
    voice._is_speaking = True
    voice._speech_idle.clear()
    voice._finish_utterance(FakeRecognizer("jarvis stop talking"))
    voice._is_speaking = False
    voice._finish_utterance(FakeRecognizer("jarvis and also open notepad"))
    assert heard == ["stop talking"], heard


def test_cooldown_expires(voice, monkeypatch):
    heard = _capture_hook(monkeypatch)
    voice._is_speaking = True
    voice._speech_idle.clear()
    voice._finish_utterance(FakeRecognizer("jarvis stop talking"))
    voice._wake_blocked_until = 0.0
    voice._finish_utterance(FakeRecognizer("jarvis open notepad"))
    assert heard == ["stop talking", "open notepad"]


def test_push_to_talk_interrupts_while_speaking(voice, monkeypatch):
    heard = _capture_hook(monkeypatch)
    voice._is_speaking = True
    voice._speech_idle.clear()
    voice.set_push_to_talk(True)
    voice._finish_utterance(FakeRecognizer("stop talking"))
    assert voice._stop_playback.is_set()
    assert heard == ["stop talking"]


def test_echo_while_speaking_needs_the_wake_word(voice, monkeypatch):
    """JARVIS' own output must never trigger a command by itself."""
    heard = _capture_hook(monkeypatch)
    voice._is_speaking = True
    voice._finish_utterance(FakeRecognizer("it is half past three in the afternoon"))
    assert heard == []


def test_wake_word_only_is_too_short_to_dispatch(voice, monkeypatch):
    heard = _capture_hook(monkeypatch)
    voice._finish_utterance(FakeRecognizer("jarvis"))
    assert heard == [], "'jarvis' alone should not run a command"


def test_empty_transcript_is_ignored(voice, monkeypatch):
    heard = _capture_hook(monkeypatch)
    voice._finish_utterance(FakeRecognizer(""))
    assert heard == []


def test_recogniser_exception_does_not_propagate(voice, monkeypatch):
    heard = _capture_hook(monkeypatch)

    class Broken:
        def AcceptWaveform(self, data):
            raise RuntimeError("decoder died")

        def FinalResult(self):
            return "{}"

    voice._finish_utterance(Broken())  # must not raise
    assert heard == []


def test_dispatch_without_a_hook_only_warns(voice, monkeypatch):
    monkeypatch.setattr("mcp.voice._command_hook", None)
    voice._finish_utterance(FakeRecognizer("jarvis lock the computer"))


def test_wake_word_barges_in_on_tts(voice, monkeypatch):
    _capture_hook(monkeypatch)
    voice._is_speaking = True
    voice._speech_idle.clear()
    voice._finish_utterance(FakeRecognizer("jarvis stop"))
    assert voice._stop_playback.is_set(), "wake word did not interrupt playback"


def test_heard_event_is_broadcast(voice, monkeypatch):
    events = []
    monkeypatch.setattr("mcp.voice._broadcast_hook", lambda m: events.append(m))
    monkeypatch.setattr("mcp.voice._command_hook", lambda text: None)
    voice._finish_utterance(FakeRecognizer("jarvis hello"))
    assert {"type": "heard", "text": "hello"} in events


def test_voice_module_does_not_import_the_app_for_stt():
    """The command hook must keep mcp.voice independent of the FastAPI app."""
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
    assert out.stdout.strip().endswith("False")


# --- capture loop ----------------------------------------------------------

class BufferOnlyRecognizer:
    """Mimics Vosk, which refuses anything that is not a real ``bytes``.

    sounddevice 0.5.x returns a cffi buffer from ``RawInputStream.read()``, and
    passing that straight to Vosk raises, which silently killed the whole
    listener.
    """

    def __init__(self):
        self.seen = []

    def AcceptWaveform(self, data):
        if not isinstance(data, bytes):
            raise TypeError(f"initializer for ctype 'char *' must be a cdata pointer, "
                            f"not {type(data).__name__}")
        self.seen.append(data)

    def FinalResult(self):
        return '{"text": ""}'

    def SetWords(self, flag):
        pass

    def Reset(self):
        pass


class FakeStream:
    """RawInputStream stand-in that yields memoryviews, like sounddevice 0.5.x."""

    def __init__(self, blocks, kwargs=None, on_read=None):
        self._blocks = list(blocks)
        self._on_read = on_read
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, count):
        if self._on_read is not None:
            self._on_read(self.kwargs)
        if self._blocks:
            # A memoryview, not bytes: cffi returns the same kind of object.
            return memoryview(self._blocks.pop(0)), False
        return memoryview(b""), False


class FakeSoundDevice:
    """A stand-in for sounddevice that models devices, levels and failures.

    ``levels`` maps a device index to the RMS that device produces;
    ``unopenable`` lists indices that raise, the way a device in use does.
    """

    def __init__(self, devices, default=None, levels=None, unopenable=()):
        self._devices = devices
        self._default = default
        self.levels = levels or {}
        self.unopenable = set(unopenable)
        self.opened = []

        class _Default:
            def __init__(self, idx):
                self.device = [idx, None]
        self.default = _Default(default)

    def query_devices(self, idx=None):
        if idx is None:
            return list(self._devices)
        return self._devices[idx]

    def RawInputStream(self, **kwargs):
        device = kwargs.get("device")
        if device is not None and device in self.unopenable:
            raise OSError(f"device {device} is busy")
        self.opened.append(dict(kwargs))
        level = self.levels.get(device if device is not None else self._default, 0.0)
        block = tone(int(level * 32767)) if level else SILENT
        return FakeStream([block] * 4, kwargs=kwargs)


def _dev(name, in_ch=1, out_ch=0):
    return {"name": name, "max_input_channels": in_ch, "max_output_channels": out_ch}


def _install_sd(monkeypatch, fake):
    import types

    mod = types.ModuleType("sounddevice")
    mod.query_devices = fake.query_devices
    mod.RawInputStream = fake.RawInputStream
    mod.default = fake.default
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return mod


def _run_loop(voice, monkeypatch, blocks, fake=None):
    """Drive _listen_loop for a few iterations with a fake device."""
    rec = BufferOnlyRecognizer()
    if fake is None:
        fake = FakeSoundDevice([_dev("Fake Mic")], default=0, levels={0: 0.2})
    _install_sd(monkeypatch, fake)
    played = []

    def raw_input_stream(**kwargs):
        played.append(kwargs)
        return FakeStream(blocks, kwargs=kwargs)

    monkeypatch.setattr("mcp.voice.KaldiRecognizer", lambda *a, **k: rec, raising=False)
    monkeypatch.setattr("mcp.voice.HAS_VOSK", True)
    monkeypatch.setattr(voice, "_vosk_model", object())
    sys.modules["sounddevice"].RawInputStream = raw_input_stream

    def stop_after_a_while():
        time.sleep(0.4)
        voice._listener_stop.set()

    threading.Thread(target=stop_after_a_while, daemon=True).start()
    voice._listen_loop()
    return rec, played


def test_capture_loop_converts_buffers_to_bytes(voice, monkeypatch):
    """Regression: a cffi buffer from sounddevice used to break every block."""
    rec, opened = _run_loop(voice, monkeypatch, [SPEECH] * 6)
    assert rec.seen, "nothing reached the recogniser"
    assert all(isinstance(d, bytes) for d in rec.seen)
    assert b"".join(rec.seen).startswith(SPEECH)
    # The capture stream must be opened in the format Vosk expects.
    assert opened, "the capture stream was never opened"
    kwargs = opened[0]
    assert kwargs["samplerate"] == 16000
    assert kwargs["blocksize"] == VAD_BLOCK
    assert kwargs["dtype"] == "int16"
    assert kwargs["channels"] == 1


def test_capture_loop_survives_a_device_that_returns_nothing(voice, monkeypatch):
    rec, _opened = _run_loop(voice, monkeypatch, [])
    assert rec.seen == []


# --- microphone selection --------------------------------------------------

def test_resolve_device_by_index(voice):
    fake = FakeSoundDevice([_dev("A"), _dev("B")], default=0)
    assert voice._resolve_input_device(fake, "1") == 1


def test_resolve_device_by_name_is_case_insensitive(voice):
    fake = FakeSoundDevice([_dev("Headset (Gadpro)"), _dev("OPPO Enco")], default=0)
    assert voice._resolve_input_device(fake, "oppo") == 1
    assert voice._resolve_input_device(fake, "HEADSET") == 0


def test_resolve_device_reports_an_unknown_name(voice):
    fake = FakeSoundDevice([_dev("A")], default=0)
    with pytest.raises(ValueError, match="No input device matches"):
        voice._resolve_input_device(fake, "nope")


def test_resolve_device_reports_an_out_of_range_index(voice):
    fake = FakeSoundDevice([_dev("A")], default=0)
    with pytest.raises(ValueError, match="does not exist"):
        voice._resolve_input_device(fake, "7")


def test_override_by_index_wins(voice, monkeypatch):
    monkeypatch.setenv("JARVIS_INPUT_DEVICE", "2")
    fake = FakeSoundDevice([_dev("A"), _dev("B"), _dev("Wanted")],
                           default=0, levels={0: 0.5, 2: 0.4})
    assert voice._select_input_device(fake) == 2


def test_override_by_name_wins(voice, monkeypatch):
    monkeypatch.setenv("JARVIS_INPUT_DEVICE", "wanted")
    fake = FakeSoundDevice([_dev("A"), _dev("Wanted Mic")], default=0,
                           levels={0: 0.5, 1: 0.4})
    assert voice._select_input_device(fake) == 1


def test_bad_override_is_an_error_not_a_silent_fallback(voice, monkeypatch):
    """Listening on a different mic than asked for is worse than not listening."""
    monkeypatch.setenv("JARVIS_INPUT_DEVICE", "does-not-exist")
    fake = FakeSoundDevice([_dev("A")], default=0, levels={0: 0.5})
    with pytest.raises(ValueError, match="No input device matches"):
        voice._select_input_device(fake)


def test_listen_loop_reports_a_bad_override_and_stops(voice, monkeypatch):
    monkeypatch.setenv("JARVIS_INPUT_DEVICE", "does-not-exist")
    fake = FakeSoundDevice([_dev("A")], default=0, levels={0: 0.5})
    _run_loop(voice, monkeypatch, [SPEECH] * 4, fake=fake)
    assert "No input device matches" in voice.get_listen_error()


def test_working_default_is_kept(voice, monkeypatch):
    monkeypatch.delenv("JARVIS_INPUT_DEVICE", raising=False)
    fake = FakeSoundDevice([_dev("Silent"), _dev("Loud")], default=0,
                           levels={0: 0.4, 1: 0.01})
    assert voice._select_input_device(fake) == 0


def test_silent_default_falls_back_to_the_loudest_input(voice, monkeypatch):
    monkeypatch.delenv("JARVIS_INPUT_DEVICE", raising=False)
    fake = FakeSoundDevice(
        [_dev("Dead Default"), _dev("Quiet"), _dev("Real Mic"), _dev("Louder")],
        default=0, levels={0: 0.0, 1: 0.005, 2: 0.3, 3: 0.4})
    assert voice._select_input_device(fake) == 3


def test_unopenable_default_falls_back(voice, monkeypatch):
    monkeypatch.delenv("JARVIS_INPUT_DEVICE", raising=False)
    fake = FakeSoundDevice([_dev("Busy"), _dev("Good")], default=0,
                           levels={1: 0.3}, unopenable={0})
    assert voice._select_input_device(fake) == 1


def test_no_working_input_falls_back_to_the_system_default(voice, monkeypatch):
    """Rather than stay silent, let PortAudio make its own choice."""
    monkeypatch.delenv("JARVIS_INPUT_DEVICE", raising=False)
    fake = FakeSoundDevice([_dev("A"), _dev("B")], default=0,
                           levels={0: 0.0, 1: 0.0}, unopenable={0, 1})
    assert voice._select_input_device(fake) is None


def test_probe_stops_after_max_devices(voice, monkeypatch):
    """Probing every input on a busy machine must not stall startup."""
    monkeypatch.delenv("JARVIS_INPUT_DEVICE", raising=False)
    monkeypatch.setattr("mcp.voice.MAX_PROBE_DEVICES", 3)
    fake = FakeSoundDevice([_dev(f"Mic {i}") for i in range(10)], default=None,
                           levels={i: 0.0 for i in range(10)})
    voice._select_input_device(fake)
    assert len(fake.opened) == 3


def test_query_devices_failure_does_not_crash(voice, monkeypatch):
    monkeypatch.delenv("JARVIS_INPUT_DEVICE", raising=False)

    class Broken:
        def query_devices(self, idx=None):
            raise OSError("no PortAudio")

        def RawInputStream(self, **kwargs):
            raise OSError("no PortAudio")

        class default:
            device = [None, None]

    assert voice._select_input_device(Broken()) is None
