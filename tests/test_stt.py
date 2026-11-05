"""Speech-recognition pipeline tests.

The energy segmenter and the wake-word matcher are pure functions, so these
run without a microphone, a Vosk model, or an audio device. The listener
lifecycle is exercised with fakes.
"""

from __future__ import annotations

import array
import struct
import sys
import threading
import time

import pytest

from mcp.voice import (
    VAD_BLOCK,
    UtteranceSegmenter,
    VoiceMCP,
    _model_path_to_load,
    strip_wake_word,
)


def tone(amplitude: int, samples: int = VAD_BLOCK) -> bytes:
    """A 16-bit PCM block at a constant amplitude (silence is 0)."""
    return struct.pack("<" + "h" * samples, *([amplitude] * samples))


SILENT = tone(0)
SPEECH = tone(8000)


def feed(seg, block, ms):
    """Push an audio block repeatedly for a duration, returning utterances emitted.

    Tests are written in milliseconds rather than block counts: the block size
    is a detection tuning decision, and a test that counts blocks silently
    changes meaning whenever it is retuned.
    """
    block_ms = seg.block_ms
    out = []
    for _ in range(max(1, ms // max(1, block_ms))):
        emitted = seg.push(block)
        if emitted is not None:
            out.append(emitted)
    return out


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
    assert feed(seg, SPEECH, 600) == [], "still talking"
    assert feed(seg, SILENT, 1200), "pause did not end the utterance"


def test_utterance_contains_the_speech():
    seg = UtteranceSegmenter(silence_ms=500)
    feed(seg, SPEECH, 400)
    out = feed(seg, SILENT, 800)[0]
    assert SPEECH in out


def test_exactly_one_utterance_per_pause():
    """A long silence must not produce a burst of empty utterances."""
    seg = UtteranceSegmenter(silence_ms=500)
    feed(seg, SPEECH, 600)
    assert len(feed(seg, SILENT, 4000)) == 1


def test_two_utterances_are_reported_separately():
    seg = UtteranceSegmenter(silence_ms=500)
    emitted = []
    emitted += feed(seg, SPEECH, 600)
    emitted += feed(seg, SILENT, 1000)
    emitted += feed(seg, SPEECH, 600)
    emitted += feed(seg, SILENT, 1000)
    assert len(emitted) == 2, f"expected two utterances, got {len(emitted)}"


def test_a_click_is_too_short_to_be_an_utterance():
    """min_speech_ms exists so mouse clicks and door slams are ignored."""
    seg = UtteranceSegmenter(silence_ms=500, min_speech_ms=1000)
    feed(seg, SPEECH, 200)
    assert feed(seg, SILENT, 1500) == [], "a short click was treated as speech"


def test_flush_returns_a_pending_utterance():
    seg = UtteranceSegmenter(silence_ms=5000)
    feed(seg, SPEECH, 600)
    out = seg.flush()
    assert out is not None and len(out) > 0


def test_trailing_silence_is_trimmed_from_the_utterance():
    """The pause that ends an utterance is not worth decoding."""
    seg = UtteranceSegmenter(silence_ms=1000, tail_ms=300)
    feed(seg, SPEECH, 600)
    out = feed(seg, SILENT, 1000)[0]
    assert out is not None
    # 600 ms of speech plus at most the 300 ms tail.
    assert len(out) <= (900 * 16000 // 1000) * 2 + 4


def test_the_lead_in_survives_the_trim():
    """Trimming the tail must not clip the syllable that starts the utterance."""
    seg = UtteranceSegmenter(silence_ms=1000, pre_roll_ms=500, tail_ms=300)
    feed(seg, SILENT, 500)
    out = feed(seg, SPEECH, 400)
    out += feed(seg, SILENT, 1500)
    assert out, "the utterance did not form"
    assert SPEECH in out[0]


def test_a_noise_burst_is_not_transcribed():
    """A short loud blip is not speech, and decoding it costs real CPU."""
    seg = UtteranceSegmenter(silence_ms=500, min_speech_ms=1000)
    feed(seg, SPEECH, 200)
    assert feed(seg, SILENT, 1500) == [], "a burst of noise was treated as speech"


# --- the model without its neural rescorer ----------------------------------

def _fake_model(tmp_path):
    """A model folder shaped like the real one, with a rescorer in it."""
    model = tmp_path / "model"
    (model / "am").mkdir(parents=True)
    (model / "graph").mkdir()
    (model / "conf").mkdir()
    (model / "ivector").mkdir()
    (model / "rescore").mkdir()
    (model / "am" / "final.mdl").write_bytes(b"am")
    (model / "graph" / "HCLG.fst").write_bytes(b"fst")
    (model / "rescore" / "rnnlm").write_bytes(b"rnnlm")
    return model


def test_the_model_is_loaded_without_the_neural_rescorer(tmp_path, monkeypatch):
    """The rescorer costs seven times the decoding time and nearly two gigs."""
    monkeypatch.delenv("JARVIS_VOSK_RNNLM", raising=False)
    model = _fake_model(tmp_path)
    chosen = _model_path_to_load(model)
    assert chosen != model, "the rescorer was left in the model"
    assert not (chosen / "rescore").exists()
    assert (chosen / "graph").is_dir() and (chosen / "am").is_dir()


def test_the_rescorer_can_be_asked_for(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_VOSK_RNNLM", "1")
    model = _fake_model(tmp_path)
    assert _model_path_to_load(model) == model


def test_a_model_without_a_rescorer_is_left_alone(tmp_path, monkeypatch):
    """Most Vosk models have no rescorer, and they must still load."""
    monkeypatch.delenv("JARVIS_VOSK_RNNLM", raising=False)
    model = tmp_path / "model"
    (model / "am").mkdir(parents=True)
    (model / "graph").mkdir()
    assert _model_path_to_load(model) == model


def test_flush_is_idempotent():
    seg = UtteranceSegmenter()
    feed(seg, SPEECH, 600)
    assert seg.flush() is not None
    assert seg.flush() is None


def test_long_utterance_is_cut_at_the_cap():
    seg = UtteranceSegmenter(silence_ms=10000, max_utterance_s=1.0)
    # 3 s of continuous speech against a 1 s cap.
    assert feed(seg, SPEECH, 3000), "continuous speech grew unbounded"


def test_capture_block_is_short_enough_to_detect_speech():
    """Regression: a 250 ms block averaged speech below the gate.

    Measured on this machine at 50% input gain: a raised voice peaks at 0.0056
    in a 32 ms frame but only 0.0019 in a 250 ms frame, against a gate of
    0.0013. Six blocks in sixty cleared it, no utterance ever formed, and the
    assistant heard nothing while the threshold itself was correct. Detection
    needs a frame short enough to preserve the peaks.
    """
    from mcp.voice import VAD_BLOCK

    assert VAD_BLOCK * 1000 // 16000 <= 50, (
        "capture blocks must be short for speech detection; a quarter second "
        "averages the voice together with the pauses around it")


def test_measured_speech_clears_the_gate_at_the_capture_block_size():
    """The level measured from hardware, fed at the size the app actually uses."""
    from mcp.voice import VAD_BLOCK

    seg = UtteranceSegmenter(block=VAD_BLOCK, threshold=0.0013, silence_ms=1000,
                             min_speech_ms=300)
    # 0.0019 is what this voice measured per 250 ms block; 0.0056 is the same
    # voice measured per 32 ms frame. Either way the utterance must form.
    for amplitude in (625, 1845):
        seg = UtteranceSegmenter(block=VAD_BLOCK, threshold=0.0013,
                                 silence_ms=1000, min_speech_ms=300)
        out = feed(seg, tone(amplitude), 600)
        out += feed(seg, SILENT, 1200)
        assert out, f"speech at {amplitude / 32768:.5f} rms did not form an utterance"


def test_pre_roll_survives_a_smaller_block():
    """Shrinking the block must not clip the first syllable off.

    The lead-in is specified as a duration. If it were a block count, cutting
    the block from 250 ms to 30 ms would quietly cut the pre-roll from 500 ms to
    60 ms and every utterance would lose its first consonant.
    """
    seg = UtteranceSegmenter(block=480, pre_roll_ms=500)
    assert seg.pre_roll_blocks * seg.block_ms >= 450, (
        f"pre-roll is only {seg.pre_roll_blocks * seg.block_ms} ms, expected ~500")
    feed(seg, SILENT, 500)
    out = feed(seg, SPEECH, 400)
    out += feed(seg, SILENT, 1500)
    if out:
        assert out[0] is not None
        # The lead-in audio must actually be in the emitted utterance.
        assert len(out[0]) > 480 * 2 * 4, "the lead-in was dropped"


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


def test_wake_word_accepts_a_list_of_names():
    """A rename must not lock the user out: both names answer."""
    for name in ("jarvis", "ameena"):
        matched, command = strip_wake_word(f"{name} what time is it", "jarvis,ameena")
        assert matched, f"{name} was not accepted"
        assert command == "what time is it"


def test_wake_word_list_ignores_spaces_and_blanks():
    assert strip_wake_word("jarvis hello", " jarvis , , jarvis ")[0] is True
    # A list of nothing but separators means "no wake word", which is the
    # documented mode where every utterance is treated as a command.
    assert strip_wake_word("hello there", " , , ") == (True, "hello there")


def test_default_wake_word_is_configurable(monkeypatch):
    monkeypatch.setenv("JARVIS_WAKE_WORD", "computer")
    v = VoiceMCP.__new__(VoiceMCP)
    VoiceMCP.__init__(v)
    assert v.wake_word == "computer"
    assert strip_wake_word("computer lock the door", v.wake_word)[1] == "lock the door"


def test_empty_input():
    assert strip_wake_word("", "jarvis") == (False, "")
    # With no wake word configured, everything is treated as a command.
    assert strip_wake_word("do the thing", "") == (True, "do the thing")


# --- listener lifecycle ----------------------------------------------------

@pytest.fixture
def voice():
    v = VoiceMCP.__new__(VoiceMCP)
    VoiceMCP.__init__(v)
    # Pin the wake word so these tests describe behaviour, not the shipped
    # default, which is free to change when the assistant is renamed.
    v.wake_word = "jarvis"
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


def test_finish_utterance_feeds_the_utterance_it_was_given(voice, monkeypatch):
    """When the caller hands over the utterance, that audio must be transcribed."""
    heard = _capture_hook(monkeypatch)
    rec = FakeRecognizer("jarvis what time is it")
    voice._finish_utterance(rec, utterance=SPEECH)
    assert rec.fed == [SPEECH]
    assert heard == ["what time is it"]


def test_room_tone_never_reaches_the_recognizer(voice, monkeypatch):
    """Vosk runs its acoustic model on every frame it is fed.

    Feeding it the room tone costs a third of a CPU core on a machine that is
    only ever silent, so the capture loop must hand it speech and nothing else.
    """
    heard = _capture_hook(monkeypatch)
    rec = FakeRecognizer("jarvis show panels")
    seg = UtteranceSegmenter(sample_rate=16000, block=480, threshold=0.0013,
                             silence_ms=1200, pre_roll_ms=500)
    captured = 0
    for _ in range(300):         # nine seconds of room tone
        block = SILENT
        level = seg.rms(block)
        seg.push(block, level)
        captured += len(block)
    for _ in range(20):          # the wake word and the command
        block = SPEECH
        level = seg.rms(block)
        seg.push(block, level)
        captured += len(block)
    for _ in range(45):          # the pause that ends the utterance
        block = SILENT
        level = seg.rms(block)
        utterance = seg.push(block, level)
        captured += len(block)
        if utterance:
            voice._finish_utterance(rec, utterance=utterance)
            rec.Reset()
    assert rec.fed, "the recognizer never heard the command"
    assert heard == ["show panels"]
    fed = b"".join(rec.fed)
    assert SPEECH in fed, "the command never reached the recognizer"
    # Only the utterance (plus its short lead-in) may reach the recognizer.
    assert len(fed) < captured / 3, (
        "the recognizer was fed %d of %d captured bytes: the room tone leaked in"
        % (len(fed), captured))


def test_segmenter_uses_a_precomputed_level():
    """The capture loop measures a block once; the segmenter must not re-measure."""
    seg = UtteranceSegmenter(sample_rate=16000, block=480, threshold=0.0013)
    assert seg.push(SPEECH, seg.rms(SPEECH)) is None  # still inside the utterance
    assert seg.push(SILENT, 0.0) is None               # measured level honoured
    assert seg.rms(SPEECH) == pytest.approx(UtteranceSegmenter.rms(SPEECH))


def test_rms_matches_a_plain_reference_loop():
    """The array-based RMS must agree with the obvious implementation."""
    def reference(data):
        samples = array.array("h")
        samples.frombytes(data)
        total = sum(s * s for s in samples)
        return (total / len(samples)) ** 0.5 / 32768.0

    for data in (SILENT, SPEECH, tone(2000), tone(16000), b"\x01\x02\x03\x04"):
        assert UtteranceSegmenter.rms(data) == pytest.approx(reference(data))


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


class FakeOutputSoundDevice:
    """A sounddevice stand-in for output selection.

    ``openable`` lists the output indices that behave like a connected
    device. Everything else raises on open, which is what Windows does to a
    Bluetooth headset that is in the device list but not connected.
    """

    def __init__(self, devices, default=None, openable=()):
        self._devices = devices
        self.probed = []

        class _Default:
            def __init__(self, idx):
                self.device = [None, idx]
        self.default = _Default(default)

        self._openable = set(openable)

    def query_devices(self, idx=None):
        if idx is None:
            return list(self._devices)
        return self._devices[idx]

    def OutputStream(self, device=None, **kwargs):
        self.probed.append(device)
        if device not in self._openable:
            raise RuntimeError("Device unavailable")

        class _Stream:
            def write(self, *a, **k):
                pass

            def start(self):
                pass

            def stop(self):
                pass

            def close(self):
                pass
        return _Stream()


def _install_fake_output(monkeypatch, fake):
    _install_sd(monkeypatch, fake)
    return fake


def _install_sd(monkeypatch, fake):
    import types

    mod = types.ModuleType("sounddevice")
    mod.query_devices = fake.query_devices
    if hasattr(fake, "RawInputStream"):
        mod.RawInputStream = fake.RawInputStream
    if hasattr(fake, "OutputStream"):
        mod.OutputStream = fake.OutputStream
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
    # Speech long enough to clear the minimum, then the pause that ends it.
    rec, opened = _run_loop(voice, monkeypatch, [SPEECH] * 12 + [SILENT] * 45)
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


# --- noise-calibrated speech gate -------------------------------------------
#
# Regression: the gate used to be a fixed 0.015, which sits just above the
# ~0.013 a quiet built-in laptop microphone produces. Speech never crossed the
# gate, so no utterance was ever closed and the assistant stayed silent. The
# gate is now derived from the room's own noise floor.

def test_quiet_microphone_level_would_have_been_ignored():
    """The level measured on real hardware must clear the gate."""
    from mcp.voice import VAD_MIN_THRESHOLD, VAD_WARN_THRESHOLD

    # Peak RMS of speech captured from a laptop array, and of room noise.
    quiet_speech = 0.013
    room_noise = 0.0007
    assert quiet_speech > VAD_MIN_THRESHOLD, "a quiet voice must clear the floor"
    assert VAD_MIN_THRESHOLD > room_noise, "silence must stay below the gate"
    # A quiet voice should not be above the point where the user is warned,
    # or every normal machine would be reported as broken.
    assert quiet_speech > VAD_WARN_THRESHOLD, (
        "the warning threshold must sit below a quiet voice, or every normal "
        "machine is reported as too loud"
    )


def _stub_blocks(voice, rms_values):
    """Feed the calibrator a fixed sequence of block RMS values."""
    voice._probe_input_blocks = (
        lambda sd, device, seconds=0.0, skip_speaking=False: list(rms_values)
    )


def test_threshold_follows_the_noise_floor(voice, monkeypatch):
    """A noisy room raises the gate, a silent room lowers it."""
    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.0007})

    _stub_blocks(voice, [0.0007] * 20)
    voice._calibrate_vad(fake, 0)
    quiet_threshold = voice.vad_threshold

    _stub_blocks(voice, [0.002] * 20)
    voice._calibrate_vad(fake, 0)
    loud_threshold = voice.vad_threshold

    assert quiet_threshold < loud_threshold
    assert voice.speaking_threshold > voice.vad_threshold


def test_peak_does_not_raise_the_gate(voice, monkeypatch):
    """Regression: calibration used the peak, so one cough silenced the mic.

    A single loud block during the probe window pushed the derived gate above
    the user's own voice, which made the assistant permanently deaf. The gate
    must follow the room, not the loudest moment in it.
    """
    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.0007})

    _stub_blocks(voice, [0.0007] * 20)
    voice._calibrate_vad(fake, 0)
    baseline = voice.vad_threshold

    # Same quiet room, but a few very loud blocks are mixed into the window.
    _stub_blocks(voice, [0.05] * 3 + [0.0007] * 20)
    voice._calibrate_vad(fake, 0)

    assert voice.vad_threshold == baseline, "one loud block moved the gate"


def test_speech_during_calibration_is_not_mistaken_for_noise(voice, monkeypatch):
    """Regression: the median reported the user's voice as the room.

    The window is sampled while the assistant starts up, which is often while
    the user is speaking. Half the blocks being speech must not raise the
    gate above the voice, or calibration would silence a working microphone.
    """
    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.0007})

    _stub_blocks(voice, [0.0007] * 20)
    voice._calibrate_vad(fake, 0)
    baseline = voice.vad_threshold

    # Half the window is the user talking at 0.004 rms.
    _stub_blocks(voice, [0.0007] * 12 + [0.004] * 12)
    voice._calibrate_vad(fake, 0)

    assert voice._noise_floor < 0.001, "the voice was measured as the room"
    assert voice.vad_threshold <= baseline * 1.5, (
        f"gate rose to {voice.vad_threshold} from {baseline} because the user "
        "spoke during calibration")


def test_jarvis_speaking_is_not_measured_as_room_noise(voice, monkeypatch):
    """The startup greeting must not raise the gate it is calibrated next to.

    A real first run measured 0.0217 of "room noise" while the greeting was
    playing and set the gate to 0.033; a restart in the same room measured
    0.00066. Those loud blocks belong to JARVIS, not the room.
    """
    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)
    reads = {"n": 0}

    class SpeakingStream:
        def __init__(self):
            self.blocks = [0.4] * 3 + [0.0007] * 8

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, count):
            if not self.blocks:
                return memoryview(b""), False
            level = self.blocks.pop(0)
            reads["n"] += 1
            # The first blocks are JARVIS's own greeting.
            voice._is_speaking = reads["n"] <= 3
            return memoryview(tone(int(level * 32767))), False

    class FakeSD:
        def RawInputStream(self, **kwargs):
            return SpeakingStream()

    voice._calibrate_vad(FakeSD(), 0)

    assert voice._noise_floor is not None
    assert voice._noise_floor < 0.001, (
        "calibration measured JARVIS's own speech as the room: "
        f"{voice._noise_floor}")
    assert voice._vad_warning is None
    assert voice.wait_for_calibration(0.1), "calibration never signalled done"


def test_quiet_voice_is_admitted_on_this_hardware(voice, monkeypatch):
    """The real measurement: room 0.0007, speech blocks 0.004.

    This is the level captured from the laptop array, and it is the case the
    old fixed 0.015 gate failed outright.
    """
    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.0007})

    _stub_blocks(voice, [0.0007] * 20)
    voice._calibrate_vad(fake, 0)

    room = 0.0007
    speech_block = 0.004
    assert voice.vad_threshold > room, "room noise would clear the gate"
    assert voice.vad_threshold < speech_block, "a real speech block would not"


def test_threshold_stays_inside_its_bounds(voice, monkeypatch):
    """The gate has a floor but deliberately no ceiling.

    A ceiling is what silenced this user: clamping the gate in a loud room put
    it below the noise, so every breath was read as speech. Now the gate always
    clears the room, and a room too loud to work in is reported instead.
    """
    from mcp.voice import VAD_MIN_THRESHOLD, VAD_NOISE_MULTIPLE

    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)

    silent = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.00001})
    voice._calibrate_vad(silent, 0)
    assert voice.vad_threshold == VAD_MIN_THRESHOLD

    # A tolerable room simply follows the noise.
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.005})
    _stub_blocks(voice, [0.005] * 20)
    voice._calibrate_vad(fake, 0)
    assert voice.vad_threshold == 0.005 * VAD_NOISE_MULTIPLE

    # An intolerable room raises the gate rather than clamping it.
    _stub_blocks(voice, [0.05] * 20)
    voice._calibrate_vad(fake, 0)
    assert voice.vad_threshold == 0.05 * VAD_NOISE_MULTIPLE
    assert voice.vad_threshold > 0.05


def test_gate_admits_speech_above_the_measured_noise(voice, monkeypatch):
    """Whatever the room, the gate must sit above the silence.

    This is the invariant that matters: a gate below the measured noise reads
    every breath as speech, so the assistant talks over itself and never
    finishes an utterance.
    """
    from mcp.voice import VAD_NOISE_MULTIPLE

    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.0007})
    for floor in (0.0, 0.0002, 0.0007, 0.001, 0.003, 0.005, 0.01, 0.05):
        _stub_blocks(voice, [floor] * 20)
        voice._calibrate_vad(fake, 0)
        assert voice.vad_threshold > floor, (
            f"gate {voice.vad_threshold} does not clear a room at {floor}")
        if floor > 0:
            assert voice.vad_threshold >= floor * VAD_NOISE_MULTIPLE, (
                "the gate must hold the minimum margin over the room")


def test_a_usable_room_keeps_speech_within_reach(voice, monkeypatch):
    """The ordinary case: gate above the room, below the voice.

    Only for a room the assistant can actually work in. A loud room is allowed
    to give up sensitivity, but it must never be the normal outcome.
    """
    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.0007})

    room = 0.0007
    speech = 0.013
    _stub_blocks(voice, [room] * 20)
    voice._calibrate_vad(fake, 0)

    assert room < voice.vad_threshold < speech, (
        f"gate {voice.vad_threshold} is not between the room ({room}) "
        f"and the voice ({speech})")


def test_loud_room_raises_the_gate_and_warns(voice, monkeypatch, caplog):
    """When the room outgrows the cap the gate must rise, and say so.

    Capping the gate below the measured noise was the original defect in this
    fix: the assistant then heard a constant stream of noise as speech. Being
    deaf in a loud room is recoverable and self-triggering is not.
    """
    from mcp.voice import VAD_WARN_THRESHOLD

    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.04})
    _stub_blocks(voice, [0.04] * 20)

    with caplog.at_level("WARNING"):
        voice._calibrate_vad(fake, 0)

    assert voice.vad_threshold > 0.04, "the gate sat below the room noise"
    assert voice.vad_threshold > VAD_WARN_THRESHOLD
    assert voice._vad_warning, "a room this loud must be reported to the user"
    assert "too loud" in voice._vad_warning
    assert any("too loud" in r.message for r in caplog.records)


def test_a_quiet_room_warns_about_nothing(voice, monkeypatch):
    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.0007})
    _stub_blocks(voice, [0.0007] * 20)
    voice._calibrate_vad(fake, 0)

    assert voice._vad_warning is None


def test_env_override_wins(voice, monkeypatch):
    monkeypatch.setenv("JARVIS_VAD_THRESHOLD", "0.03")
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.0007})
    voice._calibrate_vad(fake, 0)
    assert voice.vad_threshold == 0.03
    assert voice.speaking_threshold == 0.03 * 2.0


def test_invalid_env_override_is_ignored(voice, monkeypatch):
    monkeypatch.setenv("JARVIS_VAD_THRESHOLD", "loud")
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.0007})
    voice._calibrate_vad(fake, 0)
    assert voice.vad_threshold != 0.0
    assert voice.vad_threshold > 0


def test_unmeasurable_microphone_keeps_the_default(voice, monkeypatch):
    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)
    voice.vad_threshold = 0.011
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.0007},
                           unopenable={0})
    voice._calibrate_vad(fake, 0)
    assert voice.vad_threshold == 0.011
    assert voice._noise_floor is None


def test_health_reports_the_gate(voice):
    """The gate belongs in the status payload, since a silent mic is otherwise invisible."""
    status = voice.get_status()
    assert "vad_threshold" in status
    assert "speaking_threshold" in status
    assert "noise_floor" in status


def test_speaking_gate_is_higher_than_the_idle_gate(voice):
    """Echo rejection depends on this; the assistant must not answer itself."""
    assert voice.speaking_threshold > voice.vad_threshold


def test_barge_in_stays_reachable_for_a_quiet_voice(voice, monkeypatch):
    """A quiet microphone must still be able to interrupt.

    Raising the gate by a large multiple makes the assistant impossible to cut
    off on a laptop array, which is a worse failure than the occasional echo
    getting through: the wake word already prevents echo from causing a
    command.
    """
    from mcp.voice import VAD_SPEAKING_MULTIPLE

    monkeypatch.delenv("JARVIS_VAD_THRESHOLD", raising=False)
    fake = FakeSoundDevice([_dev("Mic")], default=0, levels={0: 0.0007})

    _stub_blocks(voice, [0.0007] * 20)
    voice._calibrate_vad(fake, 0)

    # Peak level of a voice deliberately raised to interrupt.
    raised_voice = 0.05
    assert voice.speaking_threshold < raised_voice, (
        "barge-in gate is too high to interrupt with")
    assert VAD_SPEAKING_MULTIPLE <= 2.5, "an excessive margin blocks interruption"



# -- Output device selection -----------------------------------------------

def test_connected_headset_is_preferred_over_the_laptop(voice, monkeypatch):
    """If the user is wearing earbuds, replies must go to the earbuds."""
    monkeypatch.delenv("JARVIS_OUTPUT_DEVICE", raising=False)
    fake = FakeOutputSoundDevice(
        [_dev("Speakers (Realtek)", out_ch=2), _dev("Headset (Stone 358)", out_ch=2)],
        default=0, openable={0, 1})
    _install_fake_output(monkeypatch, fake)

    assert voice._select_output_device(fake) == 1


def test_disconnected_headset_is_never_used(voice, monkeypatch):
    """Regression: Windows lists headsets the user is not wearing.

    This machine has ten Bluetooth headsets permanently in the device list.
    Choosing by name alone sent speech to hardware nobody was wearing, which
    looks exactly like a broken speaker. A device must also open.
    """
    monkeypatch.delenv("JARVIS_OUTPUT_DEVICE", raising=False)
    fake = FakeOutputSoundDevice(
        [_dev("Speakers (Realtek)", out_ch=2),
         _dev("Headset (Noise Buds VS102 Pro)", out_ch=2),
         _dev("Headset (OPPO Enco Air2i)", out_ch=2)],
        default=0, openable={0})
    _install_fake_output(monkeypatch, fake)

    assert voice._select_output_device(fake) is None, "spoke into a dead headset"


def test_earbuds_are_found_among_stale_ones(voice, monkeypatch):
    """A dead headset earlier in the list must not hide the live one."""
    monkeypatch.delenv("JARVIS_OUTPUT_DEVICE", raising=False)
    fake = FakeOutputSoundDevice(
        [_dev("Speakers (Realtek)", out_ch=2),
         _dev("Headset (Gadpro)", out_ch=2),
         _dev("Headset (Stone 358)", out_ch=2)],
        default=0, openable={0, 2})
    _install_fake_output(monkeypatch, fake)

    assert voice._select_output_device(fake) == 2


def test_no_headset_means_the_system_default(voice, monkeypatch):
    monkeypatch.delenv("JARVIS_OUTPUT_DEVICE", raising=False)
    fake = FakeOutputSoundDevice(
        [_dev("Speakers (Realtek)", out_ch=2), _dev("Headphones ()", out_ch=2)],
        default=0, openable={0})
    _install_fake_output(monkeypatch, fake)

    assert voice._select_output_device(fake) is None


def test_output_device_override_by_name(voice, monkeypatch):
    monkeypatch.setenv("JARVIS_OUTPUT_DEVICE", "stone 358")
    fake = FakeOutputSoundDevice(
        [_dev("Speakers (Realtek)", out_ch=2), _dev("Headset (Stone 358)", out_ch=2)],
        default=0, openable={0, 1})
    _install_fake_output(monkeypatch, fake)

    assert voice._select_output_device(fake) == 1
    assert fake.probed == [], "an explicit override must not be probed"


def test_output_device_override_by_index(voice, monkeypatch):
    monkeypatch.setenv("JARVIS_OUTPUT_DEVICE", "1")
    fake = FakeOutputSoundDevice(
        [_dev("Speakers (Realtek)", out_ch=2), _dev("Speakers (Steam)", out_ch=2)],
        default=0, openable={0})
    _install_fake_output(monkeypatch, fake)

    assert voice._select_output_device(fake) == 1


def test_bad_output_override_falls_back_instead_of_raising(voice, monkeypatch):
    """A typo in the setting must not silence the assistant."""
    monkeypatch.setenv("JARVIS_OUTPUT_DEVICE", "nonexistent-speaker")
    fake = FakeOutputSoundDevice(
        [_dev("Speakers (Realtek)", out_ch=2)], default=0, openable={0})
    _install_fake_output(monkeypatch, fake)

    assert voice._select_output_device(fake) is None


def test_output_choice_is_cached(voice, monkeypatch):
    """Replying must not re-probe the whole device list every time."""
    monkeypatch.delenv("JARVIS_OUTPUT_DEVICE", raising=False)
    fake = FakeOutputSoundDevice(
        [_dev("Speakers (Realtek)", out_ch=2), _dev("Headset (Stone 358)", out_ch=2)],
        default=0, openable={0, 1})
    _install_fake_output(monkeypatch, fake)

    assert voice._select_output_device(fake) == 1
    probed_after_first = len(fake.probed)
    voice._select_output_device(fake)
    voice._select_output_device(fake)
    assert len(fake.probed) == probed_after_first, "device list re-probed"


def test_replies_follow_earbuds_connected_later(voice, monkeypatch):
    """Plugging in earbuds mid-session should be noticed, not require a restart."""
    from mcp.voice import OUTPUT_RECHECK_S

    monkeypatch.delenv("JARVIS_OUTPUT_DEVICE", raising=False)
    fake = FakeOutputSoundDevice(
        [_dev("Speakers (Realtek)", out_ch=2), _dev("Headset (Stone 358)", out_ch=2)],
        default=0, openable={0})
    _install_fake_output(monkeypatch, fake)

    assert voice._select_output_device(fake) is None

    # The earbuds arrive.
    fake._openable.add(1)
    voice._output_device_cached_at -= OUTPUT_RECHECK_S + 1

    assert voice._select_output_device(fake) == 1


def test_quiet_speech_is_lifted_for_the_recognizer(voice):
    """Detection can admit a voice that is still too quiet to recognise.

    On this hardware a raised voice peaks near 0.004 RMS, about -48 dBFS, and
    Vosk returns an empty transcript for that. The gate runs on the raw signal
    while the recognizer is given a lifted copy.
    """
    quiet = 0.004
    voice._adapt_input_gain(quiet)
    for _ in range(200):
        voice._adapt_input_gain(quiet)

    assert voice._input_gain > 1.0
    scaled = UtteranceSegmenter.rms(voice._gain_audio(tone(1310)))
    assert scaled > quiet, "the recognizer is still being handed a whisper"
    assert scaled < 1.0, "the lifted audio clipped"


def test_gain_does_not_clip_loud_audio(voice):
    loud = 0.9
    for _ in range(50):
        voice._adapt_input_gain(loud)
    samples = array.array("h")
    samples.frombytes(voice._gain_audio(tone(29490)))
    assert max(samples) <= 32767 and min(samples) >= -32768
    assert UtteranceSegmenter.rms(voice._gain_audio(tone(29490))) <= 1.0


def test_gain_is_smoothed(voice):
    """One contrasting block must not snap the level, in either direction."""
    from mcp.voice import VAD_AGC_MAX_GAIN

    # A quiet block wants a high gain.
    for _ in range(50):
        voice._adapt_input_gain(0.02)
    high = voice._input_gain
    assert high > 2.0

    # A single loud block wants unity, but must not get there instantly.
    voice._adapt_input_gain(0.9)
    assert high > voice._input_gain > 1.0, "the gain fell straight to unity"

    # And one quiet block must not snap it back up either.
    before = voice._input_gain
    voice._adapt_input_gain(0.002)
    assert before < voice._input_gain < VAD_AGC_MAX_GAIN, "the gain jumped"


def test_gain_never_drops_below_unity(voice):
    """Silence must not be amplified, or the AGC manufactures its own noise."""
    for _ in range(20):
        voice._adapt_input_gain(0.0001)
    assert voice._input_gain >= 1.0


def test_a_quiet_working_microphone_is_not_abandoned(voice, monkeypatch):
    """Turning the input level down must not move JARVIS to another mic.

    The dead-input cutoff is an absolute level, so a cutoff set for a desktop
    microphone declares a laptop array dead as soon as the Windows input level
    is turned down. The assistant then listens on a different device every time
    the user adjusts their volume, which is baffling to debug.
    """
    from mcp.voice import INPUT_SILENT_RMS

    monkeypatch.delenv("JARVIS_INPUT_DEVICE", raising=False)
    quiet_but_working = 0.0003
    assert quiet_but_working > INPUT_SILENT_RMS, (
        "a quiet but working microphone would be treated as dead")

    fake = FakeSoundDevice(
        [_dev("Microphone Array (Realtek)"), _dev("Microsoft Sound Mapper")],
        default=0, levels={0: quiet_but_working, 1: 0.02})

    assert voice._select_input_device(fake) == 0, (
        "abandoned a working default input for a louder one")
