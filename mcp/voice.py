"""Voice MCP - Speech-to-Text, Text-to-Speech, and Wake Word Detection."""

from __future__ import annotations
import array
import ctypes
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from operator import mul
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.voice")

# Minimum gap between two queue-drains, so a burst of interrupts costs one
# pass. Stopping playback is never debounced.
INTERRUPT_DEBOUNCE_S = 0.25

# Optional hook the host application installs so this module can push events
# to connected clients without importing the whole app (which would boot every
# other MCP as a side effect).
_broadcast_hook = None

# Optional hook the host installs to receive recognised speech. Keeping this
# one-way is what stops mcp.voice from importing the app.
_command_hook = None


def set_broadcast_hook(fn) -> None:
    """Register a callable ``fn(message_dict)`` for outbound events."""
    global _broadcast_hook
    _broadcast_hook = fn


def set_command_hook(fn) -> None:
    """Register a callable ``fn(text)`` that receives recognised commands."""
    global _command_hook
    _command_hook = fn


# --- speech detection ------------------------------------------------------

# Samples per capture block. 480 samples at 16 kHz is 30 ms, which is the
# frame size speech detectors are built around and Vosk's own internal frame.
# This used to be 4000 samples (250 ms), chosen to avoid flooding Vosk with
# small frames. That was a mistake for detection specifically: a 250 ms RMS
# averages a syllable together with the pauses around it, so on this machine
# speech measured 0.0056 per 32 ms frame but only 0.0019 per 250 ms frame,
# against a 0.0013 gate. Only 6 blocks in 60 cleared the gate, so no utterance
# ever formed and the assistant heard nothing despite a correct threshold.
# Averaging over a quarter second destroys exactly the peak-to-RMS relationship
# a speech detector depends on. Vosk is given 30 ms frames and reassembles them.
VAD_BLOCK = 480

# Normalised RMS (0.0 - 1.0) above which a block counts as speech. Roughly
# -36 dBFS: quiet enough for a normal desk microphone in a quiet room.
VAD_THRESHOLD = 0.015

# A fixed threshold cannot serve every machine, so the effective gate is derived
# from the measured room noise. The multiple is deliberately small. The gate has
# to clear the room or it reads every breath as speech, but on a weak preamp the
# voice does not rise far above the room: the built-in array peaks at 0.013 for
# a raised voice and 0.049 at the microphone's own hiss, so there is very little
# headroom to spend. At 5x the noise the assistant was deaf rather than
# sensitive, which is the opposite of the intended fix. 1.5x rejects silence
# while still admitting a quiet voice.
#
# The derived gate is not capped. Holding it down to a ceiling in a loud room
# would put it under the noise and break it; it rises instead, and
# VAD_WARN_THRESHOLD reports that as a warning.
VAD_NOISE_MULTIPLE = 2.5
VAD_MIN_THRESHOLD = 0.0001
# This only decides when to warn. Past this point a quiet voice is unlikely to
# be detected, and the user deserves to be told rather than left debugging it.
VAD_WARN_THRESHOLD = 0.01

# Fraction of the calibration window taken as the room's noise level. A low
# percentile is used rather than the median or the mean because the window
# often contains the user speaking: the quietest blocks are the room, and
# anything louder is speech. The median would report the voice as the noise.
VAD_NOISE_PERCENTILE = 0.25
# Long enough to sample a solid number of quiet blocks, which is what reading a
# percentile requires. A 0.6s window held two or three blocks, and even 1.5s
# held only six at the old 250ms block size, so a single loud moment could land
# on the percentile being read and the gate came out anywhere between 0.002 and
# 0.044. At 30ms blocks, 2.5s is about 83 frames.
VAD_CALIBRATE_S = 2.5
# Extra time the calibrator may take when JARVIS is speaking during the probe.
# Its own voice is not the room, so those blocks are skipped and the window is
# extended rather than letting the greeting raise the gate for the session.
VAD_SPEAKING_SKIP_S = 8.0

# While the assistant is speaking, its own output leaks into the microphone, so
# the gate is raised to keep it from answering itself. The margin is a modest
# 6 dB: the wake word already gates dispatch, so echo cannot cause a command,
# and a larger multiplier would make interrupting impossible on a quiet
# microphone.
VAD_SPEAKING_MULTIPLE = 2.0

# Audio handed to the recognizer is lifted toward this level. The gate can sit
# low enough to admit a voice that is still far too quiet to recognise, and
# Vosk returns an empty transcript for audio at that level.
VAD_AGC_TARGET = 0.15
VAD_AGC_MAX_GAIN = 40.0
# Weight kept on the previous gain, so the level does not jump between blocks.
# Lower = faster adaptation to a suddenly quiet or loud voice.
VAD_AGC_SMOOTHING = 0.5

# Microphone selection. A machine with headsets, speakers and virtual audio
# inputs often has a Windows default input that carries no signal at all, so
# the default is probed and the loudest input wins if it is dead.
INPUT_PROBE_S = 0.4
# Only a genuinely dead input is skipped, not merely a quiet one. This is an
# absolute level, so it has to sit well below a quiet working microphone:
# at 0.001 the built-in array was declared dead purely because the Windows
# input level had been turned down, and JARVIS silently moved to a different
# microphone. The speech gate is calibrated against whatever device is chosen,
# so a quiet microphone needs no help picking a louder one.
INPUT_SILENT_RMS = 0.0001
# Probing opens each device in turn, so bound how many are tried; 0.4s each on
# a machine with 40 inputs would otherwise stall startup for a quarter minute.
MAX_PROBE_DEVICES = 12

# Where replies are played. Windows keeps a Bluetooth headset in the device
# list long after it disconnects, so a name match alone is not enough to prove
# the user is wearing it. These hints identify candidate *personal* outputs;
# a candidate is only used if it also opens successfully.
OUTPUT_HEADSET_HINTS = ("headphone", "headset", "earphone", "earbud",
                        "airpod", "buds")
# Bluetooth telephony endpoints open successfully but play at call quality and
# low level, which reads as "the speaker is broken". They are never chosen.
OUTPUT_HANDSFREE_HINTS = ("hands-free", "hands free", "hf audio", "ag audio")
# Bounds for JARVIS_TTS_GAIN: 0.1 could silence the assistant and above 10x a
# stray typo distorts every reply.
TTS_GAIN_MIN = 0.1
TTS_GAIN_MAX = 10.0
# A stream this short is enough to make a dead endpoint fail and costs nothing
# on a live one.
OUTPUT_PROBE_SR = 22050
OUTPUT_PROBE_BLOCKSIZE = 1024
# How long a chosen output stays valid. Long enough that answering a question
# does not re-probe, short enough that plugging in earbuds is picked up while
# the assistant is idle.
OUTPUT_RECHECK_S = 10.0


class UtteranceSegmenter:
    """Split a stream of 16-bit PCM blocks into utterances using energy alone.

    This deliberately avoids a WebRTC VAD dependency: it holds no audio state
    beyond a short pre-roll buffer, is deterministic, and is unit-testable
    without a microphone.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block: int = VAD_BLOCK,
        threshold: float = VAD_THRESHOLD,
        silence_ms: int = 800,
        min_speech_ms: int = 200,
        max_utterance_s: float = 12.0,
        pre_roll_ms: int = 500,
        tail_ms: int = 300,
    ):
        self.sample_rate = sample_rate
        self.block = block
        self.threshold = threshold
        self.block_ms = int(block * 1000 / sample_rate)
        self.silence_blocks = max(1, silence_ms // max(1, self.block_ms))
        self.min_speech_blocks = max(1, min_speech_ms // max(1, self.block_ms))
        self.max_blocks = max(1, int(max_utterance_s * 1000 / max(1, self.block_ms)))
        # Pre-roll is kept as a duration, not a block count, so that shrinking
        # the block for detection does not also shrink the lead-in and clip the
        # first syllable off every utterance.
        self.pre_roll_blocks = max(1, pre_roll_ms // max(1, self.block_ms))

        self._buffer = bytearray()
        self._pre_roll: List[bytes] = []
        self._quiet_run = 0
        self._in_speech = False
        # How many blocks of this utterance actually cleared the gate, and
        # where the last of them ended in the buffer.
        self._above_blocks = 0
        self._speech_end = 0
        self.tail_ms = tail_ms

    @staticmethod
    def rms(data: bytes) -> float:
        """Root-mean-square amplitude of a 16-bit PCM block, normalised."""
        if not data:
            return 0.0
        count = len(data) // 2
        if count == 0:
            return 0.0
        samples = array.array("h")
        samples.frombytes(data[: count * 2])
        if sys.byteorder == "big":
            samples.byteswap()
        if not samples:
            return 0.0
        total = sum(map(mul, samples, samples))
        return (total / len(samples)) ** 0.5 / 32768.0

    def _speech_blocks(self) -> int:
        return len(self._buffer) // (self.block * 2) if self.block else 0

    def _block_count(self, utterance: bytes) -> int:
        return len(utterance) // (self.block * 2) if self.block else 0

    def push(self, data: bytes, level: Optional[float] = None) -> Optional[bytes]:
        """Feed one block. Returns a complete utterance, or None.

        ``level`` is the block's RMS, when the caller has already computed it.
        Pass it in rather than paying for the same measurement twice.
        """
        if level is None:
            level = self.rms(data)
        if level >= self.threshold:
            if not self._in_speech:
                # Start on speech, but keep the preceding quiet blocks so the
                # first syllable is not clipped off.
                self._in_speech = True
                self._buffer = bytearray(b"".join(self._pre_roll))
                self._pre_roll.clear()
                self._quiet_run = 0
                self._above_blocks = 0
            self._buffer += data
            self._quiet_run = 0
            self._above_blocks += 1
            self._speech_end = len(self._buffer)
        elif self._in_speech:
            self._buffer += data
            self._quiet_run += 1
        else:
            self._pre_roll.append(data)
            if len(self._pre_roll) > self.pre_roll_blocks:
                self._pre_roll.pop(0)

        if not self._in_speech:
            return None

        # End on a long enough pause, or before the buffer grows unbounded.
        if self._quiet_run >= self.silence_blocks or self._speech_blocks() >= self.max_blocks:
            return self.flush()
        return None

    def flush(self) -> Optional[bytes]:
        """Close the current utterance, discarding anything too short to be speech."""
        if not self._in_speech or not self._buffer:
            self._in_speech = False
            self._buffer = bytearray()
            self._pre_roll.clear()
            self._quiet_run = 0
            self._above_blocks = 0
            self._speech_end = 0
            return None

        # The silence that ended the utterance is of no use to the recognizer,
        # and every second of it is a second of decoding. Keep a short tail so
        # a word that trails the last loud block still has its ending.
        tail = self.tail_ms * self.sample_rate // 1000 * 2
        end = min(len(self._buffer), self._speech_end + tail)
        utterance = bytes(self._buffer[:end])
        above_blocks = self._above_blocks

        self._in_speech = False
        self._buffer = bytearray()
        self._pre_roll.clear()
        self._quiet_run = 0
        self._above_blocks = 0
        self._speech_end = 0

        # Only the loud part counts. A click, a door, or a passing fan that
        # cleared the gate is not speech, and transcribing one would cost the
        # decoder a full pass over the audio for nothing.
        if above_blocks < self.min_speech_blocks:
            return None
        return utterance

    def _speech_blocks_len(self, utterance: bytes) -> int:
        return self._block_count(utterance)


def _link_directory(target: Path, link: Path) -> bool:
    """Make ``link`` point at ``target`` without copying anything."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError, AttributeError, ValueError):
        pass
    if os.name == "nt":
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                check=True, capture_output=True,
            )
            return True
        except Exception:
            return False
    return False


def _model_path_to_load(model_path: Path) -> Path:
    """The Vosk model to load, with or without its neural rescorer.

    The callcenter model ships a recurrent language model in ``rescore/`` that
    re-ranks the decoder's hypotheses after every utterance. Measured on this
    machine it makes a decode of five seconds of speech take 6.4 seconds
    instead of 0.8, and it holds 1.9 GB of the machine's memory open. What it
    buys is spelling: the same sentence comes back with the last word
    correctly pluralised. For an assistant that listens for commands that is
    not a trade worth making on a laptop.

    So the model is loaded through a view of the folder that leaves the
    rescorer out, built out of links so nothing is copied and nothing in the
    model is touched. ``JARVIS_VOSK_RNNLM=1`` puts the rescorer back.
    """
    if (os.environ.get("JARVIS_VOSK_RNNLM") or "").strip() in ("1", "true", "yes", "on"):
        return model_path
    if not (model_path / "rescore").is_dir():
        return model_path

    fast = model_path.parent / (model_path.name + "-fast")
    if (fast / "graph").is_dir() and (fast / "am").is_dir():
        return fast

    try:
        fast.mkdir()
        for entry in model_path.iterdir():
            if entry.name in ("rescore", "rnnlm"):
                continue
            if entry.is_dir():
                if not _link_directory(entry, fast / entry.name):
                    raise OSError("could not link %s" % entry.name)
            else:
                shutil.copy2(entry, fast / entry.name)
    except Exception as e:
        log.warning("Could not build the fast Vosk model view (%s); "
                    "loading the model with its rescorer.", e)
        shutil.rmtree(fast, ignore_errors=True)
        return model_path

    log.info("Vosk model loaded without the neural rescorer (JARVIS_VOSK_RNNLM=1 "
             "to keep it).")
    return fast


def strip_wake_word(text: str, wake_word: str = "jarvis") -> tuple[bool, str]:
    """Split ``text`` into (was the wake word used, remaining command).

    Vosk rarely hears a clean prefix, so common fillers before and after the
    wake word are tolerated: "hey jarvis", "okay jarvis, what time is it".

    ``wake_word`` may be a comma-separated list of accepted names.
    """
    if not text:
        return False, ""
    if not wake_word:
        return True, text.strip()
    names = [w.strip() for w in wake_word.split(",") if w.strip()]
    if not names:
        return True, text.strip()
    alternatives = "|".join(re.escape(n) for n in names)
    pattern = re.compile(
        r"^\s*(?:(?:hey|hi|hello|ok|okay|yo)\s+)?"
        r"(?:" + alternatives + r")\b[\s,.:;!\-]*"
        r"(?:(?:please|can\s+you|could\s+you)\s+)?",
        re.IGNORECASE,
    )
    match = pattern.match(text)
    if not match:
        return False, text.strip()
    return True, text[match.end():].strip()

# Try importing vosk
try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    HAS_VOSK = True
except ImportError:
    HAS_VOSK = False
    log.warning("vosk not installed. Voice input disabled.")

# Try importing Fish Audio SDK
try:
    from fishaudio import FishAudio
    HAS_FISH_AUDIO = True
except ImportError:
    HAS_FISH_AUDIO = False
    log.warning("fish-audio-sdk not installed. Fish Audio TTS disabled.")


class VoiceMCP(MCPBase):
    """Voice input/output MCP module."""

    name = "voice"
    description = "Speech-to-Text, Text-to-Speech, and Wake Word Detection"
    commands = [
        "voice status",
        "voice mute",
        "voice unmute",
        "voice backend",
        "set voice",
        "list voices",
    ]

    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        self.block_ms = 40
        self.channels = 1
        # Comma-separated, so the assistant can answer to more than one name.
        # Leave it blank for always-on listening: every utterance is treated
        # as a command with no wake word required.
        self.wake_word = os.getenv("JARVIS_WAKE_WORD", "")
        self.min_command_len = 3
        self.voice_cmd_silence = 1.5
        self.tts_cooldown = 1.5
        self.vad_threshold = VAD_THRESHOLD
        # Louder bar while JARVIS is talking, to ignore its own echo.
        self.speaking_threshold = VAD_THRESHOLD * VAD_SPEAKING_MULTIPLE
        # After a barge-in, ignore further wake words briefly so the tail of the
        # interrupting phrase does not immediately trigger a second command.
        self.wake_cooldown_s = 0.75

        # Fish Audio TTS Settings
        self.fish_api_key = os.getenv("FISH_API_KEY", "")
        self.fish_voice_id = os.getenv("FISH_VOICE_ID", "")
        self.fish_tts_model = os.getenv("FISH_TTS_MODEL", "s2.1-pro-free")
        self._fish_client = None

        # Speech backend. Fish Audio is the only one, so this is a single-entry
        # list rather than a preference order: see _build_tts_chain for why
        # there is deliberately no fallback.
        self._backend_order: List[str] = []
        self._disabled_backends: set[str] = set()
        self._active_backend: Optional[str] = None

        # State
        self._is_speaking = False
        self._tts_done_at = 0.0
        self._speech_q: queue.Queue[Optional[str]] = queue.Queue()
        self._speech_idle = threading.Event()
        self._speech_idle.set()
        self._muted = False
        self._last_interrupt = 0.0
        self._stop_playback = threading.Event()

        # Speech recognition
        self.vad_threshold = VAD_THRESHOLD
        self.speaking_threshold = VAD_THRESHOLD * VAD_SPEAKING_MULTIPLE
        self._noise_floor: Optional[float] = None
        # Set once the room has been measured, so startup speech can wait for
        # it instead of bleeding into the measurement.
        self._calibrated = threading.Event()
        # Set when calibration found a room it cannot work in, so the HUD can
        # say so instead of the user concluding the assistant is ignoring them.
        self._vad_warning: Optional[str] = None
        # A rolling window of the levels the listener actually receives. "It
        # does not hear me" is not diagnosable without knowing whether the
        # blocks ever crossed the gate, so this is measured rather than guessed.
        self._recent_levels: deque = deque(maxlen=600)
        self._blocks_above_gate = 0
        self._utterances_formed = 0
        self._utterances_formed_at = 0.0
        self._input_gain = 1.0
        # Software amplification for replies. Fish Audio's own level is healthy,
        # but a quiet Windows endpoint (or quiet earbuds) is common enough that
        # a boost the user can dial in beats editing the samples.
        self.tts_gain = self._read_tts_gain()
        # Where replies play. None means "the system default output".
        self._output_device_cache: Optional[int] = None
        self._output_device_cached_at: float = 0.0
        self.silence_timeout = 0.8
        self.max_utterance_s = 12.0
        self._listener_stop = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None
        self._ptt_active = threading.Event()
        self._ptt_flush = threading.Event()
        self._wake_blocked_until = 0.0
        self._last_transcript = ""
        self._last_listen_error = ""

        # Windows MCI
        try:
            self._winmm = ctypes.WinDLL("winmm")
            self._mci_lock = threading.Lock()
            self._mci_counter = 0
            self._active_alias: Optional[str] = None
        except Exception:
            self._winmm = None
            log.warning("Windows MCI not available.")

        # Vosk model
        self._vosk_model = None

    @staticmethod
    def _read_tts_gain() -> float:
        """``JARVIS_TTS_GAIN``: software volume boost applied to every reply.

        ``1.0`` leaves the audio untouched, which is the right default: Fish
        Audio already arrives at a healthy level, and the usual cause of "too
        quiet" is the Windows endpoint volume rather than the samples. Invalid
        input warns and falls back, because a typo must not silence the
        assistant.
        """
        raw = (os.environ.get("JARVIS_TTS_GAIN") or "").strip()
        if not raw:
            return 1.0
        try:
            gain = float(raw)
        except ValueError:
            log.warning("JARVIS_TTS_GAIN=%r is not a number; leaving replies "
                        "unboosted.", raw)
            return 1.0
        if not (TTS_GAIN_MIN <= gain <= TTS_GAIN_MAX):
            clamped = max(TTS_GAIN_MIN, min(TTS_GAIN_MAX, gain))
            log.warning("JARVIS_TTS_GAIN=%.2f is outside %.1f-%.1f; clamped to "
                        "%.2f.", gain, TTS_GAIN_MIN, TTS_GAIN_MAX, clamped)
            return clamped
        return gain

    def initialize(self) -> bool:
        """Initialize voice systems."""
        # Load Vosk model
        model_path = Path(__file__).resolve().parent.parent / "model"
        if HAS_VOSK and model_path.is_dir():
            try:
                SetLogLevel(-1)
                self._vosk_model = Model(str(_model_path_to_load(model_path)))
                log.info("Vosk model loaded.")
            except BaseException as e:
                # KeyboardInterrupt is a BaseException, not Exception: a Ctrl+C
                # during model load must not crash with a broken __del__.
                self._vosk_model = None
                log.warning("Vosk load failed: %s", e)

        # Initialize Fish Audio
        if HAS_FISH_AUDIO and self.fish_api_key:
            try:
                self._fish_client = FishAudio(api_key=self.fish_api_key)
                log.info("Fish Audio client initialized. Model: %s", self.fish_tts_model)
            except Exception as e:
                log.error("Fish Audio init failed: %s", e)
                self._fish_client = None
        elif not HAS_FISH_AUDIO:
            log.warning("fish-audio-sdk not installed.")
        else:
            log.warning("FISH_API_KEY not set.")

        self._build_tts_chain()

        # Start speech worker
        threading.Thread(target=self._speech_worker, daemon=True).start()

        # Open the microphone. A missing model or device is not fatal: every
        # typed command and all of TTS still work.
        self.start_listener()

        self._initialized = True
        log.info("Voice MCP initialized. TTS backends: %d", len(self._backend_order))
        return True

    def _build_tts_chain(self) -> None:
        """Pick the speech backends to use, in order.

        Windows SAPI is first: local, offline, instant, no API key needed.
        Fish Audio is second: cloud, higher quality, needs API credit.
        This way JARVIS always answers immediately, and Fish Audio only
        handles replies when the user wants premium quality.
        """
        self._backend_order = ["sapi"]
        if self._fish_client:
            self._backend_order.append("fish-audio")
        else:
            log.error(
                "Fish Audio unavailable (FISH_API_KEY missing or client failed). "
                "JARVIS will use the local SAPI voice."
            )

    def shutdown(self) -> None:
        """Shutdown voice systems."""
        self.stop_listener()
        self._speech_q.put(None)
        self._initialized = False

    def interrupt(self) -> None:
        """Stop current playback and clear the queue (for barge-in)."""
        now = time.monotonic()
        fresh = now - self._last_interrupt >= INTERRUPT_DEBOUNCE_S
        if fresh:
            self._last_interrupt = now
            while not self._speech_q.empty():
                try:
                    self._speech_q.get_nowait()
                except queue.Empty:
                    break

        # Never debounce an actual stop: a barge-in arriving right after
        # another interrupt must still halt playback immediately.
        if not self._is_speaking:
            return

        self._stop_playback.set()
        self._stop_audio_device()
        self._stop_mci()

        self._is_speaking = False
        self._tts_done_at = time.monotonic()
        self._broadcast_speaking(False)
        log.info("Speech interrupted.")

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle voice commands."""
        if command == "voice status":
            return self._get_status_text()
        elif command == "voice mute":
            self._muted = True
            return "Voice output muted, sir."
        elif command == "voice unmute":
            self._muted = False
            return "Voice output enabled, sir."
        elif command == "voice backend":
            return self._backend_report(args)
        elif command == "set voice":
            if args:
                self.fish_voice_id = args
                return f"Voice set to {args}, sir."
            return "Please specify a voice name, sir."
        elif command == "list voices":
            return (
                f"Active backend: {self._active_backend or 'none yet'}. "
                f"Fish voice: {self.fish_voice_id or 'default'}."
            )
        return None

    def _backend_report(self, requested: str = "") -> str:
        """Report the speech backend, including a backend that cannot be used.

        There is only one backend, so "switch the backend" is answered rather
        than ignored: the request may name a provider that was deliberately
        removed, and saying nothing would leave the user waiting for a voice
        that is never going to speak.
        """
        if not self._backend_order:
            return (
                "Fish Audio is unavailable, so I have no voice at all. Check "
                "FISH_API_KEY and the API credit on your Fish account."
            )
        name = requested.strip()
        if name and name.lower() not in ("fish", "fish audio", "fish-audio"):
            return (
                f"{name} is not a speech backend here. Fish Audio is the only "
                f"one, and it is the voice you are hearing."
            )
        return "Fish Audio is the only speech backend, and it is active."

    def speak(self, text: str) -> None:
        """Queue text for speech output."""
        if not self._muted and text:
            self._speech_q.put(self._clean_for_speech(text))

    def speak_stream(self, text_generator, on_sentence=None):
        """Stream text chunks to speech - starts speaking before full response.

        Args:
            text_generator: Generator yielding text tokens.
            on_sentence: Optional callback(sentence_text) called when a
                         complete sentence is queued for speech. Use this
                         to broadcast transcript to the UI in real-time.
        """
        if self._muted:
            return
        buffer = ""
        sentence_end = re.compile(r'[.!?;]\s')
        for chunk in text_generator:
            buffer += chunk
            while True:
                match = sentence_end.search(buffer)
                if match:
                    end_pos = match.end()
                    sentence = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:]
                    if sentence:
                        self._speech_q.put(self._clean_for_speech(sentence))
                        if on_sentence:
                            on_sentence(sentence)
                else:
                    break
        if buffer.strip():
            cleaned = self._clean_for_speech(buffer)
            self._speech_q.put(cleaned)
            if on_sentence:
                on_sentence(cleaned)

    def _clean_for_speech(self, text: str) -> str:
        """Remove markdown and special symbols for natural speech."""
        text = re.sub(r'[#*_~`>|]', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _broadcast_speaking(self, speaking: bool) -> None:
        """Broadcast speaking state so the HUD can enable barge-in."""
        self._emit({"type": "speaking", "speaking": speaking})

    def _emit(self, message: Dict[str, Any]) -> None:
        """Send an event to the host application, if one is listening."""
        if _broadcast_hook is None:
            return
        try:
            _broadcast_hook(message)
        except Exception as e:
            log.debug("broadcast failed: %s", e)

    def _speech_worker(self) -> None:
        """Background worker that drains the queue and speaks each item."""
        while True:
            text = self._speech_q.get()
            if text is None:
                break

            # A fresh utterance always starts unsignalled, even if the previous
            # one was aborted mid-synthesis.
            self._stop_playback.clear()
            self._is_speaking = True
            self._speech_idle.clear()
            self._broadcast_speaking(True)
            try:
                self._speak_with_fallback(text)
            except Exception as e:
                log.error("TTS error: %s", e)
            finally:
                self._is_speaking = False
                self._tts_done_at = time.monotonic()
                self._speech_idle.set()
                self._broadcast_speaking(False)
                self._speech_q.task_done()

    def _speak_with_fallback(self, text: str) -> bool:
        """Speak ``text`` using the first backend that works.

        Short replies (confirmations, single sentences) go through the local
        SAPI backend first so they are instant and never wait on the network.
        Longer replies try Fish Audio first for quality, then SAPI as fallback.
        """
        # Short text: local SAPI first (instant, no network). Long text:
        # Fish Audio first (best quality), SAPI as safety net.
        if len(text) <= 120:
            preferred = ["sapi", "fish-audio"]
        else:
            preferred = ["fish-audio", "sapi"]
        ordered = [b for b in preferred if b in self._backend_order]
        ordered += [b for b in self._backend_order if b not in ordered]

        for backend in ordered:
            if backend in self._disabled_backends:
                continue
            if self._stop_playback.is_set():
                return False
            try:
                if self._TTS_BACKENDS[backend](self, text):
                    self._active_backend = backend
                    return True
                if self._stop_playback.is_set():
                    log.debug("Speech interrupted during '%s' backend.", backend)
                    return False
                self._disable_backend(backend, "returned False")
            except Exception as e:
                if self._stop_playback.is_set():
                    log.debug("Speech interrupted during '%s' backend: %s", backend, e)
                    return False
                self._disable_backend(backend, str(e))

        log.error("All TTS backends exhausted. Tried: %s", ordered)
        return False

    def _disable_backend(self, backend: str, reason: str) -> None:
        """Permanently disable a TTS backend for this session."""
        self._disabled_backends.add(backend)
        if self._active_backend == backend:
            self._active_backend = None
        remaining = [b for b in self._backend_order if b not in self._disabled_backends]
        if remaining:
            log.warning("TTS backend '%s' disabled (%s). Falling back to: %s",
                        backend, reason, " -> ".join(remaining))
        else:
            log.error("TTS backend '%s' disabled (%s). No backends left.", backend, reason)

    # -- Speech recognition --------------------------------------------------

    def _mic_allowed(self) -> bool:
        """False when ``JARVIS_DISABLE_MIC`` opts out of capture."""
        return os.environ.get("JARVIS_DISABLE_MIC", "").strip() in ("", "0", "false", "False")

    def start_listener(self) -> bool:
        """Start capturing the microphone in the background.

        Returns False (without raising) when Vosk or an input device is
        unavailable, so a machine with no microphone still runs every other
        feature. Set ``JARVIS_DISABLE_MIC=1`` to stay silent on the mic, which
        is what headless deployments and the test suite want.
        """
        if self._listener_thread and self._listener_thread.is_alive():
            return True
        if not self._mic_allowed():
            self._last_listen_error = "disabled by JARVIS_DISABLE_MIC"
            log.info("Microphone capture disabled by configuration.")
            return False
        if not HAS_VOSK or self._vosk_model is None:
            self._last_listen_error = "no Vosk model"
            log.warning("Speech recognition unavailable: %s", self._last_listen_error)
            return False

        self._listener_stop.clear()
        self._calibrated.clear()
        self._listener_thread = threading.Thread(
            target=self._listen_loop, name="jarvis-stt", daemon=True
        )
        self._listener_thread.start()
        return True

    def wait_for_calibration(self, timeout: float = 15.0) -> bool:
        """Block until the room has been measured, or the timeout expires."""
        return self._calibrated.wait(timeout)

    def stop_listener(self) -> None:
        """Stop the capture thread. Safe to call when it was never started."""
        self._listener_stop.set()
        thread = self._listener_thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        self._listener_thread = None

    def is_listening(self) -> bool:
        """True while the capture thread is running."""
        return bool(self._listener_thread and self._listener_thread.is_alive())

    def set_push_to_talk(self, active: bool) -> None:
        """Hold-to-talk: while active, speech is a command with no wake word.

        Releasing flushes whatever has been said so far, so the user does not
        have to wait for the silence timeout.
        """
        if active:
            self._ptt_active.set()
        else:
            self._ptt_active.clear()
            self._ptt_flush.set()

    def is_push_to_talk(self) -> bool:
        return self._ptt_active.is_set()

    def _wake_blocked(self) -> bool:
        """True while a just-used wake word is still cooling down."""
        return time.monotonic() < self._wake_blocked_until

    # -- microphone selection ----------------------------------------------

    @staticmethod
    def _input_devices(sd) -> List[tuple]:
        """Every device that can record, as (index, info) pairs."""
        try:
            devices = sd.query_devices()
        except Exception as e:
            log.warning("Could not list audio devices: %s", e)
            return []
        return [(i, d) for i, d in enumerate(devices)
                if d.get("max_input_channels", 0) >= 1]

    def _resolve_input_device(self, sd, spec: str) -> int:
        """Turn a JARVIS_INPUT_DEVICE value into a device index.

        The value is either an index or a case-insensitive substring of a
        device name. Raises ValueError with something the user can act on.
        """
        spec = spec.strip()
        if spec.isdigit():
            idx = int(spec)
            if idx < 0 or idx >= len(sd.query_devices()):
                raise ValueError(
                    f"JARVIS_INPUT_DEVICE index {idx} does not exist "
                    f"(there are {len(sd.query_devices())} devices)")
            return idx

        needle = spec.lower()
        for idx, dev in self._input_devices(sd):
            if needle in str(dev.get("name", "")).lower():
                return idx
        raise ValueError(
            f"No input device matches {spec!r}. List them with: "
            f"python -c \"import sounddevice as sd; print(sd.query_devices())\"")

    def _probe_input_rms(self, sd, device) -> Optional[float]:
        """Peak RMS over a short window, or None if the device will not open.

        Probing must never take the assistant down, so every failure mode of
        PortAudio collapses to None here.
        """
        try:
            with sd.RawInputStream(
                device=device,
                samplerate=self.sample_rate,
                blocksize=VAD_BLOCK,
                dtype="int16",
                channels=self.channels,
            ) as stream:
                peak = 0.0
                deadline = time.monotonic() + INPUT_PROBE_S
                while time.monotonic() < deadline:
                    data, _overflow = stream.read(VAD_BLOCK)
                    if not data:
                        break
                    peak = max(peak, UtteranceSegmenter.rms(bytes(data)))
                return peak
        except Exception:
            return None

    def _select_input_device(self, sd):
        """Choose the microphone to capture from, or None for the system default.

        ``JARVIS_INPUT_DEVICE`` always wins. Without it, the Windows default is
        used if it actually carries signal; otherwise the loudest input does,
        because the default is very often a silent headset or virtual device.

        Raises ValueError for an unusable override: silently listening on a
        different microphone than the one that was asked for is worse than not
        listening at all.
        """
        override = (os.environ.get("JARVIS_INPUT_DEVICE") or "").strip()
        if override:
            idx = self._resolve_input_device(sd, override)
            name = sd.query_devices(idx).get("name", "?")
            peak = self._probe_input_rms(sd, idx)
            log.info("Using JARVIS_INPUT_DEVICE [%d]: %s", idx, name)
            if peak is None:
                log.warning("The configured microphone would not open; trying anyway.")
            elif peak < INPUT_SILENT_RMS:
                log.warning(
                    "The configured microphone looks silent (rms=%.5f). Check the "
                    "Windows input level, or set JARVIS_INPUT_DEVICE to another "
                    "device.", peak)
            else:
                log.info("Microphone probe OK (rms=%.5f).", peak)
            return idx

        default = None
        try:
            default = sd.default.device[0]
        except Exception as e:
            log.warning("Could not read the default input device: %s", e)

        if default is not None and default >= 0:
            name = sd.query_devices(default).get("name", "?")
            peak = self._probe_input_rms(sd, default)
            if peak is not None and peak >= INPUT_SILENT_RMS:
                log.info("Using the default microphone [%d]: %s (rms=%.5f)",
                         default, name, peak)
                return default
            log.warning(
                "The default microphone [%d] %s is silent or unavailable "
                "(rms=%s); looking for a better input...",
                default, name, f"{peak:.5f}" if peak is not None else "could not open")

        best_idx = None
        best_peak = -1.0
        for idx, _dev in self._input_devices(sd)[:MAX_PROBE_DEVICES]:
            if default is not None and idx == default:
                continue
            peak = self._probe_input_rms(sd, idx)
            if peak is not None and peak > best_peak:
                best_peak = peak
                best_idx = idx

        if best_idx is not None and best_peak >= INPUT_SILENT_RMS:
            log.info("Auto-selected microphone [%d]: %s (rms=%.5f)",
                     best_idx, sd.query_devices(best_idx).get("name", "?"), best_peak)
            return best_idx

        log.warning("No microphone carried any signal; falling back to the system default.")
        return None

    def _probe_input_blocks(self, sd, device, seconds: float = 0.0,
                            skip_speaking: bool = False) -> Optional[list]:
        """Per-block RMS over a short window, or None if the device will not open.

        With ``skip_speaking`` the assistant's own playback is excluded: a
        block captured while it is talking measures the speaker, not the room,
        and the window is extended instead so the sample is still full.
        """
        try:
            with sd.RawInputStream(
                device=device,
                samplerate=self.sample_rate,
                blocksize=VAD_BLOCK,
                dtype="int16",
                channels=self.channels,
            ) as stream:
                levels = []
                block_s = VAD_BLOCK / float(self.sample_rate or 16000)
                deadline = time.monotonic() + seconds
                hard_deadline = deadline + VAD_SPEAKING_SKIP_S
                while time.monotonic() < deadline:
                    data, _overflow = stream.read(VAD_BLOCK)
                    if not data:
                        break
                    if skip_speaking and self._is_speaking:
                        deadline = min(
                            max(deadline, time.monotonic() + block_s),
                            hard_deadline,
                        )
                        continue
                    levels.append(UtteranceSegmenter.rms(bytes(data)))
                return levels or None
        except Exception:
            return None

    def _calibrate_vad(self, sd, device) -> None:
        """Measure the room, then publish the result to waiting callers."""
        self._calibrated.clear()
        try:
            self._calibrate_vad_inner(sd, device)
        finally:
            self._calibrated.set()

    def _calibrate_vad_inner(self, sd, device) -> None:
        """Set the speech gate from the room's own noise level.

        Called once the input device is chosen. A single fixed threshold is
        wrong on real hardware: it is too high for a quiet laptop array and
        too low for a noisy desk, and in both directions the failure is silent
        rather than obvious.

        The noise level is a low percentile of the block levels, not the peak
        and not the median. A peak only measures the loudest thing that
        happened during the window, so a cough or a keystroke during
        calibration would be mistaken for the noise floor and push the gate
        above the user's own voice. The median is no better, because the
        window often contains the user speaking and the median then reports
        the voice as the room. The quietest blocks are the room.
        """
        override = os.getenv("JARVIS_VAD_THRESHOLD", "").strip()
        if override:
            try:
                self.vad_threshold = float(override)
                self.speaking_threshold = self.vad_threshold * VAD_SPEAKING_MULTIPLE
                self._noise_floor = None
                log.info(
                    "VAD threshold set to %.5f from JARVIS_VAD_THRESHOLD.",
                    self.vad_threshold)
                return
            except ValueError:
                log.warning("Ignoring invalid JARVIS_VAD_THRESHOLD=%r", override)

        levels = self._probe_input_blocks(
            sd, device, VAD_CALIBRATE_S, skip_speaking=True)
        if not levels:
            self._noise_floor = None
            self._vad_warning = None
            log.info("Could not measure the room noise; using vad_threshold=%.5f.",
                     self.vad_threshold)
            return

        ordered = sorted(levels)
        index = min(len(ordered) - 1,
                    max(0, int(len(ordered) * VAD_NOISE_PERCENTILE)))
        floor = ordered[index]
        self._noise_floor = floor

        # The gate must always clear the room: a gate below the noise reads
        # every breath as speech, so the assistant talks over itself and never
        # finishes an utterance. There is deliberately no upper clamp, because
        # clamping is exactly what silenced the user in the first place.
        derived = max(floor * VAD_NOISE_MULTIPLE, VAD_MIN_THRESHOLD)
        previous = self.vad_threshold
        self.vad_threshold = derived
        self.speaking_threshold = derived * VAD_SPEAKING_MULTIPLE
        self._vad_warning = None

        if abs(derived - VAD_THRESHOLD) > 1e-6:
            log.info(
                "VAD threshold calibrated to %.5f from a room noise level of %.5f "
                "(was %.5f; peak during calibration was %.5f).",
                derived, floor, previous, ordered[-1])
        if derived > VAD_WARN_THRESHOLD:
            self._vad_warning = (
                "The room is too loud (noise %.3f), so the gate is %.3f and a "
                "quiet voice may not be detected. Move closer to the "
                "microphone, or turn the Windows input level down: a weak "
                "microphone amplifies its own hiss along with your voice."
                % (floor, derived))
            log.warning("%s", self._vad_warning)

    def _listen_loop(self) -> None:
        """Capture microphone audio and dispatch recognised commands."""
        try:
            import sounddevice as sd
        except ImportError:
            self._last_listen_error = "sounddevice is not installed"
            log.warning("Speech recognition unavailable: %s", self._last_listen_error)
            return

        try:
            device = self._select_input_device(sd)
        except ValueError as e:
            # An explicit override that cannot be honoured is a configuration
            # error, not something to paper over by using a different mic.
            self._last_listen_error = str(e)
            log.error("%s", e)
            return

        recognizer = KaldiRecognizer(self._vosk_model, float(self.sample_rate))
        recognizer.SetWords(False)
        self._calibrate_vad(sd, device)
        segmenter = self._new_segmenter()

        backoff = 1.0
        while not self._listener_stop.is_set():
            try:
                with sd.RawInputStream(
                    device=device,
                    samplerate=self.sample_rate,
                    blocksize=VAD_BLOCK,
                    dtype="int16",
                    channels=self.channels,
                ) as stream:
                    # The pid is in the message on purpose: Windows lets several
                    # processes share a capture device, so a multi-worker run
                    # looks identical to a normal one unless you can tell the
                    # listeners apart. Two of these lines means two processes
                    # will both run whatever you say.
                    chosen = device if device is not None else "default"
                    log.info("Listening for '%s' on microphone %s (pid %d).",
                             self.wake_word, chosen, os.getpid())
                    self._last_listen_error = ""
                    backoff = 1.0

                    while not self._listener_stop.is_set():
                        # Releasing push-to-talk ends the utterance immediately.
                        if self._ptt_flush.is_set():
                            self._ptt_flush.clear()
                            pending = segmenter.flush()
                            if pending:
                                threading.Thread(
                                    target=self._finish_utterance_async,
                                    args=(pending, True),
                                    daemon=True,
                                ).start()
                            recognizer.Reset()

                        data, _overflow = stream.read(VAD_BLOCK)
                        if not data:
                            continue

                        # sounddevice 0.5.x hands back a cffi buffer, which Vosk's
                        # AcceptWaveform rejects ("must be a cdata pointer").
                        # Copying to bytes also gives the segmenter a stable,
                        # sliceable object to work with.
                        data = bytes(data)

                        # Never transcribe our own output: that is how an
                        # assistant ends up answering itself. The wake word
                        # still gates dispatch, but while JARVIS is talking we
                        # demand louder speech so the echo off the speakers is
                        # less likely to clear the gate, and the wake word or
                        # push-to-talk can cut in to interrupt.
                        segmenter.threshold = (
                            self.speaking_threshold if self._is_speaking else self.vad_threshold
                        )

                        level = segmenter.rms(data)
                        self._recent_levels.append(level)
                        # Adapt gain on every block, not just those above the
                        # gate. Otherwise a quiet mic can never bootstrap itself:
                        # nothing clears the threshold, so gain never adapts,
                        # so nothing ever clears the threshold.
                        self._adapt_input_gain(level)
                        if level >= segmenter.threshold:
                            self._blocks_above_gate += 1

                        # The recognizer is deliberately NOT fed here. Vosk runs
                        # its acoustic model on every accepted frame, so feeding
                        # it the room tone costs a third of a CPU core for a
                        # machine that is only ever silent. The segmenter
                        # already decides where an utterance starts and ends,
                        # so the recognizer only ever sees the speech itself.
                        utterance = segmenter.push(data, level)
                        if utterance:
                            self._utterances_formed += 1
                            self._utterances_formed_at = time.monotonic()
                            # Recognition runs in a background thread so the
                            # listen loop never stops capturing audio while
                            # Vosk decodes the previous utterance.
                            threading.Thread(
                                target=self._finish_utterance_async,
                                args=(utterance, False),
                                daemon=True,
                            ).start()
                            recognizer.Reset()

                        # Dynamically lower the VAD gate if no utterance has
                        # been detected for a while. A quiet microphone or a
                        # suddenly quiet room can leave the gate above the
                        # user's voice, making JARVIS appear deaf. Gradually
                        # lowering the gate until speech is detected again
                        # self-corrects without user intervention.
                        now = time.monotonic()
                        if (
                            now - self._utterances_formed_at > 10.0
                            and self.vad_threshold > VAD_MIN_THRESHOLD
                        ):
                            self.vad_threshold = max(
                                VAD_MIN_THRESHOLD,
                                self.vad_threshold * 0.9,
                            )
                            self._utterances_formed_at = now
                            log.info(
                                "VAD gate lowered to %.5f (no speech detected).",
                                self.vad_threshold,
                            )

            except Exception as e:
                # A mic that gets unplugged, or is grabbed by another app, must
                # not take the assistant down.
                self._last_listen_error = str(e)
                log.warning("Microphone error: %s", e)
                if self._listener_stop.wait(backoff):
                    break
                backoff = min(backoff * 2, 30.0)

    def _new_segmenter(self) -> UtteranceSegmenter:
        return UtteranceSegmenter(
            sample_rate=self.sample_rate,
            block=VAD_BLOCK,
            threshold=self.vad_threshold,
            silence_ms=int(self.silence_timeout * 1000),
            max_utterance_s=self.max_utterance_s,
        )

    def _adapt_input_gain(self, level: float) -> None:
        """Track a gain that brings speech up to a level a recognizer can use.

        A gate can be low enough to admit speech while the speech itself is
        still far too quiet to recognise: on this machine a raised voice peaks
        around 0.004, roughly -48 dBFS, and Vosk returns an empty transcript
        for that. Detection and recognition want different levels, so the gate
        runs on the raw signal and only the audio handed to the recognizer is
        lifted.

        The gain is smoothed rather than applied per block, so one loud block
        does not yank the level around mid-utterance.
        """
        desired = VAD_AGC_TARGET / max(level, 1e-6)
        desired = min(VAD_AGC_MAX_GAIN, max(1.0, desired))
        self._input_gain = (VAD_AGC_SMOOTHING * self._input_gain
                            + (1 - VAD_AGC_SMOOTHING) * desired)

    def _gain_audio(self, data: bytes) -> bytes:
        """Scale a block for the recognizer, without clipping."""
        gain = self._input_gain
        if abs(gain - 1.0) < 1e-3:
            return data
        try:
            import numpy as np
            samples = np.frombuffer(data[:len(data) - (len(data) % 2)], dtype=np.int16)
            scaled = samples * gain
            clipped = np.clip(scaled, -32768, 32767).astype(np.int16)
            return clipped.tobytes()
        except Exception:
            samples = array.array("h")
            samples.frombytes(data[:len(data) - (len(data) % 2)])
            for i, value in enumerate(samples):
                scaled = int(value * gain)
                samples[i] = 32767 if scaled > 32767 else (-32768 if scaled < -32768 else scaled)
            return samples.tobytes()

    def _finish_utterance(
        self,
        recognizer,
        utterance: Optional[bytes] = None,
        allow_bare: bool = False,
    ) -> None:
        """Close the current utterance, recognise it, and dispatch if it counts.

        ``utterance`` is the audio to transcribe. When it is given, it is fed to
        the recognizer here and the recognizer only ever hears speech - the
        listener deliberately does not stream room tone into it. When it is
        None the audio has already been streamed to the recognizer block by
        block, and this only collects the result.

        ``allow_bare`` accepts speech with no wake word, which is what
        push-to-talk needs on release: the button is already up by then, so the
        wake-word gate would otherwise throw the command away.
        """
        if utterance:
            try:
                recognizer.AcceptWaveform(self._gain_audio(utterance))
            except Exception as e:
                log.warning("Recognition failed: %s", e)
                return

        try:
            raw = recognizer.FinalResult()
        except Exception as e:
            log.warning("Recognition failed: %s", e)
            return

        try:
            text = str(json.loads(raw).get("text", "")).strip()
        except (ValueError, AttributeError):
            text = ""
        if not text:
            return

        self._last_transcript = text
        ptt = allow_bare or self._ptt_active.is_set()
        addressed, command = strip_wake_word(text, self.wake_word)

        if ptt:
            addressed = True  # hold-to-talk needs no wake word
        elif not addressed:
            log.debug("Ignored (no wake word): %s", text)
            return
        elif self._wake_blocked():
            # This is the tail of a phrase that already interrupted us.
            log.debug("Ignored (wake word cooling down): %s", text)
            return

        if len(command) < self.min_command_len:
            log.debug("Ignored (too short): %s", text)
            return

        log.info("HEARD: %s", command)
        self._emit({"type": "heard", "text": command})

        # Saying the wake word while JARVIS is talking is a barge-in.
        if self._is_speaking:
            self.interrupt()
            self._wake_blocked_until = time.monotonic() + self.wake_cooldown_s

        if _command_hook is not None:
            _command_hook(command)
        else:
            log.warning("Speech recognised but no command hook is installed: %s", command)

    def _finish_utterance_async(self, utterance: bytes, allow_bare: bool = False) -> None:
        """Same as _finish_utterance but creates its own recognizer so it can
        run on a background thread without blocking the microphone capture."""
        if not HAS_VOSK:
            log.warning("Vosk not available for async decode.")
            return
        try:
            recognizer = KaldiRecognizer(self._vosk_model, float(self.sample_rate))
            recognizer.SetWords(False)
        except Exception as e:
            log.warning("Could not create recognizer for async decode: %s", e)
            return
        try:
            recognizer.AcceptWaveform(self._gain_audio(utterance))
        except Exception as e:
            log.warning("Async recognition failed: %s", e)
            return

        try:
            raw = recognizer.FinalResult()
        except Exception as e:
            log.warning("Async FinalResult failed: %s", e)
            return

        try:
            text = str(json.loads(raw).get("text", "")).strip()
        except (ValueError, AttributeError):
            text = ""
        if not text:
            return

        self._last_transcript = text
        ptt = allow_bare or self._ptt_active.is_set()
        addressed, command = strip_wake_word(text, self.wake_word)

        if ptt:
            addressed = True
        elif not addressed:
            log.debug("Ignored (no wake word): %s", text)
            return
        elif self._wake_blocked():
            log.debug("Ignored (wake word cooling down): %s", text)
            return

        if len(command) < self.min_command_len:
            log.debug("Ignored (too short): %s", text)
            return

        log.info("HEARD: %s", command)
        self._emit({"type": "heard", "text": command})

        # Saying the wake word while JARVIS is talking is a barge-in.
        if self._is_speaking:
            self.interrupt()
            self._wake_blocked_until = time.monotonic() + self.wake_cooldown_s

        if _command_hook is not None:
            _command_hook(command)
        else:
            log.warning("Speech recognised but no command hook is installed: %s", command)

    # -- Speech output ------------------------------------------------------

    def _fish_audio_speak(self, text: str) -> bool:
        """Speak via Fish Audio (cloud, highest quality, needs API credit)."""
        if not self._fish_client:
            raise RuntimeError("Fish Audio client not initialized")

        from fishaudio.utils import save

        audio = self._fish_client.tts.convert(
            text=text,
            reference_id=self.fish_voice_id if self.fish_voice_id else None,
            model=self.fish_tts_model,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmppath = f.name
        try:
            save(audio, tmppath)
            return self._play_mci(tmppath)
        finally:
            try:
                os.unlink(tmppath)
            except OSError:
                pass

    def _sapi_speak(self, text: str) -> bool:
        """Speak via Windows SAPI (local, offline, no API key needed)."""
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            if self.tts_gain and abs(self.tts_gain - 1.0) > 1e-3:
                speaker.Volume = int(100 * self.tts_gain)
            speaker.Speak(text)
            return True
        except Exception as e:
            log.error("SAPI TTS error: %s", e)
            return False

    # -- Audio output ------------------------------------------------------

    @staticmethod
    def _output_devices(sd) -> List[tuple]:
        """Every device that can play, as (index, info) pairs."""
        try:
            devices = sd.query_devices()
        except Exception as e:
            log.warning("Could not list audio devices: %s", e)
            return []
        return [(i, d) for i, d in enumerate(devices)
                if d.get("max_output_channels", 0) >= 1]

    def _resolve_output_device(self, sd, spec: str) -> int:
        """Turn a JARVIS_OUTPUT_DEVICE value into a device index.

        The value is either an index or a case-insensitive substring of a
        device name. Raises ValueError with something the user can act on.
        """
        spec = spec.strip()
        if spec.isdigit():
            idx = int(spec)
            if idx < 0 or idx >= len(sd.query_devices()):
                raise ValueError(
                    f"JARVIS_OUTPUT_DEVICE index {idx} does not exist "
                    f"(there are {len(sd.query_devices())} devices)")
            return idx

        needle = spec.lower()
        for idx, dev in self._output_devices(sd):
            if needle in str(dev.get("name", "")).lower():
                return idx
        raise ValueError(
            f"No output device matches {spec!r}. List them with: "
            f"python -c \"import sounddevice as sd; print(sd.query_devices())\"")

    def _output_device_opens(self, sd, device) -> bool:
        """Whether an output device can actually be opened right now.

        This is the connection test. Windows keeps a Bluetooth headset in the
        device list long after it has disconnected, so picking a device purely
        by name would silently send speech to hardware the user is not wearing.
        A disconnected endpoint fails to open, and fails immediately, so
        opening a stream is a cheap and reliable way to tell the two apart.
        """
        try:
            stream = sd.OutputStream(device=device, channels=1,
                                     samplerate=OUTPUT_PROBE_SR,
                                     blocksize=OUTPUT_PROBE_BLOCKSIZE)
        except Exception:
            return False
        try:
            stream.write(b"\x00" * (OUTPUT_PROBE_BLOCKSIZE * 4), overflowed=True)
            stream.start()
        except Exception:
            return False
        finally:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        return True

    def _select_output_device(self, sd) -> Optional[int]:
        """Pick where speech is played: a worn headset, else the system default.

        Returns None to mean "use whatever Windows considers the default", which
        is the right answer for the built-in speakers and for any output we do
        not recognise.

        The choice is cached for a few seconds so that plugging earbuds in
        mid-conversation is noticed without probing the device list on every
        single reply.
        """
        now = time.monotonic()
        if self._output_device_cache and \
                now - self._output_device_cached_at < OUTPUT_RECHECK_S:
            return self._output_device_cache

        override = (os.environ.get("JARVIS_OUTPUT_DEVICE") or "").strip()
        if override:
            try:
                idx = self._resolve_output_device(sd, override)
            except ValueError as e:
                log.warning("%s", e)
                self._output_device_cache = None
                self._output_device_cached_at = now
                return None
            if idx != self._output_device_cache:
                name = sd.query_devices(idx).get("name", "?")
                log.info("Playing audio on JARVIS_OUTPUT_DEVICE [%d]: %s", idx, name)
            self._output_device_cache = idx
            self._output_device_cached_at = now
            return idx

        chosen = None
        for idx, dev in self._output_devices(sd)[:MAX_PROBE_DEVICES]:
            name = str(dev.get("name", "")).lower()
            if not any(k in name for k in OUTPUT_HEADSET_HINTS):
                continue
            if any(k in name for k in OUTPUT_HANDSFREE_HINTS):
                continue
            if self._output_device_opens(sd, idx):
                chosen = idx
                log.info("Playing audio on the connected headset [%d]: %s",
                         idx, dev.get("name", "?"))
                break

        if chosen is None:
            try:
                default = sd.default.device[1]
                log.debug("No headset connected; using the default output [%d] %s",
                          default, sd.query_devices(default).get("name", "?"))
            except Exception as e:
                log.debug("Could not read the default output device: %s", e)

        self._output_device_cache = chosen
        self._output_device_cached_at = now
        return chosen

    def _stop_audio_device(self) -> None:
        """Halt sounddevice playback if it is running."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def _stop_mci(self) -> None:
        """Halt MCI playback if it is running."""
        alias = self._active_alias
        if not alias or not self._winmm:
            return
        try:
            self._winmm.mciSendStringW(f"stop {alias}", None, 0, None)
        except Exception:
            pass

    def _play_mci(self, path: str) -> bool:
        """Play an audio file, preferring sounddevice and falling back to MCI.

        Playback is interruptible: :meth:`interrupt` sets ``_stop_playback``
        and stops the device, and this call returns promptly.
        """
        self._stop_playback.clear()

        try:
            import sounddevice as sd
            import soundfile as sf
            data, sr = sf.read(path, dtype='float32')
            if abs(self.tts_gain - 1.0) > 1e-3:
                import numpy as np
                # Clip rather than wrap: a large boost should flatten the
                # loudest peaks, not turn them into crackle.
                data = np.clip(data * self.tts_gain, -1.0, 1.0)
            out = self._select_output_device(sd)
            sd.play(data, sr, device=out)
            while not self._stop_playback.wait(0.05):
                try:
                    if not sd.get_stream().active:
                        break
                except Exception:
                    break
            if self._stop_playback.is_set():
                sd.stop()
            return True
        except Exception as e:
            log.debug("sounddevice playback unavailable (%s), using MCI", e)

        if not self._winmm:
            log.error("No audio output available (no sounddevice, no MCI).")
            return False

        with self._mci_lock:
            self._mci_counter += 1
            alias = f"_jtts{self._mci_counter}"
            self._active_alias = alias
        try:
            self._winmm.mciSendStringW(
                f'open "{path}" type mpegvideo alias {alias}', None, 0, None
            )
            self._winmm.mciSendStringW(f"play {alias} wait", None, 0, None)
            return True
        except Exception as e:
            # Retry as a generic wave file (Fish can also return raw WAV).
            try:
                self._winmm.mciSendStringW(
                    f'open "{path}" type waveaudio alias {alias}', None, 0, None
                )
                self._winmm.mciSendStringW(f"play {alias} wait", None, 0, None)
                return True
            except Exception as e2:
                log.error("MCI playback error: %s / %s", e, e2)
                return False
            finally:
                try:
                    self._winmm.mciSendStringW(f"close {alias}", None, 0, None)
                except Exception:
                    pass
        finally:
            self._active_alias = None

    def get_vosk_model(self):
        """Get the Vosk model for STT."""
        return self._vosk_model

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking

    def get_tts_done_time(self) -> float:
        """Get timestamp of last TTS completion."""
        return self._tts_done_at

    def get_speech_idle_event(self) -> threading.Event:
        """Set while no TTS playback is in progress."""
        return self._speech_idle

    def get_last_transcript(self) -> str:
        """The most recent thing Vosk heard, wake word included."""
        return self._last_transcript

    def get_listen_error(self) -> str:
        """Why capture is not running, or an empty string if it is."""
        return self._last_listen_error

    def has_live_tts(self) -> bool:
        """True if at least one TTS backend is available to speak."""
        return bool([b for b in self._backend_order if b not in self._disabled_backends])

    def _get_status_text(self) -> str:
        """Get voice status as text."""
        status = []
        if self.is_listening():
            mode = "push-to-talk" if self.is_push_to_talk() else f"wake word '{self.wake_word}'"
            status.append(f"Listening on {mode}")
        elif self.is_push_to_talk():
            status.append("Push-to-talk requested, but the microphone is not running")
        elif self._vosk_model:
            status.append("Vosk STT ready, microphone not started")
        else:
            status.append("Vosk STT not loaded, using typed input")

        live = [b for b in self._backend_order if b not in self._disabled_backends]
        status.append(f"TTS: {self._active_backend or 'idle'} (chain {' -> '.join(live) or 'empty'})")
        if self._disabled_backends:
            status.append(f"disabled: {', '.join(sorted(self._disabled_backends))}")
        status.append(f"muted: {self._muted}")
        return " | ".join(status)

    def _output_device_name(self) -> str:
        """Name of the device replies are playing on, for the status readout.

        Answering "why can I not hear it" should not require reading logs, and
        the two real causes are a muted device and a device the user is not
        wearing, both of which are invisible unless the target is shown.
        """
        try:
            import sounddevice as sd
            idx = self._select_output_device(sd)
            if idx is None:
                idx = sd.default.device[1]
                return "%s (system default)" % sd.query_devices(idx).get("name", "?")
            return str(sd.query_devices(idx).get("name", "?"))
        except Exception:
            return "unknown"

    def _input_level_summary(self) -> Dict[str, Any]:
        """What the listener has actually been hearing.

        Reported so a recognition problem can be told apart into its two very
        different causes: blocks that never reach the gate (the gate is too
        high, or the microphone is dead) versus blocks that clear it but
        recognise as nothing (the problem is then the model or the wording).
        """
        levels = sorted(self._recent_levels)
        if not levels:
            return {"blocks": 0}
        return {
            "blocks": len(levels),
            "median": levels[len(levels) // 2],
            "max": levels[-1],
            "min": levels[0],
            "above_gate": self._blocks_above_gate,
            "utterances": self._utterances_formed,
            "gain": round(self._input_gain, 2),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "vosk_loaded": self._vosk_model is not None,
            "listening": self.is_listening(),
            "push_to_talk": self.is_push_to_talk(),
            "wake_word": self.wake_word,
            "last_transcript": self._last_transcript,
            "listen_error": self._last_listen_error,
            "vad_threshold": self.vad_threshold,
            "speaking_threshold": self.speaking_threshold,
            "noise_floor": self._noise_floor,
            "vad_warning": self._vad_warning,
            "input_levels": self._input_level_summary(),
            "output_device": self._output_device_name(),
            "fish_audio_loaded": self._fish_client is not None,
            "tts_backend": self._active_backend,
            "tts_backends": list(self._backend_order),
            "tts_backends_live": [
                b for b in self._backend_order if b not in self._disabled_backends
            ],
            "tts_backends_disabled": sorted(self._disabled_backends),
            "tts_voice_down": not self.has_live_tts(),
            "fish_tts_model": self.fish_tts_model,
            "fish_voice_id": self.fish_voice_id,
            "muted": self._muted,
            "speaking": self._is_speaking,
        }


# Bind the backend implementation once the class body exists. There is one
# entry by design; see _build_tts_chain.
VoiceMCP._TTS_BACKENDS = {
    "fish-audio": VoiceMCP._fish_audio_speak,
    "sapi": VoiceMCP._sapi_speak,
}
