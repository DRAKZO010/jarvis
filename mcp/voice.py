"""Voice MCP - Speech-to-Text, Text-to-Speech, and Wake Word Detection."""

from __future__ import annotations
import asyncio
import ctypes
import json
import logging
import os
import queue
import re
import tempfile
import threading
import time
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

# Samples per capture block. 4000 samples at 16 kHz is 250 ms, which is fine
# enough to notice a pause between words without flooding Vosk with tiny frames.
VAD_BLOCK = 4000

# Normalised RMS (0.0 - 1.0) above which a block counts as speech. Roughly
# -36 dBFS: quiet enough for a normal desk microphone in a quiet room.
VAD_THRESHOLD = 0.015

# Microphone selection. A machine with headsets, speakers and virtual audio
# inputs often has a Windows default input that carries no signal at all, so
# the default is probed and the loudest input wins if it is dead.
INPUT_PROBE_S = 0.4
INPUT_SILENT_RMS = 0.001
# Probing opens each device in turn, so bound how many are tried; 0.4s each on
# a machine with 40 inputs would otherwise stall startup for a quarter minute.
MAX_PROBE_DEVICES = 12


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
        silence_ms: int = 1200,
        min_speech_ms: int = 300,
        max_utterance_s: float = 12.0,
        pre_roll_blocks: int = 2,
    ):
        self.sample_rate = sample_rate
        self.block = block
        self.threshold = threshold
        self.block_ms = int(block * 1000 / sample_rate)
        self.silence_blocks = max(1, silence_ms // max(1, self.block_ms))
        self.min_speech_blocks = max(1, min_speech_ms // max(1, self.block_ms))
        self.max_blocks = max(1, int(max_utterance_s * 1000 / max(1, self.block_ms)))
        self.pre_roll_blocks = pre_roll_blocks

        self._buffer = bytearray()
        self._pre_roll: List[bytes] = []
        self._quiet_run = 0
        self._in_speech = False

    @staticmethod
    def rms(data: bytes) -> float:
        """Root-mean-square amplitude of a 16-bit PCM block, normalised."""
        if not data:
            return 0.0
        count = len(data) // 2
        if count == 0:
            return 0.0
        total = 0
        for i in range(0, count * 2, 2):
            sample = int.from_bytes(data[i:i + 2], "little", signed=True)
            total += sample * sample
        return (total / count) ** 0.5 / 32768.0

    def _speech_blocks(self) -> int:
        return len(self._buffer) // (self.block * 2) if self.block else 0

    def _block_count(self, utterance: bytes) -> int:
        return len(utterance) // (self.block * 2) if self.block else 0

    def push(self, data: bytes) -> Optional[bytes]:
        """Feed one block. Returns a complete utterance, or None."""
        if self.rms(data) >= self.threshold:
            if not self._in_speech:
                # Start on speech, but keep the preceding quiet blocks so the
                # first syllable is not clipped off.
                self._in_speech = True
                self._buffer = bytearray(b"".join(self._pre_roll))
                self._pre_roll.clear()
                self._quiet_run = 0
            self._buffer += data
            self._quiet_run = 0
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
            return None
        utterance = bytes(self._buffer)
        self._in_speech = False
        self._buffer = bytearray()
        self._pre_roll.clear()
        self._quiet_run = 0
        if self._speech_blocks_len(utterance) < self.min_speech_blocks:
            return None
        return utterance

    def _speech_blocks_len(self, utterance: bytes) -> int:
        return self._block_count(utterance)


def strip_wake_word(text: str, wake_word: str = "jarvis") -> tuple[bool, str]:
    """Split ``text`` into (was the wake word used, remaining command).

    Vosk rarely hears a clean prefix, so common fillers before and after the
    wake word are tolerated: "hey jarvis", "okay jarvis, what time is it".
    """
    if not text:
        return False, ""
    if not wake_word:
        return True, text.strip()
    pattern = re.compile(
        r"^\s*(?:(?:hey|hi|hello|ok|okay|yo)\s+)?"
        r"(?:" + re.escape(wake_word) + r")\b[\s,.:;!\-]*"
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

# Try importing edge-tts (free Microsoft Edge TTS, no API key needed)
try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

# Try importing Windows SAPI (always present on Windows, fully offline)
try:
    import win32com.client
    HAS_SAPI = True
except ImportError:
    HAS_SAPI = False


class VoiceMCP(MCPBase):
    """Voice input/output MCP module."""

    name = "voice"
    description = "Speech-to-Text, Text-to-Speech, and Wake Word Detection"
    commands = [
        "voice status",
        "voice mute",
        "voice unmute",
        "set voice",
        "list voices",
    ]

    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        self.block_ms = 40
        self.channels = 1
        self.wake_word = "jarvis"
        self.min_command_len = 3
        self.voice_cmd_silence = 1.5
        self.tts_cooldown = 1.5
        self.vad_threshold = VAD_THRESHOLD
        # Louder bar while JARVIS is talking, to ignore its own echo.
        self.speaking_threshold = VAD_THRESHOLD * 3.0
        # After a barge-in, ignore further wake words briefly so the tail of the
        # interrupting phrase does not immediately trigger a second command.
        self.wake_cooldown_s = 0.75

        # Fish Audio TTS Settings
        self.fish_api_key = os.getenv("FISH_API_KEY", "")
        self.fish_voice_id = os.getenv("FISH_VOICE_ID", "")
        self.fish_tts_model = os.getenv("FISH_TTS_MODEL", "s2.1-pro")
        self.edge_voice = os.getenv("EDGE_TTS_VOICE", "en-GB-RyanNeural")
        self._fish_client = None

        # Backend preference order. Each entry is tried in turn; a backend that
        # fails is disabled for the rest of the session so we do not pay the
        # same round-trip timeout on every single reply.
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
        self.silence_timeout = 1.2
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
        self._edge_tmp_path: Optional[str] = None

    def initialize(self) -> bool:
        """Initialize voice systems."""
        # Load Vosk model
        model_path = Path(__file__).resolve().parent.parent / "model"
        if HAS_VOSK and model_path.is_dir():
            try:
                SetLogLevel(-1)
                self._vosk_model = Model(str(model_path))
                log.info("Vosk model loaded.")
            except Exception as e:
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

        # Build the TTS fallback chain, best available first.
        if self._fish_client:
            self._backend_order.append("fish-audio")
        if HAS_EDGE_TTS:
            self._backend_order.append("edge-tts")
        if HAS_SAPI:
            self._backend_order.append("sapi")

        if not self._backend_order:
            log.error("No TTS backend available. JARVIS will be silent.")
        else:
            log.info("TTS backend chain: %s", " -> ".join(self._backend_order))

        # Start speech worker
        threading.Thread(target=self._speech_worker, daemon=True).start()

        # Open the microphone. A missing model or device is not fatal: every
        # typed command and all of TTS still work.
        self.start_listener()

        self._initialized = True
        log.info("Voice MCP initialized. TTS backends: %d", len(self._backend_order))
        return True

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
        elif command == "set voice":
            if args:
                self.edge_voice = args
                return f"Voice set to {args}, sir."
            return "Please specify a voice name, sir."
        elif command == "list voices":
            return (
                f"Active backend: {self._active_backend or 'none yet'}. "
                f"Edge voice: {self.edge_voice}. "
                f"Fish voice: {self.fish_voice_id or 'default'}."
            )
        return None

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

        A backend that fails is disabled for the rest of the session, so a
        dead provider (e.g. an out-of-credits Fish Audio account) costs one
        failed call rather than one per reply.
        """
        for backend in list(self._backend_order):
            if backend in self._disabled_backends:
                continue
            if self._stop_playback.is_set():
                return False
            try:
                if self._TTS_BACKENDS[backend](self, text):
                    self._active_backend = backend
                    return True
                # A False return can mean "aborted by barge-in" rather than
                # "backend is broken". Do not punish a good backend for that.
                if self._stop_playback.is_set():
                    log.debug("Speech interrupted during '%s' backend.", backend)
                    return False
                self._disable_backend(backend, "returned False")
            except Exception as e:
                if self._stop_playback.is_set():
                    log.debug("Speech interrupted during '%s' backend: %s", backend, e)
                    return False
                self._disable_backend(backend, str(e))

        log.error("All TTS backends exhausted. Tried: %s", self._backend_order)
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
        self._listener_thread = threading.Thread(
            target=self._listen_loop, name="jarvis-stt", daemon=True
        )
        self._listener_thread.start()
        return True

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
                                self._finish_utterance(recognizer, allow_bare=True)

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

                        recognizer.AcceptWaveform(data)
                        utterance = segmenter.push(data)
                        if utterance:
                            self._finish_utterance(recognizer)
                            recognizer.Reset()

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

    def _finish_utterance(self, recognizer, allow_bare: bool = False) -> None:
        """Close the current utterance, recognise it, and dispatch if it counts.

        The audio has already been streamed to the recognizer block by block,
        so this only collects the result. Feeding the buffer again here would
        transcribe every utterance twice.

        ``allow_bare`` accepts speech with no wake word, which is what
        push-to-talk needs on release: the button is already up by then, so the
        wake-word gate would otherwise throw the command away.
        """
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

    # -- Individual TTS backends -------------------------------------------

    def _fish_audio_speak(self, text: str) -> bool:
        """Speak via Fish Audio (cloud, highest quality, needs credits)."""
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

    def _edge_tts_speak(self, text: str) -> bool:
        """Speak via Microsoft Edge TTS (free, no API key, needs network)."""
        if not HAS_EDGE_TTS:
            raise RuntimeError("edge-tts not installed")

        aborted = False

        async def _run() -> None:
            nonlocal aborted
            communicate = edge_tts.Communicate(text, self.edge_voice)
            # Stream chunk by chunk so a barge-in aborts synthesis immediately
            # instead of waiting for the whole utterance to be rendered.
            with open(self._edge_tmp_path, "wb") as fh:
                async for chunk in communicate.stream():
                    if self._stop_playback.is_set():
                        aborted = True
                        break
                    if chunk.get("type") == "audio" and chunk.get("data"):
                        fh.write(chunk["data"])

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            self._edge_tmp_path = f.name
        try:
            asyncio.run(_run())
            if aborted or self._stop_playback.is_set():
                log.debug("edge-tts synthesis aborted by interrupt.")
                return False
            if not os.path.getsize(self._edge_tmp_path):
                raise RuntimeError("edge-tts produced an empty file")
            return self._play_mci(self._edge_tmp_path)
        finally:
            try:
                os.unlink(self._edge_tmp_path)
            except OSError:
                pass

    def _sapi_speak(self, text: str) -> bool:
        """Speak via the built-in Windows SAPI voice (always available)."""
        if not HAS_SAPI:
            raise RuntimeError("Windows SAPI not available")

        import pythoncom

        pythoncom.CoInitialize()
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            fd, tmppath = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            stream.Open(tmppath, 3)  # 3 = SSFM_CREATE_FOR_WRITE
            voice.AudioOutputStream = stream
            # Rate: -10 (slower) to 10 (faster). 0 is the Windows default.
            voice.Rate = int(os.getenv("SAPI_RATE", "1"))
            voice.Volume = 100
            # 0 = SVSFlagsSync. Rendering must finish before the stream closes,
            # otherwise the WAV is truncated.
            voice.Speak(text, 0)
            stream.Close()

            if self._stop_playback.is_set():
                return False

            played = self._play_mci(tmppath)
            try:
                os.unlink(tmppath)
            except OSError:
                pass
            return played
        finally:
            pythoncom.CoUninitialize()

    # -- Audio output ------------------------------------------------------

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
            sd.play(data, sr)
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
            # Retry as a generic wave file (SAPI output is uncompressed WAV).
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
            "fish_audio_loaded": self._fish_client is not None,
            "tts_backend": self._active_backend,
            "tts_backends": list(self._backend_order),
            "tts_backends_live": [
                b for b in self._backend_order if b not in self._disabled_backends
            ],
            "tts_backends_disabled": sorted(self._disabled_backends),
            "fish_tts_model": self.fish_tts_model,
            "edge_voice": self.edge_voice,
            "muted": self._muted,
            "speaking": self._is_speaking,
        }


# Bind the backend implementations once the class body exists.
VoiceMCP._TTS_BACKENDS = {
    "fish-audio": VoiceMCP._fish_audio_speak,
    "edge-tts": VoiceMCP._edge_tts_speak,
    "sapi": VoiceMCP._sapi_speak,
}
