#!/usr/bin/env python3
"""
J.A.R.V.I.S. — Just A Rather Very Intelligent System
======================================================
Real-time AI assistant with a holographic-style web HUD.

Architecture:
  - Python backend: Vosk STT, Gemini LLM, Windows SAPI TTS, psutil metrics
  - Web frontend:   JARVIS HUD served on localhost, connected via WebSocket
  - Wake methods:   Say "Jarvis" (voice) or double-clap (amplitude detection)

Run:
  python jarvis.py

Then open http://localhost:8080 (opens automatically).
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import json
import logging
import os
import queue
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import psutil
import sounddevice as sd

try:
    import win32com.client
    _HAS_SAPI = True
except ImportError:
    _HAS_SAPI = False

# ─── Load environment ────────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent / ".env")

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jarvis")

# =============================================================================
# CONFIG
# =============================================================================
SAMPLE_RATE       = 16000        # 16 kHz — optimal for Vosk
BLOCK_MS          = 40
CHANNELS          = 1

# Double-clap detection
SPIKE_RATIO       = 7.0
COOLDOWN_S        = 0.45
MIN_DOUBLE_GAP_S  = 0.05
MAX_DOUBLE_GAP_S  = 0.35
RETRIGGER_RATIO   = 0.55
NOISE_FLOOR_ALPHA = 0.992
MIN_RMS           = 0.012
QUIET_GATE_MULT   = 2.2

# Voice
WAKE_WORD         = "jarvis"
VOSK_MODEL_PATH   = Path(__file__).resolve().parent / "model"
GEMINI_MODEL      = "gemini-2.0-flash"
MEMORY_PATH       = Path(__file__).resolve().parent / "jarvis_memory.json"
VOICE_CMD_SILENCE  = 2.0         # seconds of silence to end a command
MIN_COMMAND_LEN   = 3

# Servers
HTTP_PORT         = 8080
WS_PORT           = 8765
STATIC_DIR        = Path(__file__).resolve().parent / "static"

# =============================================================================
# GLOBAL STATE
# =============================================================================
_tts_lock         = threading.Lock()
_speech_q: queue.Queue[str | None] = queue.Queue()
_event_q: queue.Queue[dict]        = queue.Queue(maxsize=500)
_ws_clients: set                   = set()       # populated by asyncio loop


def broadcast(event: dict) -> None:
    """Thread-safe: enqueue an event for WebSocket broadcast."""
    try:
        _event_q.put_nowait(event)
    except queue.Full:
        pass


def set_status(state: str) -> None:
    broadcast({"type": "status", "state": state})


# =============================================================================
# MEMORY
# =============================================================================
def load_memory() -> dict:
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Could not load memory: %s", e)
        return {}


# =============================================================================
# TEXT-TO-SPEECH (Windows SAPI)
# =============================================================================
def _sapi_speak(text: str) -> None:
    if not _HAS_SAPI:
        log.warning("Windows SAPI not available — TTS disabled.")
        return
    with _tts_lock:
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
        except Exception as e:
            log.error("TTS error: %s", e)


def speak(text: str) -> None:
    """Queue text for speech + broadcast to frontend."""
    log.info("JARVIS says: %s", text)
    broadcast({"type": "transcript", "text": text, "speaker": "jarvis"})
    _speech_q.put(text)


def _speech_worker() -> None:
    while True:
        text = _speech_q.get()
        if text is None:
            break
        set_status("speaking")
        _sapi_speak(text)
        set_status("standby")
        _speech_q.task_done()


# =============================================================================
# GEMINI AI BRAIN
# =============================================================================
def _build_system_prompt(memory: dict) -> str:
    identity = memory.get("secure_identity", {})
    tasks    = memory.get("pending_tasks", [])
    tasks_s  = "\n".join(f"  - {t}" for t in tasks) if tasks else "  (none)"
    return (
        "You are J.A.R.V.I.S. — Just A Rather Very Intelligent System, "
        "a highly sophisticated AI assistant modeled after the AI from the Iron Man films. "
        "You are precise, calm, intelligent, and subtly witty. "
        "Keep responses concise (1-3 sentences). You are speaking out loud, "
        "so use plain sentences only — NO markdown, NO bullet points, NO formatting.\n\n"
        f"Your primary user:\n"
        f"  Name       : {identity.get('name', 'Sir')}\n"
        f"  Role       : {identity.get('role', 'Engineer')}\n"
        f"  Institution: {identity.get('institution', 'N/A')}\n"
        f"  Clearance  : {identity.get('clearance_level', 'Standard')}\n\n"
        f"Pending tasks:\n{tasks_s}\n\n"
        "Always address the user respectfully. Never break character."
    )


def ask_gemini(prompt: str, memory: dict) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return _fallback(prompt, memory)
    try:
        from google import genai
        from google.genai import types as gt
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            config=gt.GenerateContentConfig(
                system_instruction=_build_system_prompt(memory),
                temperature=0.8,
                max_output_tokens=250,
            ),
            contents=prompt,
        )
        return resp.text.strip()
    except Exception as e:
        log.error("Gemini error: %s", e)
        return "I encountered an error with my AI core. Please check the logs."


def _fallback(prompt: str, memory: dict) -> str:
    p = prompt.lower()
    name = memory.get("secure_identity", {}).get("name", "Sir")
    if any(w in p for w in ("time", "date", "clock")):
        return _time_text()
    if any(w in p for w in ("status", "cpu", "ram", "battery")):
        return _status_text()
    if any(w in p for w in ("task", "todo", "pending")):
        return _tasks_text(memory)
    return f"I'm at your service, {name}. However, my AI core is offline. Set GOOGLE_API_KEY in .env to enable full reasoning."


# =============================================================================
# BUILT-IN COMMANDS
# =============================================================================
def _time_text() -> str:
    n = datetime.now()
    return f"It is currently {n.strftime('%I:%M %p')} on {n.strftime('%A, %B %d, %Y')}."


def _status_text() -> str:
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    parts = [
        f"CPU utilization is at {cpu:.1f} percent.",
        f"RAM usage: {ram.used / (1024**3):.1f} of {ram.total / (1024**3):.1f} gigabytes.",
    ]
    try:
        bat = psutil.sensors_battery()
        if bat:
            s = "charging" if bat.power_plugged else "on battery"
            parts.append(f"Battery at {bat.percent:.0f} percent, {s}.")
    except Exception:
        pass
    return " ".join(parts)


def _tasks_text(memory: dict) -> str:
    tasks = memory.get("pending_tasks", [])
    if not tasks:
        return "No pending tasks on file, sir."
    intro = f"You have {len(tasks)} pending task{'s' if len(tasks) != 1 else ''}. "
    body  = " ".join(f"Task {i+1}: {t}" for i, t in enumerate(tasks))
    return intro + body


def handle_command(text: str, memory: dict) -> None:
    """Route a command to the right handler and speak the response."""
    cmd = text.strip()
    if len(cmd) < MIN_COMMAND_LEN:
        return
    log.info("USER command: %s", cmd)
    broadcast({"type": "transcript", "text": cmd, "speaker": "user"})
    set_status("processing")

    p = cmd.lower()
    if any(w in p for w in ("time", "date", "clock", "what day")):
        resp = _time_text()
    elif any(w in p for w in ("status", "cpu", "ram", "battery", "system")):
        resp = _status_text()
    elif any(w in p for w in ("task", "todo", "pending", "remind")):
        resp = _tasks_text(memory)
    else:
        resp = ask_gemini(cmd, memory)

    speak(resp)


# =============================================================================
# VOSK SPEECH RECOGNITION
# =============================================================================
def _load_vosk():
    if not VOSK_MODEL_PATH.is_dir():
        log.warning("Vosk model not found at '%s'. Voice recognition disabled.", VOSK_MODEL_PATH)
        return None
    try:
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        m = Model(str(VOSK_MODEL_PATH))
        log.info("Vosk model loaded.")
        return m
    except Exception as e:
        log.warning("Vosk load failed: %s", e)
        return None


def _vosk_thread(model, audio_q: queue.Queue, memory: dict) -> None:
    from vosk import KaldiRecognizer
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(False)

    awake       = False
    cmd_buf: list[str] = []
    last_speech = time.monotonic()

    log.info('Say "%s" to wake the assistant.', WAKE_WORD.capitalize())

    while True:
        # --- drain audio queue ------------------------------------------------
        try:
            raw: bytes = audio_q.get(timeout=0.5)
        except queue.Empty:
            if awake and cmd_buf and (time.monotonic() - last_speech) > VOICE_CMD_SILENCE:
                full = " ".join(cmd_buf).strip()
                cmd_buf.clear()
                awake = False
                set_status("standby")
                if len(full) >= MIN_COMMAND_LEN:
                    threading.Thread(target=handle_command, args=(full, memory), daemon=True).start()
            continue

        # --- feed recognizer --------------------------------------------------
        if rec.AcceptWaveform(raw):
            text = json.loads(rec.Result()).get("text", "").strip().lower()
        else:
            text = json.loads(rec.PartialResult()).get("partial", "").strip().lower()

        if not text:
            if awake and cmd_buf and (time.monotonic() - last_speech) > VOICE_CMD_SILENCE:
                full = " ".join(cmd_buf).strip()
                cmd_buf.clear()
                awake = False
                set_status("standby")
                if len(full) >= MIN_COMMAND_LEN:
                    threading.Thread(target=handle_command, args=(full, memory), daemon=True).start()
            continue

        last_speech = time.monotonic()

        if not awake:
            if WAKE_WORD in text:
                awake = True
                cmd_buf.clear()
                after = text.split(WAKE_WORD, 1)[-1].strip()
                if after:
                    cmd_buf.append(after)
                set_status("listening")
                speak("Yes, sir?")
                log.info("Wake word detected — listening for command…")
        else:
            cmd_buf.append(text)


# =============================================================================
# AUDIO THREAD  (mic capture + clap detection + Vosk feed)
# =============================================================================
def _audio_thread(vosk_model, audio_q: queue.Queue, memory: dict) -> None:
    blocksize      = max(1, int(SAMPLE_RATE * BLOCK_MS / 1000))
    noise_floor    = 1e-4
    last_double    = 0.0
    first_clap: float | None = None
    spike_armed    = True

    def rms(block: np.ndarray) -> float:
        b = np.mean(block.astype(np.float64), axis=1) if block.ndim > 1 else block.astype(np.float64)
        return float(np.sqrt(np.mean(b**2))) if b.size else 0.0

    try:
        mic_device = None
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0 and 'microphone' in d['name'].lower() and 'array' not in d['name'].lower():
                    mic_device = i
                    log.info("Using mic: %s (device %d)", d['name'], i)
                    break
        except Exception:
            pass

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32", blocksize=blocksize,
                            device=mic_device) as stream:
            while True:
                data, _ = stream.read(blocksize)

                # ── feed Vosk ─────────────────────────────────────────
                if vosk_model and not audio_q.full():
                    mono = data[:, 0] if data.ndim > 1 else data
                    pcm  = (mono * 32767).astype(np.int16)
                    audio_q.put_nowait(pcm.tobytes())

                # ── voice level for frontend waveform ─────────────────
                level = rms(data)
                broadcast({"type": "voice_level", "level": round(level, 6)})

                # ── adaptive noise floor ──────────────────────────────
                if level < noise_floor * QUIET_GATE_MULT:
                    noise_floor = NOISE_FLOOR_ALPHA * noise_floor + (1 - NOISE_FLOOR_ALPHA) * level
                    noise_floor = max(noise_floor, 1e-7)

                threshold = max(noise_floor * SPIKE_RATIO, MIN_RMS)
                now       = time.monotonic()

                if level < threshold * RETRIGGER_RATIO:
                    spike_armed = True

                if spike_armed and level >= threshold and (now - last_double) >= COOLDOWN_S:
                    spike_armed = False
                    if first_clap is None:
                        first_clap = now
                    else:
                        gap = now - first_clap
                        if gap < MIN_DOUBLE_GAP_S:
                            pass
                        elif gap <= MAX_DOUBLE_GAP_S:
                            first_clap  = None
                            last_double = now
                            log.info("Double clap detected — waking JARVIS.")
                            set_status("listening")
                            speak("At your service.")
                        else:
                            first_clap = now

    except KeyboardInterrupt:
        pass
    except sd.PortAudioError as e:
        log.error("Audio error: %s", e)
        broadcast({"type": "status", "state": "error"})
        broadcast({"type": "transcript", "text": "Microphone not available. Voice input disabled.", "speaker": "system"})


# =============================================================================
# SYSTEM METRICS THREAD
# =============================================================================
def _metrics_thread() -> None:
    while True:
        try:
            cpu = psutil.cpu_percent(interval=1.5)
            ram = psutil.virtual_memory()
            m: dict = {
                "type": "metrics",
                "cpu": round(cpu, 1),
                "ram_used": round(ram.used / (1024**3), 1),
                "ram_total": round(ram.total / (1024**3), 1),
                "ram_pct": round(ram.percent, 1),
            }
            try:
                bat = psutil.sensors_battery()
                if bat:
                    m["battery"]  = round(bat.percent, 0)
                    m["charging"] = bat.power_plugged
            except Exception:
                pass
            broadcast(m)
        except Exception:
            time.sleep(3)


# =============================================================================
# HTTP SERVER  (serves static/ folder)
# =============================================================================
class _QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Suppress request logs."""
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC_DIR), **kw)

    def log_message(self, fmt, *args):
        pass   # silent


def _http_thread() -> None:
    httpd = http.server.HTTPServer(("0.0.0.0", HTTP_PORT), _QuietHTTPHandler)
    httpd.serve_forever()


# =============================================================================
# WEBSOCKET SERVER  (asyncio)
# =============================================================================
async def _ws_handler(ws):
    _ws_clients.add(ws)
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
                if msg.get("type") == "command":
                    txt = msg.get("text", "").strip()
                    if txt:
                        memory = load_memory()
                        threading.Thread(target=handle_command, args=(txt, memory), daemon=True).start()
            except json.JSONDecodeError:
                pass
    finally:
        _ws_clients.discard(ws)


async def _broadcaster():
    """Drain event queue and fan-out to all connected WebSocket clients."""
    import websockets
    while True:
        events: list[dict] = []
        while not _event_q.empty():
            try:
                events.append(_event_q.get_nowait())
            except queue.Empty:
                break
        if events and _ws_clients:
            for ev in events:
                msg = json.dumps(ev)
                websockets.broadcast(_ws_clients, msg)
        await asyncio.sleep(0.04)          # ~25 fps


async def _run_ws_server():
    import websockets
    async with websockets.serve(_ws_handler, "0.0.0.0", WS_PORT):
        log.info("WebSocket server on ws://localhost:%d", WS_PORT)
        await _broadcaster()


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    print()
    print("  =========================================================")
    print("           J.A.R.V.I.S. -- SYSTEM INITIALIZING             ")
    print("       Just A Rather Very Intelligent System                ")
    print("  =========================================================")
    print()

    memory = load_memory()
    name   = memory.get("secure_identity", {}).get("name", "sir")

    # Speech worker
    threading.Thread(target=_speech_worker, daemon=True).start()

    # Vosk model
    vosk_model = _load_vosk()
    audio_q: queue.Queue[bytes] = queue.Queue(maxsize=300)

    if vosk_model:
        threading.Thread(target=_vosk_thread, args=(vosk_model, audio_q, memory), daemon=True).start()

    # Audio capture + clap detection
    threading.Thread(target=_audio_thread, args=(vosk_model, audio_q, memory), daemon=True).start()

    # System metrics
    threading.Thread(target=_metrics_thread, daemon=True).start()

    # HTTP server
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_http_thread, daemon=True).start()
    log.info("HUD server on http://localhost:%d", HTTP_PORT)

    # Open browser after a short delay
    def _open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{HTTP_PORT}")
    threading.Thread(target=_open_browser, daemon=True).start()

    # WebSocket server (runs in main thread's asyncio loop)
    try:
        asyncio.run(_run_ws_server())
    except KeyboardInterrupt:
        log.info("Shutting down. Goodbye, %s.", name)
        _speech_q.put(None)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
