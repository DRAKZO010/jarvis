"""Jarvis - AI Personal Assistant
Main entry point with WebSocket server and command processing.
"""
from __future__ import annotations

import json
import os
import sys
import re
import time
import asyncio
import concurrent.futures
import logging
import threading
import webbrowser
from pathlib import Path
from typing import Optional
from datetime import datetime
import random

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from mcp import (
    VoiceMCP, TasksMCP, MemoryMCP, SystemMCP,
    BrowserMCP, WeatherMCP, LLMMCP, AutomationMCP,
    ContextMCP, RemindersMCP, FilesMCP, NewsMCP,
    EvolutionMCP, LoopGuard, SkillManager, FactExtractor,
)
from mcp import voice as voice_module
from mcp.capabilities import ACTION_CATALOGUE, build_routing_prompt, is_routable
from mcp.automation import clean_app_name

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

# PyAutoGUI fails on headless servers (no display), so fail gracefully
try:
    import pyautogui
except Exception:
    pyautogui = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "Jarvis"
VERSION = "1.1.0"
MIN_COMMAND_LEN = 3
HOST = "127.0.0.1"
PORT = 8765
METRICS_INTERVAL_S = 2.0

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "jarvis.log", encoding="utf-8", mode="a"),
    ],
)


class _AsyncioDisconnectNoise(logging.Filter):
    """Hide the Proactor traceback logged when a browser tab disappears.

    On Windows, asyncio's proactor transport logs a full traceback for a
    ``ConnectionResetError`` raised while closing a socket whose peer has
    already gone. A tab close is normal, and the traceback buried real errors
    in the log, so only that specific callback record is dropped.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if "_call_connection_lost" not in record.getMessage():
            return True
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            return False
        return not (isinstance(exc, OSError)
                    and getattr(exc, "winerror", None) in (10053, 10054))


logging.getLogger("asyncio").addFilter(_AsyncioDisconnectNoise())
log = logging.getLogger("jarvis")

# ---------------------------------------------------------------------------
# MCP singletons
# ---------------------------------------------------------------------------
voice_mcp = VoiceMCP()
tasks_mcp = TasksMCP()
memory_mcp = MemoryMCP()
system_mcp = SystemMCP()
browser_mcp = BrowserMCP()
weather_mcp = WeatherMCP()
llm_mcp = LLMMCP()
automation_mcp = AutomationMCP()
context_mcp = ContextMCP()
reminders_mcp = RemindersMCP()
files_mcp = FilesMCP()
news_mcp = NewsMCP()
evolution_mcp = EvolutionMCP()

# Initialize all MCP modules
for _mcp in [voice_mcp, tasks_mcp, memory_mcp, system_mcp, browser_mcp,
             weather_mcp, llm_mcp, automation_mcp, context_mcp,
             reminders_mcp, files_mcp, news_mcp, evolution_mcp]:
    try:
        _mcp.initialize()
    except Exception as e:
        log.warning("MCP '%s' init failed: %s", _mcp.name, e)


def _skill_dispatcher(tool: str, args: str) -> Any:
    """Dispatch pipeline skill steps to appropriate MCP modules."""
    t = tool.lower().replace("-", "_")
    if t in ("get_weather", "weather"):
        return weather_mcp.handle("weather", args)
    if t in ("get_tasks", "tasks", "list_tasks"):
        return tasks_mcp.handle("list", args)
    if t in ("get_reminders", "reminders", "list_reminders"):
        return reminders_mcp.handle("list", args)
    if t in ("get_system_status", "system", "system_info"):
        return system_mcp.handle("system", args)
    if t in ("search_web", "web_search", "search"):
        return browser_mcp.handle("search", args)
    if t in ("read_file", "file_read"):
        return files_mcp.handle("read", args)
    return f"Executed {tool} with {args}"

loop_guard = LoopGuard()
skill_manager = SkillManager(tool_dispatcher=_skill_dispatcher)
fact_extractor = FactExtractor(llm_caller=llm_mcp.ask, memory_store=memory_mcp)



# ---------------------------------------------------------------------------
# FastAPI app & globals
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(application):
    global _loop_ref
    _loop_ref = asyncio.get_running_loop()
    _shutdown.clear()
    log.info("Event loop captured.")
    metrics_thread = threading.Thread(
        target=_metrics_loop, args=(METRICS_INTERVAL_S,), daemon=True,
        name="jarvis-metrics",
    )
    metrics_thread.start()
    broadcast({"type": "status", "state": "standby"})
    try:
        yield
    finally:
        _shutdown.set()
        log.info("Shutting down background loops.")

app = FastAPI(title=APP_NAME, lifespan=lifespan)

_websocket_clients: list[WebSocket] = []
_ws_lock = threading.Lock()
_loop_ref: asyncio.AbstractEventLoop = None
_shutdown = threading.Event()
_last_error: Optional[str] = None
_last_user_cmd = ""
_jarvis_sleeping = False
_sleep_events: list[str] = []

# Background executor for LLM command routing. Used to attempt synchronous
# routing with a short timeout; slow responses fall through to async dispatch
# so the user is never blocked waiting on the network.
_routing_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="jarvis-route"
)

# The HUD's side panels are hidden until they are asked for, so the screen
# shows the neural core and nothing else. The state lives on the server so a
# reload of the tab comes back the way the user left it.
_panels_visible = False


def set_panels(visible: bool) -> None:
    """Show or hide the HUD's side panels for every connected client."""
    global _panels_visible
    _panels_visible = bool(visible)
    broadcast({"type": "panels", "visible": _panels_visible})

# Let the voice module push "speaking" events and hand back recognised speech
# without importing this app.
try:
    voice_module.set_broadcast_hook(lambda msg: broadcast(msg))
    voice_module.set_command_hook(lambda text: handle_command(text, {}))
except Exception:
    pass

# The evolution module pushes its state to the HUD when a cycle ends or a
# proposal is queued, approved or rejected.
try:
    evolution_mcp.set_broadcast(lambda msg: broadcast(msg))
except Exception:
    pass

# ---------------------------------------------------------------------------


def _broadcast_voice_down_if_needed() -> None:
    """If no TTS backends are live, alert HUD once."""
    global _voice_down_broadcasted
    try:
        if not getattr(voice_mcp, 'has_live_tts', lambda: True)():
            if not _voice_down_broadcasted:
                broadcast({"type": "warning", "level": "voice_down",
                           "message": "Voice output is down: no live TTS backend"})
                _voice_down_broadcasted = True
            return
        _voice_down_broadcasted = False
    except Exception as e:
        log.debug("Voice down check failed: %s", e)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def broadcast(msg: dict) -> None:
    """Thread-safe broadcast to all WebSocket clients."""
    if not _loop_ref:
        return
    payload = json.dumps(msg)
    with _ws_lock:
        clients = list(_websocket_clients)
    for client in clients:
        try:
            asyncio.run_coroutine_threadsafe(client.send_text(payload), _loop_ref)
        except Exception:
            pass


def set_status(status: str) -> None:
    broadcast({"type": "status", "state": status})


# Minimum gap between two spoken replies. Without this, a command that fires
# several sub-results (e.g. "act" + "result") clips its own sentences.
SPEAK_COOLDOWN_S = 0.4
_speak_lock = threading.Lock()
_last_spoken_at = 0.0


def speak(text: str) -> None:
    """Show text in the HUD and queue it for speech.

    The transcript is shown even when the audio cooldown is active: that
    cooldown exists to stop TTS queue flooding, and must not hide a reply.
    Two identical answers to two identical questions are both legitimate, so
    de-duplication belongs at the command level, not here.
    """
    global _last_spoken_at
    if not text or not text.strip():
        return
    text = text.strip()

    log.info("SPEAK: %s", text[:80])
    broadcast({"type": "transcript", "text": text, "speaker": "jarvis"})
    try:
        if _last_user_cmd:
            context_mcp.update(_last_user_cmd, text)
    except Exception:
        pass

    now = time.monotonic()
    with _speak_lock:
        if now - _last_spoken_at < SPEAK_COOLDOWN_S:
            log.debug("Speech cooldown active, skipping audio: %s", text[:60])
            return
        _last_spoken_at = now

    voice_mcp.speak(text)


# Set reminder callback so alarms/timers speak when triggered
reminders_mcp.set_callback(speak)


def _say_async(text: str) -> None:
    """Fire-and-forget speech (non-blocking)."""
    threading.Thread(target=speak, args=(text,), daemon=True).start()


def _log_action_error(e: BaseException, context: str) -> BaseException:
    """Log action failures with a traceback and a usable type.

    A bare string can be raised (e.g. `raise "nope"`), and str() alone
    would lose the class and any context. Normalise and log with traceback.
    Also handle Python's TypeError when a non-BaseException is raised.
    """
    if isinstance(e, BaseException):
        exc = e
    elif isinstance(e, TypeError) and "exceptions must derive from BaseException" in str(e):
        exc = Exception("nope")
    else:
        exc = Exception(f"{e!r}")
    log.exception("Action error in %s: %s", context, exc)
    return exc


def _speak_and_act(func, *args) -> None:
    """Execute action and speak result. Blocks until both complete."""
    try:
        result = func(*args)
        if result:
            speak(result)
    except BaseException as e:
        exc = _log_action_error(e, getattr(func, "__name__", "action"))
        speak("I encountered an error, sir.")


def _speak_stream(generator_func, *args) -> None:
    """Stream LLM response to speech for faster first-token latency."""
    try:
        voice_mcp.speak_stream(generator_func(*args))
    except BaseException as e:
        _log_action_error(e, getattr(generator_func, "__name__", "action"))
        speak("I encountered an error, sir.")


def _speak_stream_broadcast(generator_func, *args) -> None:
    """Stream LLM response to speech AND broadcast each sentence to the UI."""
    _stream_started = threading.Event()
    collected_sentences: list[str] = []
    def _on_sentence(sentence):
        if sentence:
            collected_sentences.append(sentence)
        broadcast({
            "type": "transcript",
            "text": sentence,
            "speaker": "jarvis",
            "append": _stream_started.is_set(),
        })
        _stream_started.set()
    try:
        voice_mcp.speak_stream(generator_func(*args), on_sentence=_on_sentence)
        if collected_sentences and len(args) > 0 and isinstance(args[0], str):
            fact_extractor.process_turn_async(args[0], " ".join(collected_sentences))
    except BaseException as e:
        _log_action_error(e, getattr(generator_func, "__name__", "action"))
        speak("I encountered an error, sir.")


# How long an action gets to produce its own answer before it is announced as
# pending instead. A report answers in milliseconds and the answer is the useful
# part, so announcing first would bury it; opening an app takes seconds and
# needs the acknowledgement straight away.
ACTION_ANSWER_GRACE_S = 0.3

# Reports can be long - thirty window titles, twenty processes. Reading all of it
# aloud is worse than useless, so the spoken result is capped. The HUD receives
# the full text regardless.
ACTION_RESULT_MAX_CHARS = 400


def _speak_during_act(func, speak_text: str, *args) -> None:
    """Start action and speech at the SAME time. Both non-blocking.

    The action's own return value is spoken as well. Most of the actions routed
    through here are reports - system status, task and window lists, the
    clipboard - and discarding what they returned meant the user heard "Checking
    system status..." and never the answer, which was the only part that said
    anything.
    """
    lock = threading.Lock()
    spoken = {"result": False, "ack": False}

    def _ack_once() -> None:
        with lock:
            if spoken["ack"]:
                return
            spoken["ack"] = True
        speak(speak_text)

    def _speak_result(result) -> None:
        text = result.strip() if isinstance(result, str) else ""
        if not text:
            _ack_once()
            return
        with lock:
            if spoken["result"]:
                return
            spoken["result"] = True
        if len(text) > ACTION_RESULT_MAX_CHARS:
            text = text[:ACTION_RESULT_MAX_CHARS].rstrip() + "... sir, that is the summary."
        speak(text)

    def _do() -> None:
        try:
            try:
                result = func(*args)
            except TypeError as te:
                if "exceptions must derive from BaseException" in str(te):
                    raise Exception("nope") from te
                raise
        except BaseException as e:
            exc = _log_action_error(e, getattr(func, "__name__", "action"))
            _record_error(exc, getattr(func, "__name__", "action"))
            _try_self_heal(str(exc), getattr(func, "__name__", "action"))
            _ack_once()
            return
        _speak_result(result)

    def _announce_if_slow() -> None:
        """Acknowledge an action that is still running when the grace period ends.

        Yields to a result that has already been spoken, so a fast report is not
        announced after the fact.
        """
        time.sleep(ACTION_ANSWER_GRACE_S)
        with lock:
            if spoken["result"]:
                return
        _ack_once()

    threading.Thread(target=_do, daemon=True).start()
    threading.Thread(target=_announce_if_slow, daemon=True).start()


def _task_action(func, speak_text: str, *args) -> None:
    """Run a task mutation and push the new list straight away.

    Without this the HUD only refreshes on the next metrics tick, so an
    "Added" reply can appear before the task it refers to.
    """
    def _do():
        try:
            try:
                func(*args)
            except TypeError as te:
                if "exceptions must derive from BaseException" in str(te):
                    raise Exception("nope") from te
                raise
        except BaseException as e:
            exc = _log_action_error(e, getattr(func, "__name__", "action"))
            _record_error(exc, getattr(func, "__name__", "action"))
            _try_self_heal(str(exc), getattr(func, "__name__", "action"))
        finally:
            broadcast_tasks()
    threading.Thread(target=_do, daemon=True).start()
    speak(speak_text)


def _record_error(exc: BaseException, context: str = "") -> None:
    """Remember the most recent failure so "fix the error" has something to work on."""
    global _last_error
    _last_error = f"{type(exc).__name__}: {exc}" + (f" (in {context})" if context else "")
    log.debug("Recorded error: %s", _last_error)


def _record_sleep_event(text: str) -> None:
    """Log something that happened while Jarvis was in standby, for the wake-up brief."""
    global _sleep_events
    if len(_sleep_events) >= 50:
        _sleep_events.pop(0)
    _sleep_events.append(text)


def _try_self_heal(error_msg: str, context: str = "") -> bool:
    """Ask the LLM for a remediation and report it back to the user.

    JARVIS deliberately does not execute suggested fixes: an LLM-suggested
    shell command is not something to run unattended. Instead the suggestion
    is surfaced so a human can decide. Returns True if a usable suggestion
    was produced.
    """
    if not llm_mcp.available:
        log.info("Self-heal unavailable: %s", llm_mcp.get_status().get("error"))
        return False

    try:
        fix_prompt = (
            "An error occurred in a local assistant. Reply with a single short "
            "line naming the likely root cause and the concrete fix. No shell "
            "commands, no code blocks. If you cannot diagnose it, reply NONE.\n\n"
            f"Error: {error_msg}\n"
            f"Location: {context or 'unknown'}"
        )
        response = llm_mcp.ask(fix_prompt, use_history=False, skip_cooldown=True)
        if not response or "NONE" in response.upper()[:40]:
            return False
        suggestion = response.strip()
        log.info("Self-heal suggestion for %r: %s", error_msg, suggestion)
        speak(f"I found the likely cause, sir. {suggestion} "
              "I have not applied it - that one needs your say-so.")
        return True
    except Exception as e:
        log.error("Self-heal error: %s", e)
        return False


def _handle_fix_error(cmd: str) -> None:
    """Handle the 'fix error' command."""
    if _last_error:
        speak(f"Analyzing the last error, sir: {_last_error}")
        if not _try_self_heal(_last_error, "user_requested_fix"):
            speak("I could not diagnose that automatically, sir.")
    else:
        speak("No recent errors to fix, sir.")


def _handle_run_skill(skill_name: str) -> None:
    """Execute a skill from the skill manager."""
    name = (skill_name or "").strip().lower().replace(" ", "_")
    if not name:
        speak("Which skill would you like me to run, sir?")
        return
    res = skill_manager.execute_skill(name)
    if not res.get("ok"):
        speak(f"I could not run {name}, sir: {res.get('error', 'unknown error')}")
        return
    if res.get("type") == "pipeline":
        speak(f"Skill {name} completed successfully, sir.")
    else:
        desc = res.get("description") or "instructions loaded"
        speak(f"Skill {name} activated, sir: {desc}")


def _handle_self_review() -> None:
    """Report on the assistant's own code and behaviour. Runs in the background."""
    try:
        report = evolution_mcp.handle("self review", "")
        speak(report or "I looked myself over, sir.")
    except Exception as e:
        log.error("Self-review failed: %s", e)
        speak("I could not complete the review, sir.")


def _handle_evolve() -> None:
    """Run one self-evolution cycle. Runs in the background."""
    try:
        result = evolution_mcp.run_cycle(trigger="command")
        if result.get("ok"):
            speak(f"I have a change ready for your approval, sir: "
                  f"{result.get('summary', '')}")
        else:
            speak(f"I looked myself over, sir. {result.get('reason', 'Nothing to do.')}")
    except Exception as e:
        log.error("Evolution cycle failed: %s", e)
        speak("I could not complete the review, sir.")


def _clean_news(raw: str) -> str:
    """Clean news artifacts from raw news data."""
    import re
    if not raw:
        return raw
    # Normalize both formats
    cleaned = re.sub(r'News about[^:]*:\s*', '', raw)
    cleaned = re.sub(r'Headlines:\s*', '', cleaned)
    cleaned = cleaned.replace(';', '\n')
    # Split into individual headlines by numbered items
    items = re.split(r'(?<!\d)\d+\.\s*', cleaned)
    cleaned_items = []
    source_pattern = r'^(BBC News|BBC Sport|BBC|CNN|ESPN|The Guardian|The New York Times|FOX|FOX Sports|Yahoo Sports|Yahoo|USA Today|AP|Reuters|Sky Sports|Audacy|Google News)\s*$'
    for item in items:
        item = item.strip().strip('.')
        if not item:
            continue
        if re.match(source_pattern, item, re.IGNORECASE):
            continue
        if item == 'Google News':
            continue
        # Remove trailing source suffixes
        item = re.sub(r'\s*[-\u2013]\s*(BBC|CNN|ESPN|The Guardian|The New York Times|FOX|FOX Sports|Yahoo Sports|Yahoo|USA Today|AP|Reuters|Sky Sports|Audacy)\s*$', '', item)
        # Remove headline artifacts
        item = re.sub(r'\bLIVE:\s*', '', item, flags=re.IGNORECASE)
        item = re.sub(r'as it happened', '', item, flags=re.IGNORECASE)
        item = re.sub(r'Live updates and reaction', '', item, flags=re.IGNORECASE)
        item = re.sub(r'Highlights\s*\d{4}[^.]*\|[^.]*', '', item, flags=re.IGNORECASE)
        item = re.sub(r'\bWatch FIFA World Cup stream, penalties, score, commentary & updates\b', '', item, flags=re.IGNORECASE)
        item = re.sub(r'\bstream, penalties, score, commentary & updates\b', '', item, flags=re.IGNORECASE)
        item = re.sub(r'\bFIFA World Cup round of 32 score, commentary & updates\b', '', item, flags=re.IGNORECASE)
        item = re.sub(r'\bBest Moments\s*\d{4}[^.]*', '', item, flags=re.IGNORECASE)
        item = re.sub(r'\bVideo:\s*[^.]*', '', item, flags=re.IGNORECASE)
        item = re.sub(r'[:;,]\s*$', '', item).strip().strip('.')
        item = item.strip()
        if item and len(item) > 5:
            cleaned_items.append(item)
    if cleaned_items:
        return '. '.join(cleaned_items[:3])
    return raw


def _fetch_and_respond(fetch_type: str, payload: str) -> None:
    """Fetch data in background and respond."""
    try:
        if fetch_type == "weather":
            resp = weather_mcp.get_weather(payload)
        elif fetch_type == "news":
            if payload:
                resp = news_mcp.get_news_by_topic(payload)
            else:
                resp = news_mcp.get_news()
        elif fetch_type == "tech_news":
            resp = news_mcp.get_news("technology")
        elif fetch_type == "sports_news":
            if payload:
                resp = news_mcp.get_news_by_topic(payload)
            else:
                resp = news_mcp.get_news("sports")
        else:
            resp = "Unknown fetch type."
        
        # Clean news artifacts
        if fetch_type in ("news", "tech_news", "sports_news") and resp:
            resp = _clean_news(resp)
        
        speak(resp)
    except Exception as e:
        log.error("Background fetch error: %s", e)
        speak("I encountered an error fetching that data, sir.")


def is_hud_window_title(title: str) -> bool:
    """True when a browser window title looks like the HUD.

    The HUD is served from index.html, whose <title> is the assistant's name,
    so this has to track APP_NAME. The legacy name is still accepted so a
    tab opened before a rename keeps working, and the dev-server hostnames
    cover a page loaded before the title was set.
    """
    lowered = title.lower()
    if "chrome" not in lowered:
        return False
    return any(marker in lowered for marker in
               ("jarvis", "j.a.r.v.i.s", "localhost", "127.0.0.1"))


def _ask_llm_stream(cmd: str, memory: dict):
    """Stream LLM response for faster first-token latency."""
    try:
        identity = memory_mcp.get_identity() if hasattr(memory_mcp, 'get_identity') else {}
        facts = memory_mcp.get_facts_for_prompt() if hasattr(memory_mcp, 'get_facts_for_prompt') else ""
        tasks = tasks_mcp.get_tasks_for_prompt() if hasattr(tasks_mcp, 'get_tasks_for_prompt') else ""
        ctx = context_mcp.get_context_string() if hasattr(context_mcp, 'get_context_string') else ""
        sys_prompt = llm_mcp.build_system_prompt(identity, facts, tasks, ctx)
        yield from llm_mcp.ask_stream(cmd, system_prompt=sys_prompt)
    except Exception as e:
        log.error("LLM stream error: %s", e)
        yield "I encountered an error processing that request, sir."


def _ask_llm(cmd: str, memory: dict) -> str:
    try:
        identity = memory_mcp.get_identity() if hasattr(memory_mcp, 'get_identity') else {}
        facts = memory_mcp.get_facts_for_prompt() if hasattr(memory_mcp, 'get_facts_for_prompt') else ""
        tasks = tasks_mcp.get_tasks_for_prompt() if hasattr(tasks_mcp, 'get_tasks_for_prompt') else ""
        ctx = context_mcp.get_context_string() if hasattr(context_mcp, 'get_context_string') else ""
        sys_prompt = llm_mcp.build_system_prompt(identity, facts, tasks, ctx)
        return llm_mcp.ask(cmd, system_prompt=sys_prompt)
    except Exception as e:
        log.error("LLM error: %s", e)
        return "I encountered an error processing that request, sir."

# ---------------------------------------------------------------------------
# Fuzzy matching for voice recognition errors
# ---------------------------------------------------------------------------

# Common mishearings and synonyms
FUZZY_COMMANDS = {
    # Play/Pause
    "pose": "pause",
    "paws": "pause",
    "pouse": "pause",
    "play": "play",
    "plai": "play",
    "pley": "play",
    "resma": "resume",
    "rezoom": "resume",
    
    # Next/Previous
    "nex": "next",
    "necks": "next",
    "nect": "next",
    "previus": "previous",
    "previs": "previous",
    "previos": "previous",
    "bak": "back",
    "bck": "back",
    
    # Volume
    "vlume": "volume",
    "volme": "volume",
    "vloum": "volume",
    "wp": "up",
    "upp": "up",
    "down": "down",
    "don": "down",
    "dwn": "down",
    "mut": "mute",
    "mat": "mute",
    "moot": "mute",
    
    # Open/Close
    "opn": "open",
    "open": "open",
    "cloe": "close",
    "cls": "close",
    
    # Spotify
    "spootify": "spotify",
    "spotfiy": "spotify",
    "spottify": "spotify",
    "spooty": "spotify",
    
    # YouTube
    "youtub": "youtube",
    "youtbe": "youtube",
    "utube": "youtube",
    
    # Weather
    "wether": "weather",
    "wheather": "weather",
    "weathr": "weather",
    
    # Time
    "tim": "time",
    "tyme": "time",
    "tyem": "time",
    
    # Music
    "musik": "music",
    "musc": "music",
    "muzic": "music",
    "song": "song",
    "ongs": "song",
    "tune": "tune",
    "tunes": "tunes",
    
    # System / power
    "sleap": "sleep",
    "slepp": "sleep",
    "slepe": "sleep",
    "wak": "wake",
    "wakeup": "wake up",
    "shudown": "shutdown",
    "shudn": "shutdown",
    "reboot": "restart",
    "re boot": "restart",
    "hibrnat": "hibernate",
    
    # Reminders / tasks
    "remind": "reminder",
    "reminder": "reminder",
    "reminde": "reminder",
    "alarm": "alarm",
    "alar": "alarm",
    "timer": "timer",
    "timr": "timer",
    "task": "task",
    "tas": "task",
    "todo": "task",
    
    # Navigation / apps
    "notepad": "notepad",
    "calc": "calculator",
    "calculater": "calculator",
    "browser": "browser",
    "chrome": "chrome",
    "edge": "edge",
    
    # Common mishearings
    "please": "please",
    "pls": "please",
    "plz": "please",
    "sir": "sir",
    "ser": "sir",
    "jarvis": "jarvis",
    "jarvus": "jarvis",
    "jarvis": "jarvis",
}


# Common words that are never worth "correcting" - they carry no command
# intent, so repairing them only adds noise.
COMMON_WORDS = {
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'some', 'any', 'no', 'every',
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
    'shall', 'must', 'need', 'dare', 'ought', 'used',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between', 'out',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'each', 'both', 'few', 'more',
    'most', 'other', 'such', 'than', 'too', 'very', 'just', 'because',
    'but', 'and', 'or', 'if', 'while', 'although', 'though', 'since', 'unless',
    'until', 'whether', 'whereas', 'what', 'which', 'who', 'whom', 'whose',
    'up', 'down', 'back', 'away', 'forward',
    'please', 'could', 'set', 'it', 'also', 'much', 'well', 'right', 'now',
    'go', 'open', 'play', 'pause', 'stop', 'start', 'run', 'make', 'turn', 'say',
    'browser', 'jarvis', 'ameena', 'dashboard', 'page', 'tab', 'window',
    'cost', 'trip', 'road', 'price', 'many', 'about', 'know',
    'like', 'want', 'tell', 'give', 'show', 'find', 'search', 'two',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '10', '20', '25', '30', '40', '50', '60', '70', '75', '80', '90', '100',
}

# Fuzzy repair is only allowed to touch words that are long enough to be a
# plausible mishearing of a command word, and only a couple of words at a time.
FUZZY_MIN_WORD_LEN = 4
FUZZY_MAX_CHANGES = 2

# Words that carry no command intent, so they cannot "anchor" a fuzzy repair.
_FUZZY_NOISE = frozenset({
    "the", "and", "for", "with", "that", "this", "please", "can", "you",
    "your", "his", "her", "its", "our", "their", "there", "then", "than",
    "some", "any", "all", "not", "but", "are", "was", "were", "been",
    "have", "has", "had", "will", "would", "could", "should", "does",
    "did", "just", "very", "too", "also", "into", "onto", "over", "under",
})


def _fuzzy_correct_words(text: str, threshold: int = 1) -> list[tuple[str, bool]]:
    """Correct likely mishearings word-by-word.

    Returns a list of ``(word, was_changed)`` pairs. Words shorter than
    ``FUZZY_MIN_WORD_LEN`` are left alone: correcting them is what used to
    turn ordinary speech ("a man") into a destructive command ("a mute").
    """
    corrected: list[tuple[str, bool]] = []
    changes = 0
    for word in text.split():
        if changes >= FUZZY_MAX_CHANGES or len(word) < FUZZY_MIN_WORD_LEN or word in COMMON_WORDS:
            corrected.append((word, False))
            continue

        best_word, best_dist = word, threshold + 1
        for fuzzy, correct in FUZZY_COMMANDS.items():
            distance = _edit_distance(word, fuzzy)
            if distance < best_dist and distance <= threshold:
                best_dist, best_word = distance, correct

        if best_word != word:
            changes += 1
            corrected.append((best_word, True))
        else:
            corrected.append((word, False))
    return corrected


def _fuzzy_repair_is_trustworthy(
    original: str, corrected_pairs: list[tuple[str, bool]], matched_text: str
) -> bool:
    """Decide whether a fuzzy-repaired command is safe to execute.

    A repair is only trusted when at least one *unchanged* word of what the
    user actually said survives into the matched pattern. That anchor is what
    separates a real mishearing ("wether in london" -> "weather in london")
    from a hallucinated command ("man" -> "mute"). Without it, almost any
    unmatched sentence mutates into whatever command happens to be one edit
    away.
    """
    matched = matched_text.lower()
    for word, was_changed in corrected_pairs:
        if was_changed or len(word) < 3 or word in _FUZZY_NOISE:
            continue
        if word in matched:
            return True
    return False


def fuzzy_match(text: str, threshold: int = 1) -> str:
    """Repair likely speech-recognition errors in a short command phrase."""
    text = text.lower().strip()
    if not text:
        return text

    # A bare word that is already an exact command keyword stays as-is.
    for word, correct in FUZZY_COMMANDS.items():
        if word == text:
            return correct

    pairs = _fuzzy_correct_words(text, threshold)
    return " ".join(word for word, _ in pairs)


def _edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


# ---------------------------------------------------------------------------
# Command detection
# ---------------------------------------------------------------------------

_ACTION_PATTERNS = [
    # Time - flexible
    (r"^what\s+time(?:\s+is\s+it)?$", "time"),
    (r"^what(?:'s|\s+is)\s+(?:the\s+)?time(?:\s+is\s+it)?$", "time"),
    (r"^time$", "time"),
    (r"^what\s+(?:'s|\s*is)\s+(?:the\s+)?date$", "time"),
    (r"^date$", "time"),
    (r"^what\s+(?:'s|\s*is)\s+(?:the\s+)?day$", "time"),
    (r"^day$", "time"),
    # Greetings - flexible
    (r"^(?:good\s*(?:morning|afternoon|evening|night)|hello|hi|hey|greetings|sup|yo|howdy)$", "greeting"),
    (r"^(?:hello|hi|hey|yo)\s+(?:jarvis|sir|there)$", "greeting"),
    (r"^(?:jarvis)\s+(?:are\s+you\s+)?(?:up|there|awake|alive|online|ready|working)$", "greeting"),
    (r"^(?:jarvis)\s+(?:hello|hi|hey)$", "greeting"),
    (r"^(?:how(?:'s| is) (?:it going|everything|you doing)|how are you|how do you do|you doing (?:okay|good|well|alright))$", "greeting"),
    (r"^(?:jarvis)\s+how(?:'s| is) (?:it going|everything|you doing)$", "greeting"),
    (r"^what(?:'s|\s+is)\s+(?:up|going\s+on|happening)$", "greeting"),
    (r"^(?:thank(?:s|\s?you)|thx|ty|cheers|appreciate)$", "thanks"),
    (r"^(?:who are you|about you|your(?:s|self)?|what are you)$", "about"),
    (r"^tell\s+me\s+(?:about\s+)?(?:you|your(?:s|self)?)$", "about"),
    (r"^what\s+can\s+you\s+do$", "about"),
    (r"^(?:cancel|never\s?mind|forget\s?(?:it|that)|nvm)$", "cancel"),
    # --- HUD panels. The side panels are hidden until they are asked for, so
    # --- these have to sit above the generic "show ..." rules further down,
    # --- which would otherwise read "show panels" as the name of a file.
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:show|display|open|bring\s+up)\s+(?:me\s+)?(?:the\s+)?(?:side\s+|side\s+?bar\s+)?panels?$", "show_panels"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:show|display|open|bring\s+up)\s+(?:me\s+)?(?:the\s+)?(?:sidebar|side\s+bar|dashboard|system\s+metrics|metrics|readouts?|readings?|everything)$", "show_panels"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:hide|close|dismiss|collapse|get\s+rid\s+of)\s+(?:me\s+)?(?:the\s+)?(?:side\s+)?panels?$", "hide_panels"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:hide|close|dismiss|collapse|get\s+rid\s+of)\s+(?:me\s+)?(?:the\s+)?(?:sidebar|side\s+bar|dashboard|system\s+metrics|metrics|readouts?|readings?|everything)$", "hide_panels"),
    # --- Self-evolution. These sit with the panel rules rather than at the end
    # --- of the table, because the generic "list ..." and "show ..." rules
    # --- further down would otherwise read "list proposals" as a request to
    # --- list a file called "proposals".
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:self[\s-]?review|review\s+yourself|analy[sz]e\s+yourself|check\s+yourself)$", "self_review"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:evolve|improve\s+yourself|fix\s+yourself|self[\s-]?improve(?:ment)?)$", "evolve"),
    (r"^(?:evolution|self[\s-]?evolution)\s+(?:status|report|summary)$", "evolution_status"),
    (r"^(?:show|list|view)\s+(?:me\s+)?(?:the\s+)?(?:evolution\s+)?proposals?$", "show_proposals"),
    (r"^(?:show|list|view)\s+(?:me\s+)?(?:the\s+)?(?:pending\s+)?(?:changes|evolution)$", "show_proposals"),
    (r"^(?:approve|accept|apply)\s+(?:the\s+)?(?:evolution|change|proposal)\s+(.+)$", "approve_evolution"),
    (r"^(?:reject|decline|refuse)\s+(?:the\s+)?(?:evolution|change|proposal)\s+(.+)$", "reject_evolution"),
    (r"^(?:revert|undo|roll\s*back)\s+(?:the\s+)?(?:evolution|change|proposal)\s+(.+)$", "revert_evolution"),
    # Power commands. These used to share one pattern that mapped "shut down"
    # and "restart" to *sleep*, so each verb now has its own target.
    # Spotify - must come BEFORE generic "open" and "play" to match correctly.
    # The catch-all "spotify ..." rule is placed AFTER the spotify_volume_*
    # block further down, otherwise "spotify volume to 70" reads as a song
    # request for "volume to 70".
    (r"^(?:can\s+you\s+)?(?:please\s+)?play\s+(.+)\s+(?:on|in|through)\s+spotify$", "spotify"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?open\s+spotify(?:\s+(?:and|then)\s+play\s+(.+))?$", "spotify"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?open\s+spotify(?:\s+(?:for\s+)?(.+))?$", "spotify"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?play\s+(?:some\s+)?(?:music|songs?|tunes?)$", "spotify_random"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?shuffle\s+(?:some\s+)?(?:music|songs?|tunes?)$", "spotify_random"),
    # Spotify controls
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:play|resume|start|begin|continue)(?:\s+(?:the\s+)?(?:music|song|track|video|audio))?$", "spotify_play"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:pause|stop|halt|freeze|hold|wait)(?:\s+(?:the\s+)?(?:music|song|track|video|audio|playing))?$", "spotify_pause"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:next|skip|forward)(?:\s+(?:track|song|video))?$", "spotify_next"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:previous|back|prev|rewind|go\s+back)(?:\s+(?:track|song|video))?$", "spotify_previous"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?volume\s+up(?:\s+(?:a\s+)?(?:bit|little|lot|more|slightly))?$", "spotify_volume_up"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?volume\s+down(?:\s+(?:a\s+)?(?:bit|little|lot|more|slightly))?$", "spotify_volume_down"),
    # Media mute requires an explicit media/system target, so a bare "mute" is
    # left to the voice controls further down.
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:mute|silence|quiet|shut\s+up)\s+(?:the\s+)?(?:music|song|track|media|player|spotify|volume|sound|audio)$", "spotify_mute"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:unmute|unsilence)\s+(?:the\s+)?(?:music|song|track|media|player|spotify|volume|sound|audio)$", "spotify_volume_up"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:toggle\s+|shuffle|random(?:ize)?|mix)(?:\s+(?:the\s+)?(?:music|songs?|track))?$", "spotify_shuffle"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:toggle\s+|repeat|loop)(?:\s+(?:the\s+)?(?:music|songs?|track))?$", "spotify_repeat"),
    # Back to browser - just focus existing tab
    (r"^(?:go\s+)?back\s+to\s+(?:the\s+)?(?:browser|jarvis|dashboard|ui|frontend|interface|tab|page|window)$", "focus_jarvis"),
    # Spotify volume: only with an explicit "spotify" qualifier. Bare numeric
    # volume is the system mixer and is handled further down.
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:set|change|adjust)\s+spotify\s+volume\s+(?:to\s+)?(\d{1,3})\s*(?:%|percent|percentage)?$", "spotify_volume_set"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?spotify\s+volume\s+(?:to\s+)?(\d{1,3})\s*(?:%|percent|percentage)?$", "spotify_volume_set"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?spotify\s+volume\s+to\s+(?:the\s+)?(?:maximum|max|full|loudest)$", "spotify_volume_max"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?spotify\s+volume\s+to\s+(?:the\s+)?(?:minimum|min|lowest|quietest|softest)$", "spotify_volume_min"),
    # Up and down must be separate rules: a shared "(?:up|down)" pattern
    # mapped "spotify volume down" to spotify_volume_up.
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:turn|put)?\s*spotify\s+volume\s+up(?:\s+(?:a\s+)?(?:bit|little|lot|more|slightly))?$", "spotify_volume_up"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:turn|put)?\s*spotify\s+volume\s+down(?:\s+(?:a\s+)?(?:bit|little|lot|more|slightly))?$", "spotify_volume_down"),
    # "increase spotify volume" and "increase the volume" are the same intent.
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:increase|raise|boost)\s+(?:the\s+)?(?:spotify\s+)?volume(?:\s+(?:to\s+)?(\d{1,3})\s*(?:%|percent|percentage)?)?$", "spotify_volume_up"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:decrease|lower|reduce)\s+(?:the\s+)?(?:spotify\s+)?volume(?:\s+(?:to\s+)?(\d{1,3})\s*(?:%|percent|percentage)?)?$", "spotify_volume_down"),
    # Catch-all "spotify <something>". Must stay after every spotify_volume_*
    # rule so volume control is not parsed as a song request.
    (r"^(?:can\s+you\s+)?(?:please\s+)?spotify\s+(?:for\s+)?(.+)$", "spotify"),
    # --- Specific "open X" targets. These MUST come before the generic
    # --- "open (.+)" rule below, otherwise "open file x" launches an app.
    (r"^(?:can\s+you\s+)?(?:please\s+)?check\s+(?:my\s+)?(?:mail|email|inbox|gmail)$", "gmail"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?open\s+(?:my\s+)?(?:mail|gmail|inbox)$", "gmail"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?open\s+(?:google\s+)?maps$", "maps"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?show\s+(?:me\s+)?(?:the\s+)?map(?:\s+(?:of|for)\s+(.+))?$", "maps"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?open\s+(?:the\s+)?folder\s+(.+)$", "open_folder"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?open\s+(?:the\s+)?(?:file|document|doc)\s+(.+)$", "open_file"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:read|cat|show|display)\s+(?:me\s+)?(?:the\s+)?(?:file|document)\s+(.+)$", "read_file"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:show|open)\s+(?:me\s+)?(?:the\s+)?contents\s+of\s+(?:the\s+)?(?:file\s+)?(.+)$", "read_file"),
    # --- Media transport. "start timer/reminder" is claimed here, before the
    # --- generic "start (.+)" launcher rule further down.
    (r"^(?:can\s+you\s+)?(?:please\s+)?start\s+(?:an?\s+)?(?:timer|countdown)\s*(?:for\s+)?(.+)$", "set_timer"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?start\s+(?:an?\s+)?(?:reminder|alarm)\s*(?:for\s+)?(.+)$", "set_reminder"),
    # --- Web navigation. Payload must look like a real URL/domain so that
    # --- "go to the gym" is not treated as a website request. These MUST come
    # --- before the generic "open (.+)" launcher below, otherwise
    # --- "open github.com" tries to launch an app called "github.com".
    (r"^(?:can\s+you\s+)?(?:please\s+)?browse\s+((?:https?://\S+|(?:[\w-]+\.)+[a-z]{2,}(?:/\S*)?))$", "open_website"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?go\s+to\s+((?:https?://\S+|(?:[\w-]+\.)+[a-z]{2,}(?:/\S*)?))$", "open_website"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?open\s+((?:https?://\S+|(?:[\w-]+\.)+[a-z]{2,}(?:/\S*)?))$", "open_website"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?open\s+(?:the\s+)?(?:website|site|url|link|page)\s+((?:https?://\S+|(?:[\w-]+\.)+[a-z]{2,}(?:/\S*)?))$", "open_website"),
    # --- Generic launchers (now safe: specifics already matched above)
    (r"^(?:can\s+you\s+)?(?:please\s+)?open\s+(.+)$", "open_app"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?launch\s+(.+)$", "open_app"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?run\s+(?:the\s+)?app\s+(.+)$", "open_app"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?start\s+(?:the\s+)?app(?:lication)?\s+(.+)$", "open_app"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?start\s+(.+)$", "open_app"),
    # Search - flexible. "search for files X" is a disk search, not a web
    # search, so it has to be claimed before the generic "search" rule.
    (r"^search\s+(?:for\s+)?(?:the\s+)?(?:file|files|document|documents)\s+(.+)$", "find_file"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?search\s+(?:for\s+)?(.+)$", "search"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?google\s+(.+)$", "search"),
    # YouTube - must come BEFORE generic "play"
    (r"^(?:can\s+you\s+)?(?:please\s+)?youtube\s+(?:for\s+)?(.+)$", "youtube"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?play\s+(.+)\s+(?:on|in|through)\s+youtube$", "youtube"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?play\s+(.+)$", "youtube"),
    # Brightness - flexible
    (r"^(?:can\s+you\s+)?(?:please\s+)?set\s+(?:the\s+)?brightness\s+(?:to\s+)?(\d+)$", "brightness"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:the\s+)?brightness\s+(?:to\s+)?(\d+)$", "brightness"),
    # --- System volume. Numeric / max / min level = the machine's volume.
    # --- Relative "up"/"down" is handled above as a media control.
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:set|change|adjust|turn|make)\s+(?:the\s+)?(?:system\s+|master\s+)?volume\s+(?:to\s+)?(\d{1,3})\s*(?:%|percent|percentage)?$", "volume"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:the\s+)?volume\s+(?:to\s+)?(\d{1,3})\s*(?:%|percent|percentage)?$", "volume"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:set|change|adjust|turn|make)\s+(?:the\s+)?(?:system\s+|master\s+)?volume\s+to\s+(?:the\s+)?(?:maximum|max|full|loudest)$", "volume"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:set|change|adjust|turn|make)\s+(?:the\s+)?(?:system\s+|master\s+)?volume\s+to\s+(?:the\s+)?(?:minimum|min|lowest|quietest|softest)$", "volume"),
    # Memory - flexible
    (r"^(?:can\s+you\s+)?(?:please\s+)?remember\s+(.+)$", "remember"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?recall\s+(.+)$", "recall"),
    (r"^what\s+(?:do\s+)?(?:i|you)\s+(?:know|have)\s+(?:about|of|on)\s+(.+)$", "recall"),
    # Standby / brief
    (r"^(?:can\s+you\s+)?(?:please\s+)?brief\s+(?:me)?(?:\s+on\s+(.+))?$", "brief"),
    (r"^what\s+(?:happened|went\s+on)(?:\s+while\s+(?:i\s+was\s+)?(?:away|gone|asleep))?$", "brief"),
    # Tasks - flexible
    (r"^(?:can\s+you\s+)?(?:please\s+)?add\s+task\s*[:\-]?\s*(.+)$", "add_task"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:add|create)\s+(?:a\s+)?task\s+(.+)$", "add_task"),
    (r"^(?:task|todo|to\s?do)\s+(?:add|create)\s+(.+)$", "add_task"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?complete\s+task\s+(\d+)$", "complete_task"),
    (r"^task\s+(\d+)\s+(?:done|complete|finished)$", "complete_task"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?delete\s+task\s+(\d+)$", "delete_task"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?remove\s+task\s+(\d+)$", "delete_task"),
    (r"^task\s+(\d+)\s+(?:delete|remove)$", "delete_task"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?delete\s+tasks?\s+(?:number\s+)?([\d\s,]+)$", "delete_multiple_tasks"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?clear\s+(?:all\s+)?tasks$", "clear_all_tasks"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?delete\s+all\s+tasks$", "clear_all_tasks"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?remove\s+all\s+tasks$", "clear_all_tasks"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?delete\s+task\s+(?:named?|called?|with\s+description)\s+(.+)$", "delete_task_by_desc"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?remove\s+task\s+(?:named?|called?|with\s+description)\s+(.+)$", "delete_task_by_desc"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:show|list|display|what(?:'s| are))\s+(?:my\s+)?(?:tasks?|todos?|to-dos?)$", "list_tasks"),
    (r"^(?:my\s+)?(?:tasks?|todos?|to-dos?)$", "list_tasks"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?clear\s+(?:completed|done)\s+tasks?$", "clear_tasks"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?remove\s+(?:completed|done)\s+tasks?$", "clear_tasks"),
    # System - flexible
    (r"^(?:can\s+you\s+)?(?:please\s+)?system\s+(?:info|status|stats|info)$", "system_info"),
    (r"^what(?:'s| is) (?:my )?(?:pc|computer|system|laptop) (?:status|info|specs?)$", "system_info"),
    (r"^(?:pc|computer|system|laptop) (?:status|info|specs?)$", "system_info"),
    (r"^what(?:'s| is) (?:my )?(?:pc|computer|system|laptop) (?:doing|running|using)$", "running_apps"),
    (r"^what(?:'s| is| are)?\s+(?:the\s+)?(?:running|active|open)(?:\s+(?:apps?|programs?|process(?:es)?))?$", "running_apps"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:show|what(?:'s| are)) (?:the )?(?:running|active|open) (?:apps?|programs?)$", "running_apps"),
    # Bare "running apps" / "active programs" with no leading question word.
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:the\s+)?(?:running|active|open)\s+(?:apps?|programs?)$", "running_apps"),
    # Individual hardware metrics, phrased the way people actually speak.
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:how\s+(?:much|many)\s+)?(?:cpu|processor)(?:\s+(?:usage|use|load|status|info))?$", "cpu_info"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:how\s+(?:much|many)\s+)?(?:ram|memory)(?:\s+(?:usage|use|status|info))?$", "ram_info"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?battery(?:\s+(?:status|level|charge))?$", "battery_info"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:disk|storage|drive)(?:\s+(?:usage|use|space|status|info))?$", "disk_info"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?network(?:\s+(?:usage|use|status|info|speed))?$", "network_info"),
    # Execute - flexible
    (r"^(?:can\s+you\s+)?(?:please\s+)?run\s+(.+)$", "execute"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:execute|terminal|shell|command)\s+(.+)$", "execute"),
    # File operations
    (r"^(?:can\s+you\s+)?(?:please\s+)?create\s+(?:a\s+)?file\s+(?:at\s+)?(.+)$", "create_file"),
    # "write a file called x" / "write a new file at x" create a file; the
    # bare "write ..." rule further down types the text instead.
    (r"^(?:can\s+you\s+)?(?:please\s+)?write\s+(?:a\s+)?(?:new\s+|empty\s+)?file\s+(?:at|called|named)\s+(.+)$", "create_file"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:read|open|show|cat)\s+(?:the\s+)?file\s+(.+)$", "read_file"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?delete\s+(?:the\s+)?file\s+(.+)$", "delete_file"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?copy\s+(.+)$", "copy_file"),
    # Negative lookahead so "move mouse to 100 200" is not read as a file move.
    (r"^(?:can\s+you\s+)?(?:please\s+)?move\s+(?!mouse\b)(.+)$", "move_file"),
    # "list voices" must be claimed before the "list ..." directory rule.
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:list|show)\s+(?:the\s+)?(?:available\s+)?voices?$", "list_voices"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:ls|dir|list)\s*(?:folder|directory)?\s*(?!\s*windows?\b)(.+)?$", "list_directory"),
    # Window management (US and UK spellings)
    (r"^(?:can\s+you\s+)?(?:please\s+)?mini(?:mi[sz]e|mize)\s+(?:the\s+)?(?:window|app|this)?$", "minimize_window"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?maxi(?:mi[sz]e|mize)\s+(?:the\s+)?(?:window|app|this)?$", "maximize_window"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:no\s+)?close\s+(?:the\s+)?(?:window|app|this)$", "close_window"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:no\s+)?close\s+(?:the\s+)?(?:browser\s+)?tab$", "close_tab"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:no\s+)?close\s+this\s+tab$", "close_tab"),
    # Closing an app by name. The verb list deliberately excludes "stop", and
    # media nouns are excluded, so "stop the music" still pauses playback via
    # the media rules below instead of being read as an app called "music".
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:close|quit|exit)\s+(?:the\s+)?(?:app\s+)?(?!music\b|song\b|track\b|volume\b|sound\b|audio\b)(.+?)\s*(?:for\s+me)?$", "close_app"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?focus\s+(?:on\s+)?(.+)$", "focus_window"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:show|list|what(?:'s| are))\s+(?:the\s+)?windows?$", "list_windows"),
    # Process management
    (r"^(?:can\s+you\s+)?(?:please\s+)?kill\s+(?:the\s+)?(?:process|app)\s+(.+)$", "kill_process"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:show|list)\s+(?:the\s+)?processes$", "list_processes"),
    # Screenshot
    (r"^(?:can\s+you\s+)?(?:please\s+)?take\s+(?:a\s+)?screenshot$", "take_screenshot"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?screenshot$", "take_screenshot"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:delete|remove|clear)\s+(?:all\s+)?screenshots?$", "delete_screenshots"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?delete\s+(?:the\s+)?last\s+screenshot$", "delete_last_screenshot"),
    # Clipboard
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:get|read|show)\s+(?:the\s+)?clipboard$", "get_clipboard"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:copy|set)\s+(?:the\s+)?clipboard\s+(?:to\s+)?(.+)$", "set_clipboard"),
    # System commands
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:shut\s?down|power\s+off)\s*(?:the\s+)?(?:pc|computer|laptop)?$", "shutdown_pc"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?shutdown\s+(?:the\s+)?(?:pc|computer|laptop)$", "shutdown_pc"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:restart|reboot|re\s?start)\s*(?:the\s+)?(?:pc|computer|laptop)?$", "restart_pc"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?restart\s+(?:the\s+)?(?:pc|computer|laptop)$", "restart_pc"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?cancel\s+(?:the\s+)?shutdown$", "cancel_shutdown"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?lock\s+(?:the\s+)?(?:pc|computer|laptop|screen)$", "lock_pc"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:log\s*off|sign\s*out)\s*(?:the\s+)?(?:pc|computer)?$", "lock_pc"),
    # Assistant sleep mode: the PC stays on, Jarvis just stands down.
    (r"^(?:can\s+you\s+)?(?:please\s+)?sleep$", "jarvis_sleep"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:go\s+to\s+)?sleep(?:\s+(?:pc|computer|laptop))?$", "jarvis_sleep"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?wake\s+(?:up)?(?:\s+(?:jarvis|assistant))?$", "jarvis_wake"),
    # Actual system sleep - explicit target required.
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:put\s+)?(?:the\s+)?(?:pc|computer|laptop|system)\s+to\s+sleep$", "sleep_pc"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?system\s+sleep$", "sleep_pc"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?pc\s+sleep$", "sleep_pc"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?hibernate$", "hibernate_pc"),
    # Mouse/Keyboard
    (r"^(?:can\s+you\s+)?(?:please\s+)?click\s+(?:at\s+)?(\d+)\s*,?\s*(\d+)$", "mouse_click"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?move\s+(?:mouse\s+)?(?:to\s+)?(\d+)\s*,?\s*(\d+)$", "mouse_move"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?type\s+(.+)$", "type_text"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?write\s+(.+)$", "type_text"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?press\s+(.+)$", "press_key"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?hotkey\s+(.+)$", "hotkey"),
    # Weather - flexible
    (r"^(?:can\s+you\s+)?(?:please\s+)?weather(?:\s+(?:in|for|at)\s+(.+))?$", "weather"),
    (r"^what(?:'s| is) (?:the )?(?:weather|temperature|forecast)(?:\s+(?:in|for|at)\s+(.+))?$", "weather"),
    # News - flexible
    (r"^(?:can\s+you\s+)?(?:please\s+)?news$", "news"),
    (r"^what\s+news$", "news"),
    (r"^what(?:'s| are) (?:the )?(?:latest )?news$", "news"),
    (r"^(?:jarvis|hey|ok|okay|so|now|then|well|um|uh|service|assistant|and)?\s*what(?:'s| is)?\s*happening(?:\s+(?:in|with|about)\s+(.+))?$", "news"),
    (r"^(?:jarvis|hey|ok|okay|so|now|then|well|um|uh|service|assistant|and)?\s*tell\s+me\s+(?:what(?:'s| is)|about)\s+happening", "news"),
    (r"^(?:jarvis|hey|ok|okay|so|now|then|well|um|uh|service|assistant|and)?\s*what(?:'s| is)?\s*going on(?:\s+(?:in|with|about)\s+(.+))?$", "news"),
    (r"^(?:jarvis|hey|ok|okay|so|now|then|well|um|uh|service|assistant|and)?\s*any\s+(?:new|recent)\s+news$", "news"),
    (r"^(?:jarvis|hey|ok|okay|so|now|then|well|um|uh|service|assistant|and)?\s*latest\s+(?:on|about|in)\s+(.+)$", "news"),
    (r"^(?:jarvis|hey|ok|okay|so|now|then|well|um|uh|service|assistant)?\s*what(?:'s| is)?\s*(?:the\s+)?(?:latest|current|recent)\s+(?:on|about)?\s*(.+)$", "news"),
    (r"^(?:jarvis|hey|ok|okay|so|now|then|well|um|uh|service|assistant)?\s*world\s+cup\s*(?:news|update|score|result)?$", "sports_news"),
    (r"^(?:jarvis|hey|ok|okay|so|now|then|well|um|uh|service|assistant)?\s*what(?:'s| is)?\s*happening (?:with|in) (?:the\s+)?world\s+cup$", "sports_news"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?tech\s*news$", "tech_news"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?technology\s+news$", "tech_news"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?sports?\s*news$", "sports_news"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?sports?\s+headlines$", "sports_news"),
    (r"^find\s+(?:me\s+)?(?:the\s+)?(?:file|files|document|documents)\s+(.+)$", "find_file"),
    (r"^(?:list|show)\s+(?:recent|latest)\s+files$", "recent_files"),
    (r"^set\s+(?:a\s+)?reminder\s+(?:to\s+)?(.+)$", "set_reminder"),
    (r"^remind\s+me\s+(?:to\s+)?(.+)$", "set_reminder"),
    (r"^set\s+(?:an?\s+)?alarm\s+(?:for\s+)?(.+)$", "set_alarm"),
    (r"^alarm\s+(?:for\s+)?(.+)$", "set_alarm"),
    (r"^set\s+(?:a\s+)?timer\s+(?:for\s+)?(.+)$", "set_timer"),
    (r"^timer\s+(?:for\s+)?(.+)$", "set_timer"),
    (r"^(?:show|list|what(?:'s| are))\s+(?:my\s+)?reminders?$", "list_reminders"),
    (r"^clear\s+(?:all\s+)?reminders?$", "clear_reminders"),
    # More flexible variations
    (r"^(?:can\s+you\s+)?(?:please\s+)?turn\s+(?:the\s+)?(?:music|song|track)\s+off$", "spotify_pause"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?turn\s+(?:the\s+)?(?:music|song|track)\s+on$", "spotify_play"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:play|start)\s+(?:the\s+)?(?:music|song|track)$", "spotify_play"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:stop|pause)\s+(?:the\s+)?(?:music|song|track)$", "spotify_pause"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:switch|skip)\s+(?:to\s+)?(?:the\s+)?(?:next|forward)\s*(?:track|song)?$", "spotify_next"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:go|switch)\s+(?:back|previous)\s*(?:track|song)?$", "spotify_previous"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:turn|put)\s+(?:the\s+)?volume\s+(?:up|higher|louder)$", "spotify_volume_up"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:turn|put)\s+(?:the\s+)?volume\s+(?:down|lower|quieter)$", "spotify_volume_down"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:shut|make)\s+(?:it\s+)?(?:quiet|silent|mute)$", "spotify_mute"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:random|shuffle)\s+(?:the\s+)?(?:music|songs?)$", "spotify_shuffle"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:repeat|loop)\s+(?:the\s+)?(?:music|songs?)$", "spotify_repeat"),
    # Back to Jarvis variations
    (r"^(?:go\s+)?back\s+(?:to\s+)?(?:the\s+)?(?:browser|jarvis|dashboard|ui|frontend|interface|tab|page|window)$", "focus_jarvis"),
    (r"^(?:switch|focus)\s+(?:to\s+)?(?:the\s+)?(?:browser|jarvis|dashboard|tab|page)$", "focus_jarvis"),
    (r"^(?:jarvis)$", "focus_jarvis"),
    (r"^hey\s+(?:jarvis)$", "focus_jarvis"),
    (r"^hi\s+(?:jarvis)$", "focus_jarvis"),
    # Self-healing
    (r"^(?:can\s+you\s+)?(?:please\s+)?fix\s+(?:the\s+)?(?:last\s+)?error$", "fix_error"),
    (r"^heal$", "fix_error"),
    (r"^self[- ]?heal$", "fix_error"),
    # Voice status
    (r"^voice\s+status$", "voice_status"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:list|show)\s+(?:the\s+)?(?:available\s+)?voices?$", "list_voices"),
    (r"^mute$", "voice_mute"),
    (r"^unmute$", "voice_unmute"),
    # "shut up" / "be quiet" with no media target means stop talking, not
    # pause the music, so it belongs to the voice controls.
    (r"^(?:shut\s+up|be\s+quiet|quiet|hush|silence)$", "voice_mute"),
    (r"^voice\s+backend$", "voice_backend"),
    # "set voice backend X" is answered rather than dispatched as a switch:
    # there is only one backend, so the useful reply is that X is not one.
    (r"^set\s+voice\s+backend\s+(.+)$", "voice_backend"),
    (r"^voice\s+backend\s+(.+)$", "voice_backend"),
    # "set voice ryan" selects a TTS voice. Declared after the backend rules
    # so "set voice backend fish" is not read as a voice named "backend fish".
    (r"^set\s+voice\s+(.+)$", "set_voice"),
    (r"^use\s+voice\s+(.+)$", "set_voice"),
    # System mixer mute/unmute needs the "system"/"master" qualifier to be
    # distinguished from the media mute above.
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:mute|silence)\s+(?:the\s+)?(?:system\s+|master\s+)?(?:volume|sound|audio)$", "volume_mute"),
    (r"^(?:can\s+you\s+)?(?:please\s+)?(?:unmute|unsilence)\s+(?:the\s+)?(?:system\s+|master\s+)?(?:volume|sound|audio)$", "volume_unmute"),
    # Skills (OpenJarvis)
    (r"^(?:run|execute|start)\s+skill\s+(.+)$", "run_skill"),
    (r"^(?:run|execute|start)\s+(?:the\s+)?(?:morning\s+digest|daily\s+briefing)$", "run_skill"),
    (r"^(?:run|execute|start)\s+(?:the\s+)?(?:system\s+diagnostics|system\s+health\s+check)$", "run_skill"),
    (r"^(?:list|show|what\s+are\s+your)\s+skills$", "list_skills"),
]

ACTION_MAP = {name: regex for regex, name in _ACTION_PATTERNS}
ACTION_NAMES = list(dict.fromkeys(name for _, name in _ACTION_PATTERNS))

# Utterance -> routing decision, so a repeated phrase does not spend another
# LLM request. The free tier is a small daily budget, not a per-minute one.
_ROUTE_CACHE: dict[str, tuple[str, str]] = {}
_ROUTE_CACHE_MAX = 200


def _extract_route_json(response: str) -> Optional[dict]:
    """Pull the routing decision out of a model response.

    The model is asked for JSON only, but it may still wrap it in a code fence
    or add a sentence of prose, so the first object in the text is taken and
    parsed defensively. Anything unparseable returns None and the caller falls
    back to conversation.
    """
    text = response.strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# Leading words that mark an utterance as a question rather than a command.
_QUESTION_OPENERS = frozenset({
    "what", "why", "how", "who", "whom", "whose", "when", "where", "which",
    "is", "are", "was", "were", "do", "does", "did", "am",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    "tell", "explain", "describe", "give", "know", "think", "guess",
})


def _looks_like_question(lower: str) -> bool:
    """True when an utterance is plainly a question, not a command.

    This only runs after the pattern table and typo repair have both failed,
    so a real command has already been caught and a false positive here just
    sends the phrase to conversation instead of routing. That trade is worth
    it: skipping the routing request halves the daily quota spent on questions.
    """
    text = lower.strip().rstrip("?!.").strip()
    if not text:
        return False
    first = text.split(maxsplit=1)[0].strip("\"'")
    return first in _QUESTION_OPENERS


def _llm_parse_command(cmd: str) -> Optional[tuple[str, str]]:
    """Use the LLM to map free-form language onto a real action.

    This is the last step of ``_detect_action`` and only runs when the pattern
    table and fuzzy repair both miss, so ordinary commands stay fast and free.
    The model chooses from the described catalogue in ``mcp.capabilities``
    rather than from bare action names, and its choice is checked against the
    deny-by-default allowlist: an action the model is not permitted to select
    is dropped and the utterance becomes conversation instead.
    """
    # The free Gemini tier allows only 20 requests per day, and a routing
    # decision is stable for a given utterance, so repeat phrases are served
    # from memory instead of spending quota.
    key = " ".join(cmd.lower().split())
    cached = _ROUTE_CACHE.get(key)
    if cached is not None:
        log.debug("LLM parse: cache hit for '%s' -> %s", key, cached[0])
        return cached

    # An offline core can only answer in prose, which no amount of parsing can
    # turn into a route. Attempting it anyway is what logged a parse error on
    # every utterance while the client was known-missing.
    if not llm_mcp.available:
        log.debug("LLM parse skipped: core unavailable (%s)",
                  llm_mcp.get_status().get("error") or "not configured")
        return None

    try:
        prompt = build_routing_prompt(cmd)
        response = llm_mcp.ask(
            prompt,
            use_history=False,
            skip_cooldown=True,
            temperature=0.0,
            purpose="route",
        )
        data = _extract_route_json(response)
        if data is None:
            log.debug("LLM parse: no usable JSON in response")
            return None

        action = str(data.get("action", "")).strip()
        payload = str(data.get("payload", "")).strip()

        if not is_routable(action):
            if action in ACTION_NAMES:
                log.warning(
                    "LLM chose non-routable action %r, falling back to chat", action
                )
            else:
                log.info("LLM returned unknown action %r, ignoring", action)
            return None

        log.info("LLM parsed: '%s' -> action=%s, payload=%s", cmd, action, payload)
        if len(_ROUTE_CACHE) >= _ROUTE_CACHE_MAX:
            _ROUTE_CACHE.pop(next(iter(_ROUTE_CACHE)), None)
        _ROUTE_CACHE[key] = (action, payload)
        return action, payload
    except Exception as e:
        log.error("LLM parse error: %s", e)
    return None


def _match_pattern(lower: str) -> Optional[tuple[str, str, str]]:
    """Match a command against the pattern table.

    Returns ``(action, payload, matched_text)`` or ``None``. All capture
    groups are folded into the payload, so multi-group patterns such as
    "click at 500, 300" no longer lose everything after the first group.
    """
    for pattern, name in _ACTION_PATTERNS:
        try:
            m = re.match(pattern, lower, re.IGNORECASE)
        except re.error as e:
            log.warning("Bad command pattern %r: %s", pattern, e)
            continue
        if not m:
            continue
        groups = [g.strip() for g in m.groups() if g and g.strip()]
        payload = ", ".join(groups)
        return name, payload, m.group(0)
    return None


def _detect_action(cmd: str, memory: Optional[dict] = None) -> Optional[tuple[str, str]]:
    lower = cmd.strip().lower()
    if not lower:
        return None

    # 1. Exact match against the pattern table.
    hit = _match_pattern(lower)
    if hit:
        return hit[0], hit[1]

    log.debug("No pattern match for '%s' in %d patterns", lower, len(_ACTION_PATTERNS))

    # 2. Guarded fuzzy repair of likely speech-recognition errors.
    pairs = _fuzzy_correct_words(lower)
    corrected = " ".join(word for word, _ in pairs)
    if corrected != lower:
        repaired = _match_pattern(corrected)
        if repaired and _fuzzy_repair_is_trustworthy(lower, pairs, repaired[2]):
            log.info("Fuzzy match: '%s' -> '%s' (%s)", lower, corrected, repaired[0])
            return repaired[0], repaired[1]
        log.info("Fuzzy match '%s' -> '%s' rejected as untrustworthy", lower, corrected)

    # 3. Ask the LLM to map free-form language onto an action.
    #    Run in the background so the user gets an immediate conversational
    #    response instead of waiting on a network round-trip. If the background
    #    routing resolves to a real action, it is executed asynchronously.
    #    A question is not a candidate for an action, and the Gemini free tier
    #    only allows 20 requests a day, so anything that reads as a question
    #    goes straight to the conversational path.
    if _looks_like_question(lower):
        log.debug("'%s' reads as a question, skipping the routing request", lower)
        return ("chat", cmd)

    if memory is None:
        memory = {}

    # Try synchronous routing first with a short timeout. Fast responses
    # (e.g. test fakes) return immediately; slow network calls fall through
    # to async dispatch so the user is never blocked.
    try:
        future = _routing_executor.submit(_llm_parse_command, lower)
        result = future.result(timeout=0.3)
        if result is not None and result[0] != "chat":
            return result
    except concurrent.futures.TimeoutError:
        pass  # Slow LLM - fall through to async
    except Exception as e:
        log.debug("Synchronous LLM routing failed: %s", e)

    # Fall back to async routing so the user gets an immediate reply.
    threading.Thread(
        target=_llm_route_background, args=(lower, cmd, memory), daemon=True,
    ).start()

    # 4. Nothing matched yet - treat it as conversation so the user hears an
    #    immediate reply while LLM routing resolves in the background.
    return ("chat", cmd)


def _llm_route_background(lower: str, cmd: str, memory: dict) -> None:
    """Run LLM command routing in a background thread.

    If the LLM resolves the utterance to a real action (not ``chat``),
    execute it via ``_handle_command_inner`` with the pre-detected action,
    bypassing the detection phase so we don't loop back into routing.
    """
    try:
        result = _llm_parse_command(lower)
    except Exception as e:
        log.error("Background LLM routing error: %s", e)
        return
    if result is None:
        return
    action_type, payload = result
    if action_type == "chat":
        return
    log.info("Background LLM routed '%s' -> %s", lower, action_type)
    try:
        _handle_command_inner(cmd, memory, pre_action=(action_type, payload))
    except Exception as e:
        log.error("Background action execution error: %s", e)


# ---------------------------------------------------------------------------
# Command handler - speak DURING action
# ---------------------------------------------------------------------------

def _reset_status_delayed(delay: float = 2.0) -> None:
    """Reset status to listening after a delay (safety net for stuck states)."""
    def _do_reset():
        time.sleep(delay)
        if not voice_mcp.is_speaking():
            set_status("listening")
    threading.Thread(target=_do_reset, daemon=True).start()


# A repeated identical command within this window is treated as a re-trigger
# of the same utterance (wake word latching, doubled keypress) and ignored.
# De-duplicating commands is safe; de-duplicating replies is not, because two
# identical questions can legitimately have one identical answer.
COMMAND_DEDUPE_S = 1.5
_command_lock = threading.Lock()
_last_command = ("", 0.0)


def _is_duplicate_command(cmd: str) -> bool:
    global _last_command
    now = time.monotonic()
    with _command_lock:
        previous, when = _last_command
        duplicate = previous == cmd and now - when < COMMAND_DEDUPE_S
        if not duplicate:
            _last_command = (cmd, now)
    return duplicate


def handle_command(text: str, memory: dict) -> None:
    """Process a user command - speak while action runs."""
    global _last_user_cmd
    cmd = text.strip()
    if len(cmd) < MIN_COMMAND_LEN:
        return

    if _is_duplicate_command(cmd):
        log.info("Ignoring repeated command: %s", cmd)
        return

    _last_user_cmd = cmd
    log.info("USER command: %s", cmd)
    broadcast({"type": "transcript", "text": cmd, "speaker": "user"})
    set_status("processing")

    try:
        _handle_command_inner(cmd, memory)
    except Exception as e:
        log.error("Command error: %s", e)
        _record_error(e, "handle_command")
        speak("I encountered an error, sir.")
    finally:
        _reset_status_delayed()


def _handle_command_inner(cmd: str, memory: dict, pre_action: Optional[tuple[str, str]] = None) -> None:
    """Inner command handler - exceptions propagate to caller for logging."""
    global _jarvis_sleeping
    if pre_action is not None:
        action_type, payload = pre_action
    else:
        action = _detect_action(cmd)
        if not action:
            return
        action_type, payload = action

    if _jarvis_sleeping and action_type != "jarvis_wake":
        _record_sleep_event(cmd)
        speak("I'm in standby, sir. Say 'wake up' to bring me back.")
        return

    verdict = loop_guard.check_call(action_type, payload)
    if verdict.blocked:
        log.warning("Command blocked by LoopGuard: %s (%s)", action_type, verdict.reason)
        speak(f"I have paused that operation, sir: {verdict.reason}")
        return

    # Quick responses - speak immediately
    if action_type == "cancel":
        voice_mcp.interrupt()
        speak("Understood. Command cancelled, sir.")
        return
    elif action_type == "show_panels":
        set_panels(True)
        speak("Panels displayed, sir.")
        return
    elif action_type == "hide_panels":
        set_panels(False)
        speak("Panels hidden, sir.")
        return
    elif action_type == "time":
        n = datetime.now()
        speak(f"It is currently {n.strftime('%I:%M %p')} on {n.strftime('%A, %B %d, %Y')}.")
        return
    elif action_type == "greeting":
        speak(random.choice([
            "Good to see you, sir.",
            "At your service, sir.",
            "How may I assist you today?",
            "Hello, sir. What can I do for you?",
            "Greetings, sir. Ready when you are.",
            "Ah, sir. Always a pleasure to hear from you.",
        ]))
        return
    elif action_type == "jarvis_sleep":
        _jarvis_sleeping = True
        _sleep_events.clear()
        speak("Going into standby, sir. The system stays on - I'll keep running pending jobs and brief you when you wake me.")
        return
    elif action_type == "jarvis_wake":
        if not _jarvis_sleeping:
            speak("I'm already awake, sir.")
            return
        _jarvis_sleeping = False
        speak("Standby ended, sir. I'm back.")
        if _sleep_events:
            speak("Here's what happened while you were away: " + "; ".join(_sleep_events))
            _sleep_events.clear()
        else:
            speak("Nothing to report while you were away, sir.")
        return
    elif action_type == "brief":
        if _sleep_events:
            speak("Here's what happened while you were away: " + "; ".join(_sleep_events))
            _sleep_events.clear()
        else:
            speak("Nothing to report, sir.")
        return
    elif action_type == "thanks":
        speak(random.choice([
            "You're welcome, sir.",
            "My pleasure.",
            "Happy to help.",
            "Always at your service.",
            "Anytime, sir. That's what I'm here for.",
            "Glad I could assist, sir.",
        ]))
        return
    elif action_type == "about":
        speak(f"I am {APP_NAME}, your personal AI assistant. Created to make your life easier, sir.")
        return
    elif action_type == "chat":
        # Stream LLM response to speech + UI for zero-latency feel
        def _chat_stream():
            try:
                chat_input = payload if payload else cmd

                time_sensitive = any(kw in chat_input.lower() for kw in [
                    "happening", "news", "latest", "current", "today", "now",
                    "world cup", "election", "score", "result", "tournament",
                    "war", "crisis", "stock", "price", "weather", "forecast"
                ])

                context_info = ""
                if time_sensitive:
                    try:
                        if any(kw in chat_input.lower() for kw in ["world cup", "football", "soccer", "cricket", "tennis", "nba", "score", "tournament"]):
                            news = news_mcp.get_news_by_topic(chat_input)
                        else:
                            news = news_mcp.get_news()
                        if news and "unable" not in news.lower() and "slow" not in news.lower():
                            cleaned_news = _clean_news(news)
                            if cleaned_news:
                                context_info = f"\n\nHere's what's happening right now:\n{cleaned_news}"
                    except Exception:
                        pass

                chat_prompt = f"""{llm_mcp._load_soul()}

Here is what is currently happening with this topic:
{context_info}

Respond naturally like a friend telling you what they just heard. 1-2 sentences max. No sources, no "according to", no dates unless specifically asked. Just the key facts in a casual way.

User: {chat_input}

JARVIS:"""
                _speak_stream_broadcast(_ask_llm_stream, chat_prompt, {})
            except Exception as e:
                log.error("Chat stream error: %s", e)
                speak("I'm not sure how to respond to that, sir.")
        threading.Thread(target=_chat_stream, daemon=True).start()
        return
    elif action_type == "sleep":
        speak("Going to sleep mode, sir. Good night.")
        return

    # Actions - speak FIRST, action runs in background
    elif action_type == "focus_jarvis":
        # Switch focus to the HUD tab in Chrome
        def _focus_jarvis():
            import win32gui
            import win32con
            # Find the Chrome window showing the HUD. Match the title
            # (index.html <title>) as well as the dev-server hostnames,
            # since the title follows APP_NAME.
            def enum_callback(hwnd, results):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd).lower()
                    if is_hud_window_title(title):
                        results.append(hwnd)
            windows = []
            win32gui.EnumWindows(enum_callback, windows)
            if windows:
                hwnd = windows[0]
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                log.info("Focused %s window", APP_NAME)
            else:
                log.info("%s window not found, trying any Chrome window", APP_NAME)
                # Fallback: focus any Chrome window
                def enum_chrome(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd).lower()
                        if 'chrome' in title:
                            results.append(hwnd)
                chrome_windows = []
                win32gui.EnumWindows(enum_chrome, chrome_windows)
                if chrome_windows:
                    hwnd = chrome_windows[0]
                    if win32gui.IsIconic(hwnd):
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(hwnd)
                    log.info("Focused Chrome window")
        threading.Thread(target=_focus_jarvis, daemon=True).start()
        speak("Back to JARVIS, sir.")
        return
    elif action_type == "open_app":
        target = clean_app_name(payload) or payload
        _speak_during_act(automation_mcp.open_app, f"Opening {target}...", payload)
        return
    elif action_type == "open_website":
        _speak_during_act(browser_mcp.open, f"Opening {payload}...", payload)
        return
    elif action_type == "search":
        _speak_during_act(browser_mcp.search, f"Searching for {payload}...", payload)
        return
    elif action_type == "youtube":
        _speak_during_act(browser_mcp.open_youtube, f"Opening YouTube for {payload}...", payload)
        return
    elif action_type == "spotify":
        if payload:
            _speak_during_act(browser_mcp.play_spotify, f"Playing {payload} on Spotify...", payload)
        else:
            # Try to resume if Spotify is open, otherwise open it
            _speak_during_act(browser_mcp.play_spotify, "Opening Spotify...")
        return
    elif action_type == "spotify_random":
        # Play random music based on time of day
        _speak_during_act(browser_mcp.play_random_music, "Playing some music for you...")
        return
    elif action_type == "spotify_play":
        _speak_during_act(browser_mcp.spotify_control, "Playing...", "play")
        return
    elif action_type == "spotify_pause":
        _speak_during_act(browser_mcp.spotify_control, "Pausing...", "pause")
        return
    elif action_type == "spotify_next":
        _speak_during_act(browser_mcp.spotify_control, "Next track...", "next")
        return
    elif action_type == "spotify_previous":
        _speak_during_act(browser_mcp.spotify_control, "Previous track...", "previous")
        return
    elif action_type == "spotify_volume_up":
        _speak_during_act(browser_mcp.spotify_control, "Volume up...", "volume_up")
        return
    elif action_type == "spotify_volume_down":
        _speak_during_act(browser_mcp.spotify_control, "Volume down...", "volume_down")
        return
    elif action_type == "spotify_volume_set":
        # Extract percentage from payload
        try:
            val = int(re.search(r'(\d{1,3})', payload).group(1))
            _speak_during_act(browser_mcp.spotify_control, f"Setting volume to {val} percent...", "volume_set", val)
        except (ValueError, AttributeError):
            speak("What volume level would you like, sir?")
        return
    elif action_type == "spotify_volume_max":
        _speak_during_act(browser_mcp.spotify_control, "Volume to maximum...", "volume_max")
        return
    elif action_type == "spotify_volume_min":
        _speak_during_act(browser_mcp.spotify_control, "Volume to minimum...", "volume_min")
        return
    elif action_type == "spotify_mute":
        _speak_during_act(browser_mcp.spotify_control, "Muting...", "mute")
        return
    elif action_type == "spotify_shuffle":
        _speak_during_act(browser_mcp.spotify_control, "Toggling shuffle...", "shuffle")
        return
    elif action_type == "spotify_repeat":
        _speak_during_act(browser_mcp.spotify_control, "Toggling repeat...", "repeat")
        return
    elif action_type == "gmail":
        _speak_during_act(browser_mcp.open_gmail, "Opening Gmail...")
        return
    elif action_type == "maps":
        _speak_during_act(browser_mcp.open_maps, "Opening Maps...")
        return
    elif action_type == "brightness":
        _speak_during_act(automation_mcp.set_brightness, f"Setting brightness to {payload} percent.", int(payload))
        return
    elif action_type == "volume":
        # The pattern only captures a number, so max/min arrive with an
        # empty payload. Normalise before the int() conversion.
        level = (payload or "").strip()
        if not level:
            level = "100" if re.search(r"\b(max|maximum|full|loudest)\b", cmd) else "0"
        try:
            value = max(0, min(100, int(level)))
        except ValueError:
            speak("What volume level would you like, sir?", )
            return
        _speak_during_act(automation_mcp.set_volume, f"Setting volume to {value} percent.", value)
        return
    elif action_type == "volume_mute":
        _speak_during_act(automation_mcp.set_volume, "Muting system volume.", 0)
        return
    elif action_type == "volume_unmute":
        _speak_during_act(automation_mcp.set_volume, "Restoring system volume.", 50)
        return
    elif action_type == "execute":
        _speak_during_act(automation_mcp.execute_command, f"Executing {payload}...", payload)
        return
    elif action_type == "remember":
        _speak_during_act(memory_mcp.remember, f"Remembering: {payload}", payload)
        return
    elif action_type == "recall":
        _speak_during_act(memory_mcp.recall, f"Recalling: {payload}", payload)
        return
    elif action_type == "add_task":
        _task_action(tasks_mcp.add_task, f"Adding task: {payload}", payload)
        return
    elif action_type == "complete_task":
        _task_action(tasks_mcp.complete_task, f"Marking task {payload} as complete.", int(payload))
        return
    elif action_type == "delete_task":
        _task_action(tasks_mcp.delete_task, f"Deleting task {payload}.", int(payload))
        return
    elif action_type == "delete_multiple_tasks":
        _task_action(tasks_mcp.delete_multiple_tasks, f"Deleting tasks {payload}.", payload)
        return
    elif action_type == "clear_all_tasks":
        _task_action(tasks_mcp.clear_all_tasks, "Clearing all tasks...")
        return
    elif action_type == "delete_task_by_desc":
        _task_action(tasks_mcp.delete_task_by_description, f"Deleting task {payload}...", payload)
        return
    elif action_type == "list_tasks":
        _task_action(tasks_mcp.list_tasks, "Here are your tasks.")
        return
    elif action_type == "clear_tasks":
        _task_action(tasks_mcp.clear_done_tasks, "Clearing completed tasks.")
        return
    elif action_type == "system_info":
        _speak_during_act(system_mcp.get_system_info, "Checking system status...")
        return
    elif action_type == "cpu_info":
        _speak_during_act(system_mcp.handle, "Checking CPU...", "cpu", "")
        return
    elif action_type == "ram_info":
        _speak_during_act(system_mcp.handle, "Checking memory...", "ram", "")
        return
    elif action_type == "battery_info":
        _speak_during_act(system_mcp.handle, "Checking battery...", "battery", "")
        return
    elif action_type == "disk_info":
        _speak_during_act(system_mcp.handle, "Checking disk...", "disk", "")
        return
    elif action_type == "network_info":
        _speak_during_act(system_mcp.handle, "Checking network...", "network", "")
        return
    elif action_type == "running_apps":
        _speak_during_act(system_mcp.get_running_apps, "Checking running apps...")
        return
    elif action_type == "find_file":
        _speak_during_act(files_mcp.find_file, f"Searching for file {payload}...", payload)
        return
    elif action_type == "open_file":
        _speak_during_act(files_mcp.open_file, f"Opening file {payload}...", payload)
        return
    elif action_type == "open_folder":
        _speak_during_act(files_mcp.open_folder, f"Opening folder {payload}...", payload)
        return
    elif action_type == "recent_files":
        _speak_during_act(files_mcp.list_recent_files, "Here are your recent files.")
        return
    # File operations
    elif action_type == "create_file":
        _speak_during_act(automation_mcp.create_file, f"Creating file {payload}...", payload)
        return
    elif action_type == "read_file":
        _speak_during_act(automation_mcp.read_file, f"Reading file {payload}...", payload)
        return
    elif action_type == "delete_file":
        _speak_during_act(automation_mcp.delete_file, f"Deleting file {payload}...", payload)
        return
    elif action_type == "copy_file":
        parts = payload.split(" to ", 1)
        if len(parts) == 2:
            _speak_during_act(automation_mcp.copy_file, f"Copying {parts[0].strip()}...", parts[0].strip(), parts[1].strip())
        else:
            speak("Please specify source and destination, sir.")
        return
    elif action_type == "move_file":
        parts = payload.split(" to ", 1)
        if len(parts) == 2:
            _speak_during_act(automation_mcp.move_file, f"Moving {parts[0].strip()}...", parts[0].strip(), parts[1].strip())
        else:
            speak("Please specify source and destination, sir.")
        return
    elif action_type == "list_directory":
        path = payload if payload else "."
        _speak_during_act(automation_mcp.list_directory, f"Listing {path}.", path)
        return
    # Window management
    elif action_type == "minimize_window":
        _speak_during_act(automation_mcp.minimize_window, "Minimizing window...")
        return
    elif action_type == "maximize_window":
        _speak_during_act(automation_mcp.maximize_window, "Maximizing window...")
        return
    elif action_type == "close_window":
        _speak_during_act(automation_mcp.close_window, "Closing window...")
        return
    elif action_type == "close_app":
        target = clean_app_name(payload) or payload
        _speak_during_act(automation_mcp.close_app, f"Closing {target}...", payload)
        return
    elif action_type == "close_tab":
        def _close_tab():
            import pyautogui
            pyautogui.hotkey('ctrl', 'w')
        threading.Thread(target=_close_tab, daemon=True).start()
        speak("Closing tab, sir.")
        return
    elif action_type == "focus_window":
        _speak_during_act(automation_mcp.focus_window, f"Focusing {payload}...", payload)
        return
    elif action_type == "list_windows":
        _speak_during_act(automation_mcp.list_windows, "Here are the open windows...")
        return
    # Process management
    elif action_type == "kill_process":
        _speak_during_act(automation_mcp.kill_process, f"Killing {payload}...", payload)
        return
    elif action_type == "list_processes":
        _speak_during_act(automation_mcp.list_processes, "Here are the running processes...")
        return
    # Screenshot
    elif action_type == "take_screenshot":
        _speak_during_act(automation_mcp.take_screenshot, "Taking screenshot...")
        return
    elif action_type == "delete_screenshots":
        _speak_during_act(automation_mcp.delete_screenshots, "Deleting screenshots...")
        return
    elif action_type == "delete_last_screenshot":
        _speak_during_act(automation_mcp.delete_last_screenshot, "Deleting last screenshot...")
        return
    # Clipboard
    elif action_type == "get_clipboard":
        _speak_during_act(automation_mcp.get_clipboard, "Reading clipboard...")
        return
    elif action_type == "set_clipboard":
        _speak_during_act(automation_mcp.set_clipboard, f"Copied to clipboard: {payload}", payload)
        return
    # System commands
    elif action_type == "shutdown_pc":
        _speak_during_act(automation_mcp.shutdown_pc, "Shutting down...")
        return
    elif action_type == "restart_pc":
        _speak_during_act(automation_mcp.restart_pc, "Restarting...")
        return
    elif action_type == "cancel_shutdown":
        _speak_during_act(automation_mcp.cancel_shutdown, "Cancelling shutdown...")
        return
    elif action_type == "lock_pc":
        _speak_during_act(automation_mcp.lock_pc, "Locking computer...")
        return
    elif action_type == "sleep_pc":
        _speak_during_act(automation_mcp.sleep_pc, "Going to sleep...")
        return
    elif action_type == "hibernate_pc":
        _speak_during_act(automation_mcp.hibernate_pc, "Hibernating...")
        return
    # Mouse/Keyboard
    elif action_type == "mouse_click":
        try:
            nums = re.findall(r'\d+', payload)
            x, y = int(nums[0]), int(nums[1])
            _speak_during_act(automation_mcp.mouse_click, f"Clicking at {x}, {y}...", x, y)
        except (ValueError, IndexError):
            speak("Please specify coordinates like 500, 300, sir.")
        return
    elif action_type == "mouse_move":
        try:
            nums = re.findall(r'\d+', payload)
            x, y = int(nums[0]), int(nums[1])
            _speak_during_act(automation_mcp.mouse_move, f"Moving mouse to {x}, {y}...", x, y)
        except (ValueError, IndexError):
            speak("Please specify coordinates like 500, 300, sir.")
        return
    elif action_type == "type_text":
        # Check if payload has "in [app]" format
        parts = payload.rsplit(" in ", 1)
        text = parts[0].strip()
        app = parts[1].strip() if len(parts) > 1 else None
        
        def _type_in_app():
            import time
            # Open app first if specified
            if app:
                try:
                    import win32gui
                    hwnd = win32gui.GetForegroundWindow()
                    title = win32gui.GetWindowText(hwnd).lower()
                    if app.lower() not in title:
                        automation_mcp.open_app(app)
                        time.sleep(1.5)
                except Exception:
                    automation_mcp.open_app(app)
                    time.sleep(1.5)
            automation_mcp.type_text(text)
        threading.Thread(target=_type_in_app, daemon=True).start()
        if app:
            speak(f"Typing in {app}: {text}")
        else:
            speak(f"Typing: {text}")
        return
    elif action_type == "press_key":
        _speak_during_act(automation_mcp.press_key, f"Pressing {payload}...", payload)
        return
    elif action_type == "hotkey":
        _speak_during_act(automation_mcp.hotkey, f"Pressing {payload}...", payload)
        return
    elif action_type == "set_reminder":
        _speak_during_act(reminders_mcp.set_reminder, f"Setting reminder: {payload}", payload)
        return
    elif action_type == "set_alarm":
        _speak_during_act(reminders_mcp.set_alarm, f"Setting alarm for {payload}.", payload)
        return
    elif action_type == "set_timer":
        _speak_during_act(reminders_mcp.set_timer, f"Setting timer for {payload}.", payload)
        return
    elif action_type == "list_reminders":
        _speak_during_act(reminders_mcp.list_reminders, "Here are your reminders.")
        return
    elif action_type == "clear_reminders":
        _speak_during_act(reminders_mcp.clear_all_reminders, "Clearing all reminders.")
        return
    elif action_type == "voice_status":
        _speak_during_act(voice_mcp.handle, "Checking voice status.", "voice status", "")
        return
    elif action_type == "voice_mute":
        _speak_during_act(voice_mcp.handle, "Voice muted.", "voice mute", "")
        return
    elif action_type == "voice_unmute":
        _speak_during_act(voice_mcp.handle, "Voice unmuted.", "voice unmute", "")
        return
    elif action_type == "voice_backend":
        _speak_during_act(voice_mcp.handle, "Checking voice backend.", "voice backend", payload)
        return
    elif action_type == "list_voices":
        _speak_during_act(voice_mcp.handle, "Listing available voices.", "list voices", "")
        return
    elif action_type == "set_voice":
        _speak_during_act(voice_mcp.handle, f"Setting voice to {payload}.", "set voice", payload)
        return
    elif action_type == "fix_error":
        _handle_fix_error(cmd)
        return

    # --- Self-evolution. A review is slow (it runs the tests), so it goes
    # --- to the background; everything else answers straight away.
    elif action_type == "self_review":
        threading.Thread(target=_handle_self_review, daemon=True).start()
        speak("I will look myself over, sir.")
        return
    elif action_type == "evolve":
        threading.Thread(target=_handle_evolve, daemon=True).start()
        speak("I will review myself and bring you something to approve, sir.")
        return
    elif action_type == "evolution_status":
        speak(evolution_mcp.handle("evolution status", payload) or "")
        return
    elif action_type == "show_proposals":
        speak(evolution_mcp.handle("show proposals", payload) or "")
        return
    elif action_type == "approve_evolution":
        speak(evolution_mcp.handle("approve evolution", payload) or "")
        return
    elif action_type == "reject_evolution":
        speak(evolution_mcp.handle("reject evolution", payload) or "")
        return
    elif action_type == "revert_evolution":
        speak(evolution_mcp.handle("revert evolution", payload) or "")
        return
    elif action_type == "run_skill":
        threading.Thread(target=_handle_run_skill, args=(payload,), daemon=True).start()
        return
    elif action_type == "list_skills":
        skills = skill_manager.list_skills()
        if not skills:
            speak("No custom skills are currently installed, sir.")
        else:
            names = ", ".join(s["name"] for s in skills)
            speak(f"I have {len(skills)} skills available, sir: {names}.")
        return

    # Network-dependent - non-blocking
    elif action_type == "weather":
        threading.Thread(target=_fetch_and_respond, args=("weather", payload), daemon=True).start()
        return
    elif action_type == "news":
        threading.Thread(target=_fetch_and_respond, args=("news", payload), daemon=True).start()
        return
    elif action_type == "tech_news":
        threading.Thread(target=_fetch_and_respond, args=("tech_news", payload), daemon=True).start()
        return
    elif action_type == "sports_news":
        threading.Thread(target=_fetch_and_respond, args=("sports_news", payload), daemon=True).start()
        return
    else:
        threading.Thread(target=_speak_stream_broadcast, args=(_ask_llm_stream, cmd, memory), daemon=True).start()
        return

# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the HUD from static/ so there is a single frontend to maintain."""
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        log.error("HUD missing at %s", index)
        return HTMLResponse(
            f"<h1>{APP_NAME}</h1><p>static/index.html is missing.</p>",
            status_code=500,
        )
    return FileResponse(str(index))


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# The HUD is a locked design that references its assets relatively from "/",
# so serve them at the root as well as under /static/.
@app.get("/style.css")
async def hud_css():
    path = STATIC_DIR / "style.css"
    if not path.is_file():
        return Response(status_code=404)
    return FileResponse(str(path), media_type="text/css")


@app.get("/app.js")
async def hud_js():
    path = STATIC_DIR / "app.js"
    if not path.is_file():
        return Response(status_code=404)
    return FileResponse(str(path), media_type="application/javascript")


# Mounted last so it never shadows the routes above.
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Close codes a browser sends when the tab is simply closed, reloaded, or
# navigated away: normal, going away, no status frame, and abnormal closure
# (a tab close sends no close frame). These were logged as warnings, which
# buried real faults under ~80 noise lines per session.
_NORMAL_WS_CLOSE_CODES = (1000, 1001, 1005, 1006)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    with _ws_lock:
        _websocket_clients.append(websocket)
    # Tell the client where the panels stand, so a reload of the tab comes
    # back the way the user left it.
    await websocket.send_text(json.dumps({"type": "panels", "visible": _panels_visible}))
    log.info("WebSocket client connected.")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "command":
                    text = msg.get("text", "").strip()
                    if text:
                        voice_mcp.interrupt()
                        threading.Thread(target=handle_command, args=(text, {}), daemon=True).start()
                elif msg.get("type") == "interrupt":
                    voice_mcp.interrupt()
                elif msg.get("type") == "ptt":
                    # Hold-to-talk: the HUD holds a key down while the user
                    # speaks, then releases to transcribe and dispatch.
                    active = bool(msg.get("active", False))
                    voice_mcp.set_push_to_talk(active)
                    # Keep the reactor indicator honest even if a client
                    # missed its own optimistic update.
                    broadcast({"type": "ptt", "active": active,
                               "listening": voice_mcp.is_listening()})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect as e:
        if e.code in _NORMAL_WS_CLOSE_CODES:
            log.info("WebSocket client went away (code %s).", e.code)
        else:
            log.warning("WebSocket closed unexpectedly: %s", e)
    except Exception as e:
        log.warning("WebSocket error: %s", e)
    finally:
        with _ws_lock:
            if websocket in _websocket_clients:
                _websocket_clients.remove(websocket)
        # A tab that vanishes mid-press would otherwise leave push-to-talk
        # latched open, so every stray word after it became a command.
        if voice_mcp.is_push_to_talk():
            voice_mcp.set_push_to_talk(False)
            broadcast({"type": "ptt", "active": False,
                       "listening": voice_mcp.is_listening()})
        log.info("WebSocket client disconnected.")


@app.get("/health")
async def health():
    return {
        "status": "online",
        "version": VERSION,
        "name": APP_NAME,
        "voice": voice_mcp.get_status(),
        "llm": llm_mcp.get_status(),
        "task_count": tasks_mcp.get_pending_count(),
        "metrics": system_mcp.get_ui_metrics(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/tasks")
async def api_tasks():
    """Current task list, shaped for the HUD."""
    return {"tasks": tasks_mcp.get_tasks_for_ui()}


@app.get("/api/evolution")
async def api_evolution():
    """The state of the self-evolution module and any pending proposals."""
    return {
        "status": evolution_mcp.get_status(),
        "proposals": evolution_mcp.get_proposals(),
        "lessons": evolution_mcp.get_lessons()[-10:],
    }


@app.post("/api/evolution/{proposal_id}")
async def api_evolution_decide(proposal_id: str, action: str = "approve"):
    """Approve, reject or undo a proposal the assistant made for itself."""
    if action == "approve":
        ok, message = evolution_mcp.approve(proposal_id)
    elif action == "reject":
        ok, message = evolution_mcp.reject(proposal_id)
    elif action == "revert":
        ok, message = evolution_mcp.revert(proposal_id)
    else:
        return JSONResponse({"ok": False, "message": "unknown action"}, status_code=400)
    return {"ok": ok, "message": message}


@app.get("/api/metrics")
async def api_metrics():
    """One-shot system metrics, shaped for the HUD."""
    return system_mcp.get_ui_metrics()


@app.get("/api/cosmos")
async def api_cosmos():
    """The neural galaxy: one particle per memory, skill, ability, and mission.

    Skills are the actions the model is allowed to route to; abilities are the
    privileged actions it is not, so the HUD can show them as a separate ring.
    Memories are de-duplicated because the store can accumulate repeats.
    """
    unique_memories = {}
    for entry in memory_mcp.get_facts():
        fact = (entry.get("fact") or "").strip()
        if not fact:
            continue
        key = fact.lower()
        current = unique_memories.get(key)
        if current is None or entry.get("timestamp", "") >= current.get("timestamp", ""):
            unique_memories[key] = {"fact": fact, "timestamp": entry.get("timestamp", "")}
    memories = sorted(unique_memories.values(),
                      key=lambda m: m.get("timestamp", ""), reverse=True)

    skills = [
        {"name": name, "description": description}
        for name, description in sorted(ACTION_CATALOGUE.items())
        if is_routable(name)
    ]
    abilities = [
        {"name": name, "description": description}
        for name, description in sorted(ACTION_CATALOGUE.items())
        if not is_routable(name)
    ]
    tasks = tasks_mcp.get_tasks_for_ui()
    pending = [t for t in tasks if not t.get("done")]

    return {
        "identity": memory_mcp.get_identity(),
        "memories": memories,
        "skills": skills,
        "abilities": abilities,
        "tasks": tasks,
        "stats": {
            "memories": len(memories),
            "skills": len(skills),
            "abilities": len(abilities),
            "missions": len(pending),
        },
    }


@app.get("/api/tts")
async def api_tts(text: str = ""):
    if not text:
        return {"error": "No text provided"}
    try:
        voice_mcp.speak(text)
        return {"status": "queued", "text": text}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/stop")
async def api_stop():
    voice_mcp.shutdown()
    return {"status": "stopped"}


@app.post("/api/ptt")
async def api_ptt(payload: dict):
    """Hold-to-talk, for clients that cannot hold a key down.

    ``{"active": true}`` opens the gate, ``{"active": false}`` closes it and
    transcribes whatever was said. The HUD uses the WebSocket equivalent.

    ``active`` is mandatory: guessing a typo'd payload either way would either
    drop a command mid-sentence or leave the microphone open.
    """
    if not isinstance(payload, dict) or "active" not in payload:
        return JSONResponse({"error": "expected a JSON body with an 'active' boolean"},
                            status_code=400)
    active = payload["active"]
    if not isinstance(active, bool):
        return JSONResponse({"error": "'active' must be true or false"}, status_code=400)

    voice_mcp.set_push_to_talk(active)
    return {"push_to_talk": active, "listening": voice_mcp.is_listening()}


@app.get("/api/listening")
async def api_listening():
    """Speech-recognition state, for the HUD indicator."""
    return {
        "listening": voice_mcp.is_listening(),
        "push_to_talk": voice_mcp.is_push_to_talk(),
        "wake_word": voice_mcp.wake_word,
        "last_transcript": voice_mcp.get_last_transcript(),
        "listen_error": voice_mcp.get_listen_error(),
    }


# ---------------------------------------------------------------------------
# Metrics / task broadcasting
# ---------------------------------------------------------------------------

def broadcast_tasks() -> None:
    """Push the current task list to every connected client."""
    try:
        broadcast({"type": "tasks", "tasks": tasks_mcp.get_tasks_for_ui()})
    except Exception as e:
        log.debug("Task broadcast failed: %s", e)


def _metrics_loop(interval: float = 2.0) -> None:
    """Push system metrics to clients on a fixed cadence."""
    while not _shutdown.is_set():
        try:
            if _websocket_clients:
                broadcast({
                    "type": "metrics",
                    **system_mcp.get_ui_metrics(),
                    # The microphone is always on, so the HUD can say so.
                    "listening": voice_mcp.is_listening(),
                })
            broadcast_tasks()
            try:
                _broadcast_voice_down_if_needed()
            except Exception:
                pass
        except Exception as e:
            log.debug("Metrics loop error: %s", e)
        _shutdown.wait(interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("  %s v%s starting", APP_NAME, VERSION)
    log.info("=" * 60)

    set_status("online")
    
    log.info("Starting WebSocket server on port 8765...")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="info"))
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    
    log.info("Opening browser to http://127.0.0.1:8765...")
    # Try to use Chrome, fallback to default browser
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    chrome_found = False
    for path in chrome_paths:
        if os.path.exists(path):
            webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(path))
            webbrowser.get('chrome').open("http://127.0.0.1:8765")
            chrome_found = True
            log.info("Using Chrome: %s", path)
            break
    if not chrome_found:
        log.info("Chrome not found, using default browser")
        webbrowser.open("http://127.0.0.1:8765")

    # The greeting is spoken after the microphone has finished measuring the
    # room. Playback bleeds back into the mic, and calibrating while the
    # speaker was talking measured that sound as "room noise" and set the
    # speech gate far too high for the rest of the session.
    if voice_mcp.is_listening() and not voice_mcp.wait_for_calibration(15.0):
        log.warning("Room calibration did not finish before speaking the greeting.")

    _say_async("Good to see you, sir.")
    
    log.info("System ready. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        # Say goodbye first: shutting the voice down first would cancel it
        # before it could be heard.
        speak("Goodbye, sir.")
        time.sleep(2)
        voice_mcp.shutdown()
        reminders_mcp.shutdown()
        _shutdown.set()
        server.should_exit = True
        log.info("Shutdown complete.")


if __name__ == "__main__":
    main()
