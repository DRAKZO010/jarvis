import os
import sys
import time
import queue
import json
import logging
import threading
import subprocess
import webbrowser
import tempfile
import shutil
import tkinter as tk
from tkinter import scrolledtext
from pathlib import Path
import psutil
import sounddevice as sd
import numpy as np

import win32com.client
import pythoncom
from vosk import Model, KaldiRecognizer

# --- HARDWARE CORE CONFIGURATIONS ---
SAMPLE_RATE = 16000  # Vosk operates optimally at 16kHz
BLOCK_MS = 100
MIN_RMS = 0.012
SPIKE_RATIO = 2.5
NOISE_FLOOR_ALPHA = 0.992

# --- AUTOMATED WORKSPACE LINKS (FROM YOUR ORIGINAL JARVIS.PY) ---
SONG_URI = "https://open.spotify.com/track/39shmbIHICJ2Wxnk1fPSdz?si=2900c75c2e2d4b82"
CLAUDE_CODE_URL = "https://claude.ai/new"
BINANCE_BTC_URL = "https://www.binance.com/en/trade/BTC_USDT"

CLAUDE_CHROME_MONITOR = 1
BINANCE_CHROME_MONITOR = 3

# UI Styling Matrix
BG_COLOR = "#0D1117"
TEXT_COLOR = "#58A6FF"
ACCENT_COLOR = "#238636"
CONSOLE_COLOR = "#161B22"

log_queue = queue.Queue()
ui_refresh_queue = queue.Queue()
MEMORY_FILE = Path("jarvis_memory.json")
MODEL_PATH = "model"

class QueueLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_queue.put(log_entry)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("Jarvis")
log_handler = QueueLogHandler()
log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
log.addHandler(log_handler)

# --- GLOBAL LIFECYCLE MANAGERS ---
vosk_model = None
is_listening_for_speech = threading.Event()
last_interaction_time = 0.0
WAKE_TIMEOUT_SECONDS = 30.0

# --- IDENTITY AND PROJECT REGISTER ---
def load_system_memory():
    """Reads secure user context parameters and logs tasks without displaying them unprompted"""
    if not MEMORY_FILE.exists():
        default_memory = {
            "secure_identity": {
                "name": "Mohammed Nabeel",
                "role": "Robotics & Automation Engineer",
                "institution": "Park College of Engineering and Technology",
                "clearance_level": "Alpha-01"
            },
            "pending_tasks": [
                "Calibrate structural link parameters for MG90S bipedal locomotion.",
                "Review classification algorithms for disaster zone drone object tracking.",
                "Verify custom electronic layouts and schematic routing configurations."
            ]
        }
        with open(MEMORY_FILE, 'w') as f:
            json.dump(default_memory, f, indent=4)
        return default_memory
    try:
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def update_system_memory(key, data):
    memory = load_system_memory()
    memory[key] = data
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory, f, indent=4)
    except Exception as e:
        log.error("Failed to write data safely to memory matrix: %s", e)

# --- NATIVE WIN32 DISPLAY DETECTION & SNAP MACROS ---
def _win32_sorted_monitor_rects():
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    class RECT(ctypes.Structure):
        _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), 
                    ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

    collected = []
    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(RECT), wintypes.LPARAM)
    def _cb(_hm, _hdc, lprc, _lp):
        r = lprc.contents
        collected.append((int(r.left), int(r.top), int(r.right), int(r.bottom)))
        return True

    ctypes.windll.user32.EnumDisplayMonitors(None, None, _cb, 0)
    collected.sort(key=lambda t: (t[0], t[1]))
    return collected

def _chrome_monitor_bounds(one_based_index):
    rects = _win32_sorted_monitor_rects()
    if not rects:
        return (0, 0, 1920, 1080)
    idx = max(0, min(one_based_index - 1, len(rects) - 1))
    return rects[idx]

def _chrome_top_level_browser_hwnds_win32():
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    found = set()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _lp):
        if user32.GetWindow(hwnd, 4) or (user32.GetWindowLongW(hwnd, -20) & 0x00000080):
            return True
        if not user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        hproc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if hproc:
            buf = ctypes.create_unicode_buffer(4096)
            sz = wintypes.DWORD(len(buf))
            if kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(sz)):
                if os.path.basename(buf.value).lower() == "chrome.exe":
                    found.add(int(hwnd))
            kernel32.CloseHandle(hproc)
        return True

    user32.EnumWindows(_enum, 0)
    return found

def _chrome_snap_window_to_monitor_win32(hwnd, one_based_monitor):
    import ctypes
    ml, mt, mr, mb = _chrome_monitor_bounds(one_based_monitor)
    user32 = ctypes.windll.user32
    user32.ShowWindow(hwnd, 9)  # Restore
    user32.SetWindowPos(hwnd, 0, ml, mt, mr - ml, mb - mt, 0x0040 | 0x0020)
    user32.ShowWindow(hwnd, 3)  # Maximize
    user32.SetForegroundWindow(hwnd)
    user32.keybd_event(0x7A, 0, 0, 0)  # Send F11 key down
    user32.keybd_event(0x7A, 0, 0x0002, 0)  # Send F11 key up

def _open_url_in_chrome(url, label, monitor_idx):
    chrome_path = shutil.which("chrome") or shutil.which("google-chrome")
    if sys.platform == "win32":
        for p in [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]:
            if os.path.isfile(p):
                chrome_path = p
                break
    if not chrome_path:
        webbrowser.open(url)
        return

    before = _chrome_top_level_browser_hwnds_win32() if sys.platform == "win32" else None
    cmd = [chrome_path, "--new-window", url]
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if sys.platform == "win32" and before is not None:
        def _snap_job():
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                time.sleep(0.2)
                now = _chrome_top_level_browser_hwnds_win32()
                new_hwnds = now - before
                if new_hwnds:
                    _chrome_snap_window_to_monitor_win32(list(new_hwnds)[0], monitor_idx)
                    break
        threading.Thread(target=_snap_job, daemon=True).start()

# --- AUTOMATION INITIALIZERS ---
def spawn_workspace_layout():
    """Launches full system layout: Spotify ambient track, Claude AI, and Binance market trackers"""
    log.info("🎼 Initiating ambient audio sequences...")
    if sys.platform == "win32":
        os.startfile(SONG_URI)
    else:
        webbrowser.open(SONG_URI)
    
    time.sleep(1.0)
    log.info(f"🌐 Deploying Claude Enterprise platform onto monitor {CLAUDE_CHROME_MONITOR}...")
    _open_url_in_chrome(CLAUDE_CODE_URL, "Claude", CLAUDE_CHROME_MONITOR)
    
    log.info(f"📈 Syncing Binance live telemetry streams onto monitor {BINANCE_CHROME_MONITOR}...")
    _open_url_in_chrome(BINANCE_BTC_URL, "Binance", BINANCE_CHROME_MONITOR)

def open_cursor_ide():
    """Brings local Cursor window into immediate screen focus and activates fullscreen mapping"""
    exe_path = shutil.which("cursor")
    if sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            p = os.path.join(local_app, "Programs", "cursor", "Cursor.exe")
            if os.path.isfile(p):
                exe_path = p
    if not exe_path:
        log.warning("Cursor IDE application execution file pathway not found.")
        return

    log.info("💻 Deploying integrated workspace compilation systems...")
    subprocess.Popen([exe_path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# --- THREAD-SAFE SAPI SPEECH SYNTHESIS ---
def speak_output(text: str, wait=False):
    def _speak_worker(speech_text):
        try:
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(speech_text)
        except Exception as e:
            log.error("SAPI Native Driver Fault: %s", e)
        finally:
            pythoncom.CoUninitialize()
            
    t = threading.Thread(target=_speak_worker, args=(text,), daemon=True)
    t.start()
    if wait:
        t.join()

# --- CENTRAL ROUTER ENGINE ---
def execute_system_action_routine(clean_input: str, user_text: str, tasks: list) -> str:
    memory = load_system_memory()
    
    # 1. Verification Sequence Challenge
    if "verify identity" in clean_input or "who am i" in clean_input or "reveal identity" in clean_input:
        ident = memory.get("secure_identity", {})
        return f"Identity confirmation match verified. Operator profile parsed as {ident.get('name')}, second-year candidate specializing in {ident.get('role')} at {ident.get('institution')}. Clearance level: {ident.get('clearance_level', 'Unknown')}."

    # 2. Automated Custom Layout Deployments (From your script)
    elif "initialize workspace" in clean_input or "open code" in clean_input or "launch project" in clean_input or "deploy systems" in clean_input:
        threading.Thread(target=spawn_workspace_layout, daemon=True).start()
        threading.Thread(target=open_cursor_ide, daemon=True).start()
        return "Command verified, Boss. Initiating secure environment layouts, spawning browser telemetry modules, and executing local development windows now."

    # 3. Environment Performance Telemetry
    elif "system status" in clean_input or "diagnostic check" in clean_input or "telemetry" in clean_input:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        return f"Core engine diagnostics show processor metrics at {cpu_usage}% with volatile memory consumption running at {memory_usage}%, Boss. Local terminal link online and monitoring operational parameters."

    # 4. Long-Term Workflow Additions
    elif "add task" in clean_input or "log objective" in clean_input or "remind me to" in clean_input:
        new_item = user_text.replace("add task", "").replace("Add task", "").replace("log objective", "").replace("remind me to", "").strip()
        if new_item:
            tasks.append(new_item)
            update_system_memory("pending_tasks", tasks)
            return f"Understood, Boss. I have written that entry directly into your pending workspace array: '{new_item}'."
        return "Please define explicit objective variables to save to the database logs, Boss."

    # 5. Active Task Inquiries
    elif "what are my tasks" in clean_input or "read objectives" in clean_input or "current workflows" in clean_input:
        if tasks:
            return f"We have {len(tasks)} logged workflows currently standing in the pipeline, Boss. Primary target focus is: {tasks[0]}."
        return "Your scheduled workflow database is entirely clear of items, Boss."

    elif "clear tasks" in clean_input or "flush objectives" in clean_input:
        update_system_memory("pending_tasks", [])
        return "Workspace records successfully updated, Boss. Task ledger flushed."

    elif "hello" in clean_input or "jarvis" in clean_input:
        return "At your service, Boss. Core arrays are active and tracking microphone inputs."

    return "Linguistic expression successfully mapped, Boss, but no automated software routing macros matched this instruction."

def interpret_and_respond(user_text: str):
    global last_interaction_time
    if not user_text.strip():
        return

    last_interaction_time = time.time()
    log.info(f"User: \"{user_text}\"")
    
    memory = load_system_memory()
    tasks = memory.get("pending_tasks", [])
    clean_input = user_text.lower()

    if "stand down" in clean_input or "go to sleep" in clean_input or "terminate session" in clean_input:
        msg = "Acknowledged, Boss. Reverting interface channels to quiet background telemetry. Let me know when you need assistance."
        log.info(f"Jarvis: \"{msg}\"")
        speak_output(msg)
        is_listening_for_speech.clear()
        return

    response = execute_system_action_routine(clean_input, user_text, tasks)
    log.info(f"Jarvis: \"{response}\"")
    
    is_listening_for_speech.clear()
    speak_output(response, wait=True)
    
    if "reverting interface channels" not in response.lower():
        last_interaction_time = time.time()
        is_listening_for_speech.set()

def watch_conversation_timeout():
    global last_interaction_time
    while True:
        if is_listening_for_speech.is_set():
            if time.time() - last_interaction_time > WAKE_TIMEOUT_SECONDS:
                is_listening_for_speech.clear()
                log.info("🔒 Inactivity timeout reached. Dropping down to raw amplitude monitoring mode.")
                speak_output("Standing down. System loops running silently in the background, Boss.")
        time.sleep(1)

def trigger_initial_welcome():
    global last_interaction_time
    msg = "Main grid parameters loaded successfully, Boss. Environment controls are online and stable. Waiting for your verbal directives."
    log.info("Accessing native memory registries and checking local arrays...")
    speak_output(msg, wait=True)
    
    last_interaction_time = time.time()
    log.info("🎤 Continuous conversation interface active. Speak your command...")
    is_listening_for_speech.set()

# --- INTEGRATED DUAL-MODE AUDIO INGESTION CORE ---
def audio_orchestrator_worker():
    global vosk_model
    
    if os.path.exists(MODEL_PATH):
        try:
            log.info("Loading speech recognition matrix from local path...")
            vosk_model = Model(MODEL_PATH)
            log.info("🎉 Speech engine cached successfully. STT interfaces are live and tracking.")
        except Exception as e:
            log.error("Failed to parse local Vosk weights: %s", e)
    else:
        log.warning("❌ CRITICAL INGESTION PATHWAY FAILURE: Vosk model path missing! Voice streaming unavailable. Rename folder to exactly 'model'.")

    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE) if vosk_model else None
    noise_floor = 0.005
    first_clap_time = None
    
    def callback(indata, frames, time_info, status):
        nonlocal noise_floor, first_clap_time
        raw_samples = indata[:, 0]
        rms = np.sqrt(np.mean(raw_samples**2))
        
        # Continuous Speech Capture Logic
        if is_listening_for_speech.is_set() and recognizer is not None:
            pcm_data = (raw_samples * 32767).astype(np.int16).tobytes()
            if recognizer.AcceptWaveform(pcm_data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text.strip():
                    threading.Thread(target=interpret_and_respond, args=(text,), daemon=True).start()
            return

        # Baseline Amplitude Double-Clap Trigger Logic
        if rms < noise_floor * 1.5:
            noise_floor = (NOISE_FLOOR_ALPHA * noise_floor) + ((1.0 - NOISE_FLOOR_ALPHA) * rms)
            
        threshold = max(MIN_RMS, noise_floor * SPIKE_RATIO)
        
        if rms > threshold:
            now = time.time()
            if first_clap_time is None:
                first_clap_time = now
            else:
                gap = now - first_clap_time
                if 0.2 <= gap <= 0.8:
                    first_clap_time = None
                    if not is_listening_for_speech.is_set():
                        log.info("Double Clap Confirmed! Awakening system conversation engine...")
                        threading.Thread(target=trigger_initial_welcome, daemon=True).start()
                elif gap > 0.8:
                    first_clap_time = now

    block_samples = int(SAMPLE_RATE * BLOCK_MS / 1000)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=block_samples, callback=callback):
        while True:
            time.sleep(1)

# --- GUI DASHBOARD MATRIX ---
class JarvisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("J.A.R.V.I.S. Command Matrix")
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        panel_width = 450
        self.geometry(f"{panel_width}x{screen_height-80}+{screen_width-panel_width-10}+10")
        self.configure(bg=BG_COLOR)
        self.attributes("-topmost", True)  

        title_lbl = tk.Label(self, text="⚡ J.A.R.V.I.S. SYSTEM CORE", font=("Consolas", 14, "bold"), fg=TEXT_COLOR, bg=BG_COLOR)
        title_lbl.pack(pady=10)

        self.diag_frame = tk.LabelFrame(self, text=" GRID METRICS ", font=("Consolas", 10, "bold"), fg=ACCENT_COLOR, bg=BG_COLOR, bd=1)
        self.diag_frame.pack(fill="x", padx=15, pady=5)
        
        self.cpu_lbl = tk.Label(self.diag_frame, text="CPU Usage: Polling...", font=("Consolas", 11), fg="#C9D1D9", bg=BG_COLOR, anchor="w")
        self.cpu_lbl.pack(fill="x", padx=10, pady=2)
        
        self.ram_lbl = tk.Label(self.diag_frame, text="RAM Usage: Polling...", font=("Consolas", 11), fg="#C9D1D9", bg=BG_COLOR, anchor="w")
        self.ram_lbl.pack(fill="x", padx=10, pady=2)

        log_lbl = tk.Label(self, text="⚙️ SYSTEM ACTIVITY LOGS", font=("Consolas", 10, "bold"), fg=TEXT_COLOR, bg=BG_COLOR)
        log_lbl.pack(anchor="w", padx=15, pady=(15, 2))

        self.console = scrolledtext.ScrolledText(self, font=("Consolas", 9), fg="#A3D1FF", bg=CONSOLE_COLOR, insertbackground="white", bd=0)
        self.console.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.listen_for_updates()

    def listen_for_updates(self):
        while not log_queue.empty():
            try:
                msg = log_queue.get_nowait()
                self.console.insert(tk.END, msg + "\n")
                self.console.see(tk.END)
            except queue.Empty:
                break
        
        while not ui_refresh_queue.empty():
            try:
                m_type, value = ui_refresh_queue.get_nowait()
                if m_type == "CPU":
                    self.cpu_lbl.config(text=f"CPU Metrics: {value}%")
                elif m_type == "RAM":
                    self.ram_lbl.config(text=f"RAM Allocation: {value}%")
            except queue.Empty:
                break
                
        self.after(100, self.listen_for_updates)

def run_diagnostics_loop():
    while True:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        ui_refresh_queue.put(("CPU", cpu))
        ui_refresh_queue.put(("RAM", ram))
        time.sleep(2)

if __name__ == "__main__":
    load_system_memory()
    
    threading.Thread(target=audio_orchestrator_worker, daemon=True).start()
    threading.Thread(target=run_diagnostics_loop, daemon=True).start()
    threading.Thread(target=watch_conversation_timeout, daemon=True).start()
    
    app = JarvisApp()
    app.mainloop()
