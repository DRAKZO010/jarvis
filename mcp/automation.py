"""Automation MCP - Full computer control, apps, file ops, system commands."""

from __future__ import annotations
import logging
import os
import re
import shutil
import subprocess
try:
    import pyautogui
    import win32gui
    import win32con
    import win32api
except ImportError:
    pyautogui = None
    win32gui = None
    win32con = None
    win32api = None
from typing import Any, Dict, Optional

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.automation")

KNOWN_APPS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "word": "winword.exe",
    "excel": "excel.exe",
    "powerpoint": "powerpnt.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "task manager": "taskmgr.exe",
    "command prompt": "cmd.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "settings": "ms-settings:",
    "snipping tool": "snippingtool.exe",
    "camera": "microsoft.windows.camera:",
    "maps": "bingmaps:",
    "edge": "msedge.exe",
    "chrome": "chrome.exe",
    "firefox": "firefox.exe",
    "vscode": "code",
    "visual studio code": "code",
    "terminal": "wt.exe",
    "spotify": "spotify:",
    "discord": "discord.exe",
    "slack": "slack.exe",
    "teams": "ms-teams:",
    "zoom": "zoom.exe",
    "obs": "obs64.exe",
    "steam": "steam.exe",
    "blender": "blender.exe",
    "photoshop": "photoshop.exe",
    "illustrator": "illustrator.exe",
    "premiere": "premiere.exe",
    "after effects": "afterfx.exe",
    "audacity": "audacity.exe",
    "gimp": "gimp.exe",
    "7zip": "7z.exe",
    "winrar": "winrar.exe",
    "vlc": "vlc.exe",
    "itunes": "itunes.exe",
    "whatsapp": "whatsapp:",
    "telegram": "telegram:",
    "signal": "signal:",
    "thunderbird": "thunderbird.exe",
    "pdf": "acrobat.exe",
    "acrobat": "acrobat.exe",
    "reader": "AcroRd32.exe",
    "sublime": "sublime_text.exe",
    "sublime text": "sublime_text.exe",
    "atom": "atom.exe",
    "intellij": "idea64.exe",
    "pycharm": "pycharm64.exe",
    "webstorm": "webstorm64.exe",
    "android studio": "studio64.exe",
    "xcode": "xcode.exe",
    "unity": "unity.exe",
    "unreal": "unrealeditor.exe",
    "godot": "godot.exe",
}

# Dangerous commands that require confirmation
DANGEROUS_PATTERNS = [
    r"del\s+[a-z]:\\",
    r"rmdir\s+[a-z]:\\",
    r"format\s+[a-z]:",
    r"rm\s+-rf\s+/",
    r"shutdown",
    r"restart",
    r"logoff",
]


class AutomationMCP(MCPBase):
    """System automation MCP module."""

    name = "automation"
    description = "System automation, apps, brightness, volume"
    commands = [
        "open app",
        "launch",
        "start",
        "set brightness",
        "brightness",
        "set volume",
        "volume",
        "execute",
        "run",
    ]

    def __init__(self):
        super().__init__()
        self._pending_confirmation: Optional[Dict[str, str]] = None

    def initialize(self) -> bool:
        """Initialize automation module."""
        self._initialized = True
        log.info("Automation MCP initialized.")
        return True

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle automation commands."""
        if command in ("open app", "launch", "start"):
            return self.open_app(args)

        if command in ("set brightness", "brightness"):
            if args and args.isdigit():
                return self.set_brightness(int(args))
            return "Please specify a brightness level (0-100), sir."

        if command in ("set volume", "volume"):
            if args and args.isdigit():
                return self.set_volume(int(args))
            return "Please specify a volume level (0-100), sir."

        if command in ("execute", "run"):
            if args:
                return self.execute_command(args)
            return "What command should I execute, sir?"

        return None

    def open_app(self, app_name: str) -> str:
        """Open an application or website."""
        if not app_name:
            return "What should I open, sir?"

        app = app_name.lower().strip()
        
        # Check known websites first (browser module)
        from mcp.browser import BrowserMCP
        browser = BrowserMCP()
        
        # Check if Spotify is mentioned - redirect to Spotify handler
        if "spotify" in app:
            # Extract song if mentioned (e.g., "spotify and play song" or "spotify song name")
            song = ""
            if "play" in app:
                # Try to extract song name after "play"
                match = re.search(r'play\s+(.+?)(?:\s+on\s+spotify|$)', app)
                if match:
                    song = match.group(1).strip()
            return browser.play_spotify(song)
        
        # Check if it looks like a URL
        if any(x in app for x in (".com", ".org", ".net", ".io", "www", "http")):
            return browser.open(app_name)
        
        # Check known apps
        if app in KNOWN_APPS:
            cmd = KNOWN_APPS[app]
            if cmd.startswith("ms-") or cmd.startswith("bing"):
                os.system(f'start "" "{cmd}"')
            else:
                subprocess.Popen([cmd], shell=False)
            return f"Opening {app_name}, sir."

        # Try to find and launch
        try:
            subprocess.Popen(["start", "", app_name], shell=True, executable="cmd.exe")
            return f"Attempting to launch {app_name}."
        except Exception:
            return f"I couldn't find an application called {app_name}, sir."

    def set_brightness(self, level: int) -> str:
        """Set screen brightness."""
        level = max(0, min(100, level))
        try:
            cmd = (
                f"(Get-WmiObject -Namespace root/wmi "
                f"-Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, {level})"
            )
            subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True, timeout=5,
            )
            return f"Brightness set to {level} percent, sir."
        except Exception:
            return "Unable to adjust brightness, sir."

    def set_volume(self, level: int) -> str:
        """Set system volume using PowerShell AudioDevice module."""
        level = max(0, min(100, level))
        try:
            # Convert percentage to 0-65535 range for Windows
            volume_value = int(level * 65535 / 100)
            
            # Use PowerShell to set volume via Windows Audio Session API
            cmd = f"""
            Add-Type -TypeDefinition '
            using System;
            using System.Runtime.InteropServices;
            public class Audio {{
                [DllImport("winmm.dll")]
                public static extern int waveOutSetVolume(IntPtr hwo, uint dwVolume);
                
                [DllImport("winmm.dll")]
                public static extern int waveOutGetVolume(IntPtr hwo, out uint dwVolume);
            }}
            '
            $volume = {volume_value}
            [Audio]::waveOutSetVolume([IntPtr]::Zero, $volume)
            """
            subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True, timeout=5,
            )
            return f"Volume set to {level} percent, sir."
        except Exception as e:
            log.error("Volume set failed: %s", e)
            return "Unable to adjust volume, sir."

    def execute_command(self, cmd: str) -> str:
        """Execute a system command with safety checks."""
        # Check for dangerous commands
        cmd_lower = cmd.lower().strip()
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, cmd_lower):
                return f"Command blocked for safety: {cmd[:50]}..."

        # Block certain dangerous commands entirely
        blocked = ["format", "del /s", "rmdir /s", "rm -rf"]
        if any(b in cmd_lower for b in blocked):
            return "This command is too dangerous, sir. I cannot execute it."

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=15
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                output = result.stderr.strip() or "Command failed."
            if len(output) > 500:
                output = output[:500] + "..."
            return output or "Command executed successfully."
        except subprocess.TimeoutExpired:
            return "Command timed out."
        except FileNotFoundError:
            return f"Command not found: {cmd.split()[0]}"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "known_apps": len(KNOWN_APPS),
        }

    # ---- File Operations ----
    def create_file(self, path: str, content: str = "") -> str:
        """Create a file with optional content."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File created: {path}"
        except Exception as e:
            return f"Error creating file: {e}"

    def read_file(self, path: str) -> str:
        """Read file contents."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > 2000:
                content = content[:2000] + "... (truncated)"
            return content or "File is empty."
        except Exception as e:
            return f"Error reading file: {e}"

    def delete_file(self, path: str) -> str:
        """Delete a file."""
        try:
            if os.path.exists(path):
                os.remove(path)
                return f"Deleted: {path}"
            return f"File not found: {path}"
        except Exception as e:
            return f"Error deleting file: {e}"

    def copy_file(self, src: str, dst: str) -> str:
        """Copy a file."""
        try:
            shutil.copy2(src, dst)
            return f"Copied {src} to {dst}"
        except Exception as e:
            return f"Error copying file: {e}"

    def move_file(self, src: str, dst: str) -> str:
        """Move a file."""
        try:
            shutil.move(src, dst)
            return f"Moved {src} to {dst}"
        except Exception as e:
            return f"Error moving file: {e}"

    def list_directory(self, path: str = ".") -> str:
        """List directory contents."""
        try:
            items = os.listdir(path)
            dirs = [f"[DIR] {i}" for i in items if os.path.isdir(os.path.join(path, i))]
            files = [i for i in items if os.path.isfile(os.path.join(path, i))]
            return "\n".join(dirs[:20] + files[:20]) or "Directory is empty."
        except Exception as e:
            return f"Error listing directory: {e}"

    # ---- Window Management ----
    def minimize_window(self) -> str:
        """Minimize active window."""
        try:
            hwnd = win32gui.GetForegroundWindow()
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            return "Window minimized."
        except Exception as e:
            return f"Error minimizing: {e}"

    def maximize_window(self) -> str:
        """Maximize active window."""
        try:
            hwnd = win32gui.GetForegroundWindow()
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            return "Window maximized."
        except Exception as e:
            return f"Error maximizing: {e}"

    def close_window(self) -> str:
        """Close active window."""
        try:
            hwnd = win32gui.GetForegroundWindow()
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return "Window closed."
        except Exception as e:
            return f"Error closing: {e}"

    def focus_window(self, title: str) -> str:
        """Focus window by title."""
        try:
            def enum_callback(hwnd, results):
                if win32gui.IsWindowVisible(hwnd):
                    t = win32gui.GetWindowText(hwnd).lower()
                    if title.lower() in t:
                        results.append(hwnd)
            windows = []
            win32gui.EnumWindows(enum_callback, windows)
            if windows:
                hwnd = windows[0]
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                return f"Focused: {title}"
            return f"Window not found: {title}"
        except Exception as e:
            return f"Error focusing: {e}"

    def list_windows(self) -> str:
        """List all open windows."""
        try:
            windows = []
            def enum_callback(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title:
                        windows.append(title)
            win32gui.EnumWindows(enum_callback, None)
            return "\n".join(windows[:30]) or "No windows found."
        except Exception as e:
            return f"Error listing windows: {e}"

    # ---- Process Management ----
    def kill_process(self, name: str) -> str:
        """Kill a process by name."""
        try:
            subprocess.run(["taskkill", "/F", "/IM", name], capture_output=True)
            return f"Killed process: {name}"
        except Exception as e:
            return f"Error killing process: {e}"

    def list_processes(self) -> str:
        """List running processes."""
        try:
            result = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")
            procs = []
            for line in lines[:20]:
                parts = line.split('","')
                if len(parts) >= 2:
                    procs.append(f"{parts[0]} - {parts[1]} KB")
            return "\n".join(procs) or "No processes found."
        except Exception as e:
            return f"Error listing processes: {e}"

    # ---- Screenshot ----
    def take_screenshot(self, save_path: str = "") -> str:
        """Take a screenshot and save to screenshots folder."""
        try:
            import os
            from datetime import datetime
            
            # Create screenshots folder
            screenshots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(screenshots_dir, filename)
            
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            return f"Screenshot saved: {filename}"
        except Exception as e:
            return f"Error taking screenshot: {e}"

    def delete_screenshots(self) -> str:
        """Delete all screenshots."""
        try:
            import os
            import glob
            
            screenshots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screenshots")
            if not os.path.exists(screenshots_dir):
                return "No screenshots folder found."
            
            files = glob.glob(os.path.join(screenshots_dir, "*.png"))
            if not files:
                return "No screenshots to delete."
            
            count = len(files)
            for f in files:
                os.remove(f)
            return f"Deleted {count} screenshot(s), sir."
        except Exception as e:
            return f"Error deleting screenshots: {e}"

    def delete_last_screenshot(self) -> str:
        """Delete the most recent screenshot."""
        try:
            import os
            import glob
            
            screenshots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screenshots")
            if not os.path.exists(screenshots_dir):
                return "No screenshots folder found."
            
            files = sorted(glob.glob(os.path.join(screenshots_dir, "*.png")))
            if not files:
                return "No screenshots to delete."
            
            last_file = files[-1]
            os.remove(last_file)
            return f"Deleted: {os.path.basename(last_file)}"
        except Exception as e:
            return f"Error deleting screenshot: {e}"

    # ---- Clipboard ----
    def get_clipboard(self) -> str:
        """Get clipboard content."""
        try:
            import win32clipboard
            win32clipboard.OpenClipboard()
            data = win32clipboard.GetClipboardData()
            win32clipboard.CloseClipboard()
            return data or "Clipboard is empty."
        except Exception as e:
            return f"Error getting clipboard: {e}"

    def set_clipboard(self, text: str) -> str:
        """Set clipboard content."""
        try:
            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text)
            win32clipboard.CloseClipboard()
            return "Clipboard updated."
        except Exception as e:
            return f"Error setting clipboard: {e}"

    # ---- System Commands ----
    def shutdown_pc(self) -> str:
        """Shutdown the computer."""
        try:
            subprocess.run(["shutdown", "/s", "/t", "60"], capture_output=True)
            return "Shutting down in 60 seconds. Say 'cancel shutdown' to abort."
        except Exception as e:
            return f"Error: {e}"

    def restart_pc(self) -> str:
        """Restart the computer."""
        try:
            subprocess.run(["shutdown", "/r", "/t", "60"], capture_output=True)
            return "Restarting in 60 seconds. Say 'cancel shutdown' to abort."
        except Exception as e:
            return f"Error: {e}"

    def cancel_shutdown(self) -> str:
        """Cancel scheduled shutdown."""
        try:
            subprocess.run(["shutdown", "/a"], capture_output=True)
            return "Shutdown cancelled."
        except Exception as e:
            return f"Error: {e}"

    def lock_pc(self) -> str:
        """Lock the computer."""
        try:
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], capture_output=True)
            return "Computer locked."
        except Exception as e:
            return f"Error: {e}"

    def sleep_pc(self) -> str:
        """Put computer to sleep."""
        try:
            subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], capture_output=True)
            return "Going to sleep."
        except Exception as e:
            return f"Error: {e}"

    def hibernate_pc(self) -> str:
        """Hibernate the computer."""
        try:
            subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "1,1,0"], capture_output=True)
            return "Hibernating."
        except Exception as e:
            return f"Error: {e}"

    # ---- Mouse/Keyboard ----
    def mouse_click(self, x: int, y: int) -> str:
        """Click at coordinates."""
        try:
            pyautogui.click(x, y)
            return f"Clicked at ({x}, {y})"
        except Exception as e:
            return f"Error: {e}"

    def mouse_move(self, x: int, y: int) -> str:
        """Move mouse to coordinates."""
        try:
            pyautogui.moveTo(x, y)
            return f"Mouse moved to ({x}, {y})"
        except Exception as e:
            return f"Error: {e}"

    def type_text(self, text: str) -> str:
        """Type text using keyboard."""
        try:
            import time
            time.sleep(0.3)  # Small delay to ensure window is focused
            # Use clipboard for better compatibility
            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text)
            win32clipboard.CloseClipboard()
            # Paste with Ctrl+V
            pyautogui.hotkey('ctrl', 'v')
            return f"Typed: {text}"
        except Exception as e:
            return f"Error typing: {e}"

    def press_key(self, key: str) -> str:
        """Press a key."""
        try:
            pyautogui.press(key)
            return f"Pressed: {key}"
        except Exception as e:
            return f"Error: {e}"

    def hotkey(self, keys: str) -> str:
        """Press hotkey combination (e.g., 'ctrl+c')."""
        try:
            key_list = keys.split("+")
            pyautogui.hotkey(*key_list)
            return f"Pressed: {keys}"
        except Exception as e:
            return f"Error: {e}"
