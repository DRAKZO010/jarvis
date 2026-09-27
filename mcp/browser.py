"""Browser MCP - Web browsing, search, and internet features."""

from __future__ import annotations
import logging
import os
import re
import webbrowser
from typing import Any, Dict, Optional

import requests

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.browser")

# Register Chrome as browser
_chrome_paths = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]
_chrome_registered = False
for _path in _chrome_paths:
    if os.path.exists(_path):
        try:
            webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(_path))
            _chrome_registered = True
            log.info("Chrome registered: %s", _path)
            break
        except:
            pass


def _open_url(url: str) -> None:
    """Open URL using Chrome if available, else default browser."""
    if _chrome_registered:
        webbrowser.get('chrome').open(url)
    else:
        webbrowser.open(url)


class BrowserMCP(MCPBase):
    """Browser and web MCP module."""

    name = "browser"
    description = "Web browsing, search, and internet features"
    commands = [
        "open",
        "search",
        "google",
        "youtube",
        "gmail",
        "email",
        "maps",
        "read",
        "fetch",
    ]

    KNOWN_WEBSITES = {
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
        "gmail": "https://mail.google.com",
        "email": "https://mail.google.com",
        "outlook": "https://outlook.live.com",
        "twitter": "https://x.com",
        "x": "https://x.com",
        "instagram": "https://www.instagram.com",
        "facebook": "https://www.facebook.com",
        "reddit": "https://www.reddit.com",
        "github": "https://github.com",
        "stackoverflow": "https://stackoverflow.com",
        "netflix": "https://www.netflix.com",
        "amazon": "https://www.amazon.com",
        "wikipedia": "https://www.wikipedia.org",
        "chatgpt": "https://chat.openai.com",
        "claude": "https://claude.ai",
        "spotify": "https://open.spotify.com",
        "linkedin": "https://www.linkedin.com",
        "maps": "https://maps.google.com",
        "news": "https://news.google.com",
        "discord": "https://discord.com",
        "twitch": "https://www.twitch.tv",
        "tiktok": "https://www.tiktok.com",
        "pinterest": "https://www.pinterest.com",
        "quora": "https://www.quora.com",
        "medium": "https://medium.com",
        "linkedin": "https://www.linkedin.com",
        "slack": "https://slack.com",
        "notion": "https://www.notion.so",
        "figma": "https://www.figma.com",
        "canva": "https://www.canva.com",
        "duolingo": "https://www.duolingo.com",
        "coursera": "https://www.coursera.org",
        "udemy": "https://www.udemy.com",
        "leetcode": "https://leetcode.com",
        "geeksforgeeks": "https://www.geeksforgeeks.org",
        "w3schools": "https://www.w3schools.com",
        "mdn": "https://developer.mozilla.org",
    }

    def initialize(self) -> bool:
        """Initialize browser module."""
        self._initialized = True
        log.info("Browser MCP initialized.")
        return True

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle browser commands."""
        if command == "open":
            return self.open(args)

        if command in ("search", "google"):
            return self.search(args)

        if command == "youtube":
            return self.open_youtube(args)

        if command in ("gmail", "email"):
            return self.open_gmail()

        if command == "maps":
            return self.open_maps(args)

        if command in ("read", "fetch"):
            return self.read_webpage(args)

        return None

    def open(self, target: str) -> str:
        """Open a website or application."""
        if not target:
            return "What should I open, sir?"

        target_lower = target.lower().strip()

        # Check known websites
        if target_lower in self.KNOWN_WEBSITES:
            url = self.KNOWN_WEBSITES[target_lower]
            try:
                _open_url(url)
                return f"Opening {target}, sir."
            except Exception:
                return f"Unable to open {target}, sir."

        # Check if it's a URL
        if any(x in target_lower for x in (".com", ".org", ".net", ".io", "www")):
            url = target if target.startswith("http") else f"https://{target}"
            try:
                _open_url(url)
                return f"Opening {url}, sir."
            except Exception:
                return f"Unable to open {url}, sir."

        # Try as application
        try:
            _open_url(target)
            return f"Attempting to open {target}, sir."
        except Exception:
            return f"I couldn't find {target}, sir."

    def search(self, query: str) -> str:
        """Search the web."""
        if not query:
            return "What should I search for, sir?"
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        try:
            _open_url(url)
            return f"Searching for {query}, sir."
        except Exception:
            return f"Unable to search for {query}, sir."

    def open_youtube(self, query: str = "") -> str:
        """Open YouTube or search YouTube."""
        if query:
            url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        else:
            url = "https://www.youtube.com"
        try:
            _open_url(url)
            return f"Opening YouTube{f' for {query}' if query else ''}, sir."
        except Exception:
            return "Unable to open YouTube, sir."

    def play_spotify(self, query: str = "") -> str:
        """Play music on Spotify desktop app."""
        import time
        import subprocess
        import os
        
        if query:
            # Search for song using Spotify URI
            subprocess.Popen(
                f'start "" "spotify:search:{query}"',
                shell=True, executable="cmd.exe"
            )
            return f"Searching for {query} on Spotify, sir."
        else:
            # Try to find Spotify window and press space to resume
            try:
                import win32gui
                import win32con
                import win32api
                
                # Find Spotify window
                def enum_callback(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd).lower()
                        if "spotify" in title:
                            results.append(hwnd)
                
                windows = []
                win32gui.EnumWindows(enum_callback, windows)
                
                if windows:
                    hwnd = windows[0]
                    title = win32gui.GetWindowText(hwnd)
                    log.info("Found Spotify: %s", title)
                    # Restore if minimized
                    if win32gui.IsIconic(hwnd):
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    # Bring to front using Alt trick
                    win32api.keybd_event(0x12, 0, 0, 0)  # Alt down
                    win32gui.SetForegroundWindow(hwnd)
                    win32api.keybd_event(0x12, 0, win32con.KEYEVENTF_KEYUP, 0)  # Alt up
                    time.sleep(0.3)
                    # Press space to toggle play/pause
                    win32api.keybd_event(0x20, 0, 0, 0)
                    time.sleep(0.05)
                    win32api.keybd_event(0x20, 0, win32con.KEYEVENTF_KEYUP, 0)
                    return "Resuming Spotify, sir."
                else:
                    # Open Spotify desktop app and wait for it
                    log.info("Opening Spotify desktop app")
                    # Try common install paths
                    spotify_paths = [
                        os.path.expandvars(r"%APPDATA%\Spotify\Spotify.exe"),
                        r"C:\Users\Public\Desktop\Spotify.lnk",
                        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps\Spotify.exe"),
                    ]
                    opened = False
                    for path in spotify_paths:
                        if os.path.exists(path):
                            subprocess.Popen([path])
                            opened = True
                            break
                    if not opened:
                        try:
                            subprocess.Popen(["explorer.exe", "spotify:"])
                        except:
                            pass
                    # Wait for Spotify to load with retry
                    for attempt in range(10):  # Try for 10 seconds
                        time.sleep(1)
                        windows = []
                        win32gui.EnumWindows(enum_callback, windows)
                        if windows:
                            hwnd = windows[0]
                            if win32gui.IsIconic(hwnd):
                                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                            win32api.keybd_event(0x12, 0, 0, 0)
                            win32gui.SetForegroundWindow(hwnd)
                            win32api.keybd_event(0x12, 0, win32con.KEYEVENTF_KEYUP, 0)
                            time.sleep(0.3)
                            win32api.keybd_event(0x20, 0, 0, 0)
                            time.sleep(0.05)
                            win32api.keybd_event(0x20, 0, win32con.KEYEVENTF_KEYUP, 0)
                            log.info("Spotify ready after %d seconds", attempt + 1)
                            return "Opening and playing Spotify, sir."
                    return "Opening Spotify, sir."
            except Exception as e:
                log.error("Spotify error: %s", e)
                # Fallback: try to open
                try:
                    subprocess.Popen(["explorer.exe", "spotify:"])
                except:
                    pass
                return "Opening Spotify, sir."

    def spotify_control(self, action: str, value: int = 0) -> str:
        """Control Spotify playback using global media keys."""
        import time
        import win32api
        import win32con
        
        log.info("Spotify control: %s (value=%d)", action, value)
        
        if action in ("play", "pause"):
            win32api.keybd_event(0xB3, 0, 0, 0)
            time.sleep(0.05)
            win32api.keybd_event(0xB3, 0, win32con.KEYEVENTF_KEYUP, 0)
            log.info("Sent media play/pause")
            return "Toggling playback, sir."
        
        elif action == "next":
            win32api.keybd_event(0xB0, 0, 0, 0)
            time.sleep(0.05)
            win32api.keybd_event(0xB0, 0, win32con.KEYEVENTF_KEYUP, 0)
            log.info("Sent media next")
            return "Next track, sir."
        
        elif action == "previous":
            win32api.keybd_event(0xB1, 0, 0, 0)
            time.sleep(0.05)
            win32api.keybd_event(0xB1, 0, win32con.KEYEVENTF_KEYUP, 0)
            log.info("Sent media previous")
            return "Previous track, sir."
        
        elif action == "volume_up":
            win32api.keybd_event(0xAF, 0, 0, 0)
            time.sleep(0.05)
            win32api.keybd_event(0xAF, 0, win32con.KEYEVENTF_KEYUP, 0)
            return "Volume up, sir."
        
        elif action == "volume_down":
            win32api.keybd_event(0xAE, 0, 0, 0)
            time.sleep(0.05)
            win32api.keybd_event(0xAE, 0, win32con.KEYEVENTF_KEYUP, 0)
            return "Volume down, sir."
        
        elif action == "volume_set":
            # Set volume to specific level using keyboard shortcuts
            # Mute first to get current state, then use volume up/down to reach target
            target = max(0, min(100, value))
            log.info("Setting volume to %d%%", target)
            
            # Use Windows audio session API if available, otherwise use volume keys
            try:
                # Try using nircmd or direct volume control
                # Calculate steps (each volume key press = ~2% change)
                # First mute, then unmute to reset to known state
                # Then press volume up enough times to reach target
                steps = target // 2  # Approximate steps
                for _ in range(50):  # Press down to reset
                    win32api.keybd_event(0xAE, 0, 0, 0)
                    time.sleep(0.01)
                    win32api.keybd_event(0xAE, 0, win32con.KEYEVENTF_KEYUP, 0)
                time.sleep(0.1)
                for _ in range(steps):  # Press up to target
                    win32api.keybd_event(0xAF, 0, 0, 0)
                    time.sleep(0.01)
                    win32api.keybd_event(0xAF, 0, win32con.KEYEVENTF_KEYUP, 0)
            except:
                pass
            return f"Volume set to {target} percent, sir."
        
        elif action == "volume_max":
            # Set volume to maximum
            log.info("Setting volume to max")
            for _ in range(50):  # Press up many times
                win32api.keybd_event(0xAF, 0, 0, 0)
                time.sleep(0.01)
                win32api.keybd_event(0xAF, 0, win32con.KEYEVENTF_KEYUP, 0)
            return "Volume set to maximum, sir."
        
        elif action == "volume_min":
            # Set volume to minimum
            log.info("Setting volume to min")
            for _ in range(50):  # Press down many times
                win32api.keybd_event(0xAE, 0, 0, 0)
                time.sleep(0.01)
                win32api.keybd_event(0xAE, 0, win32con.KEYEVENTF_KEYUP, 0)
            return "Volume set to minimum, sir."
        
        elif action == "mute":
            win32api.keybd_event(0xAD, 0, 0, 0)
            time.sleep(0.05)
            win32api.keybd_event(0xAD, 0, win32con.KEYEVENTF_KEYUP, 0)
            return "Muting, sir."
        
        elif action == "shuffle":
            try:
                import win32gui
                def enum_callback(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd).lower()
                        if "spotify" in title:
                            results.append(hwnd)
                windows = []
                win32gui.EnumWindows(enum_callback, windows)
                if windows:
                    hwnd = windows[0]
                    win32api.keybd_event(0x12, 0, 0, 0)
                    win32gui.SetForegroundWindow(hwnd)
                    win32api.keybd_event(0x12, 0, win32con.KEYEVENTF_KEYUP, 0)
                    time.sleep(0.2)
                    win32api.keybd_event(0x11, 0, 0, 0)
                    win32api.keybd_event(0x53, 0, 0, 0)
                    time.sleep(0.05)
                    win32api.keybd_event(0x53, 0, win32con.KEYEVENTF_KEYUP, 0)
                    win32api.keybd_event(0x11, 0, win32con.KEYEVENTF_KEYUP, 0)
            except:
                pass
            return "Toggling shuffle, sir."
        
        elif action == "repeat":
            try:
                import win32gui
                def enum_callback(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd).lower()
                        if "spotify" in title:
                            results.append(hwnd)
                windows = []
                win32gui.EnumWindows(enum_callback, windows)
                if windows:
                    hwnd = windows[0]
                    win32api.keybd_event(0x12, 0, 0, 0)
                    win32gui.SetForegroundWindow(hwnd)
                    win32api.keybd_event(0x12, 0, win32con.KEYEVENTF_KEYUP, 0)
                    time.sleep(0.2)
                    win32api.keybd_event(0x11, 0, 0, 0)
                    win32api.keybd_event(0x52, 0, 0, 0)
                    time.sleep(0.05)
                    win32api.keybd_event(0x52, 0, win32con.KEYEVENTF_KEYUP, 0)
                    win32api.keybd_event(0x11, 0, win32con.KEYEVENTF_KEYUP, 0)
            except:
                pass
            return "Toggling repeat, sir."
        
        return "Unknown command, sir."

    def ask_song_or_random(self) -> str:
        """Ask what song to play or suggest random based on time."""
        from datetime import datetime
        hour = datetime.now().hour
        
        # Time-based suggestions
        if 5 <= hour < 12:
            mood = "morning"
            suggestions = ["good morning songs", "upbeat breakfast music", "feel good morning playlist"]
        elif 12 <= hour < 17:
            mood = "afternoon"
            suggestions = ["focus music", "work beats", "chill afternoon playlist"]
        elif 17 <= hour < 21:
            mood = "evening"
            suggestions = ["evening vibes", "sunset music", "chill evening playlist"]
        else:
            mood = "night"
            suggestions = ["late night music", "chill night vibes", "relaxing night playlist"]
        
        import random
        random_suggestion = random.choice(suggestions)

        return (
            f"What song would you like to hear, sir? "
            f"Or I can play some {mood} vibes for you, like {random_suggestion}."
        )

    def play_random_music(self) -> str:
        """Play random music - resume if open, otherwise open Spotify."""
        # Try to resume if Spotify is open
        result = self.play_spotify()
        return result

    def open_gmail(self) -> str:
        """Open Gmail."""
        try:
            _open_url("https://mail.google.com")
            return "Opening Gmail, sir."
        except Exception:
            return "Unable to open Gmail, sir."

    def open_maps(self, query: str = "") -> str:
        """Open Google Maps."""
        if query:
            url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
        else:
            url = "https://maps.google.com"
        try:
            _open_url(url)
            return f"Opening Maps{f' for {query}' if query else ''}, sir."
        except Exception:
            return "Unable to open Maps, sir."

    def read_webpage(self, url: str) -> str:
        """Read and summarize a webpage."""
        if not url:
            return "Please provide a URL, sir."
        try:
            if not url.startswith("http"):
                url = f"https://{url}"
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code == 200:
                text = r.text[:2000]
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return f"Here's what I found: {text[:500]}..."
            return f"Unable to read {url}, sir."
        except Exception:
            return f"Error reading {url}, sir."

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "known_sites": len(self.KNOWN_WEBSITES),
        }
