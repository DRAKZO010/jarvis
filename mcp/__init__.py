"""JARVIS capability modules (MCP).

Each module owns one capability area and is used directly by the app in
``jarvis.py``. There is no central registry: the app wires the modules it
needs and dispatches to them, so module lifetime is explicit.
"""

from .base import MCPBase
from .voice import VoiceMCP
from .tasks import TasksMCP
from .memory import MemoryMCP
from .system import SystemMCP
from .browser import BrowserMCP
from .weather import WeatherMCP
from .llm import LLMMCP
from .automation import AutomationMCP
from .context import ContextMCP
from .reminders import RemindersMCP
from .files import FilesMCP
from .news import NewsMCP

__all__ = [
    "MCPBase",
    "VoiceMCP",
    "TasksMCP",
    "MemoryMCP",
    "SystemMCP",
    "BrowserMCP",
    "WeatherMCP",
    "LLMMCP",
    "AutomationMCP",
    "ContextMCP",
    "RemindersMCP",
    "FilesMCP",
    "NewsMCP",
]
