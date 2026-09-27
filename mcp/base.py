"""Base MCP class shared by all capability modules."""

from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("jarvis.mcp")


class MCPBase:
    """Base class for all MCP feature modules."""

    name: str = "base"
    description: str = "Base MCP module"
    commands: List[str] = []

    def __init__(self):
        self._initialized = False
        self._handlers: Dict[str, Callable] = {}

    def initialize(self) -> bool:
        """Initialize the MCP module. Override in subclasses."""
        self._initialized = True
        log.info("MCP '%s' initialized.", self.name)
        return True

    def shutdown(self) -> None:
        """Shutdown the MCP module. Override in subclasses."""
        self._initialized = False
        log.info("MCP '%s' shut down.", self.name)

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle a command. Override in subclasses."""
        return None

    def get_status(self) -> Dict[str, Any]:
        """Return module status. Override in subclasses."""
        return {"name": self.name, "initialized": self._initialized}

    def get_capabilities(self) -> List[str]:
        """Return list of capabilities. Override in subclasses."""
        return self.commands.copy()

    def register_handler(self, command: str, handler: Callable) -> None:
        """Register a command handler."""
        self._handlers[command] = handler

    def _match_command(self, text: str) -> Tuple[Optional[str], str]:
        """Match text against registered commands. Returns (command, args)."""
        text_lower = text.lower().strip()
        for cmd in self.commands:
            if text_lower.startswith(cmd):
                args = text_lower[len(cmd):].strip()
                return cmd, args
        return None, text
