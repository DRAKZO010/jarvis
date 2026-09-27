"""Context MCP - Conversation context and history tracking."""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.context")

CONTEXT_PATH = Path(__file__).resolve().parent.parent / "jarvis_context.json"


class ContextMCP(MCPBase):
    """Conversation context MCP module."""

    name = "context"
    description = "Conversation context and history tracking"
    commands = ["clear history", "conversation history"]

    def __init__(self):
        super().__init__()
        self.last_command = ""
        self.last_response = ""
        self.last_action: Optional[Tuple[str, str]] = None
        self.pending_confirmation: Optional[Dict] = None
        self.conversation_history: List[Dict] = []
        self.max_history = 30
        self._topics: List[str] = []  # Track conversation topics
        self._user_preferences: Dict[str, Any] = {}  # Learn user preferences

    def initialize(self) -> bool:
        """Initialize context module."""
        # Load saved context
        saved = self._load_context()
        if saved:
            self.conversation_history = saved.get("history", [])
            self._topics = saved.get("topics", [])
            self._user_preferences = saved.get("preferences", {})
        self._initialized = True
        log.info("Context MCP initialized with %d history entries.", len(self.conversation_history))
        return True

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle context commands."""
        if command == "clear history":
            self.clear()
            return "Conversation history cleared, sir."
        if command == "conversation history":
            return self.get_full_history_text()
        return None

    def update(self, command: str, response: str, action: Optional[Tuple[str, str]] = None) -> None:
        """Update conversation context."""
        self.last_command = command
        self.last_response = response
        self.last_action = action
        
        entry = {
            "user": command,
            "jarvis": response,
            "timestamp": datetime.now().isoformat(),
        }
        if action:
            entry["action"] = action[0]
        
        self.conversation_history.append(entry)
        
        # Extract topics from command
        self._extract_topics(command)
        
        # Learn user preferences
        self._learn_preferences(command)
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # Save context periodically
        if len(self.conversation_history) % 5 == 0:
            self._save_context()

    def _extract_topics(self, text: str) -> None:
        """Extract conversation topics from text."""
        keywords = ["weather", "music", "news", "task", "reminder", "email", "file", "search", "open"]
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower and keyword not in self._topics:
                self._topics.append(keyword)
                # Keep only last 10 topics
                if len(self._topics) > 10:
                    self._topics.pop(0)

    def _learn_preferences(self, text: str) -> None:
        """Learn user preferences from commands."""
        text_lower = text.lower()
        
        # Detect app preferences
        if "open chrome" in text_lower or "use chrome" in text_lower:
            self._user_preferences["browser"] = "chrome"
        elif "open edge" in text_lower or "use edge" in text_lower:
            self._user_preferences["browser"] = "edge"
        
        # Detect search preferences
        if "google" in text_lower:
            self._user_preferences["search_engine"] = "google"
        elif "bing" in text_lower:
            self._user_preferences["search_engine"] = "bing"

    def set_confirmation(self, action_type: str, payload: str, prompt: str) -> None:
        """Set a pending confirmation."""
        self.pending_confirmation = {"action": action_type, "payload": payload, "prompt": prompt}

    def clear_confirmation(self) -> None:
        """Clear pending confirmation."""
        self.pending_confirmation = None

    def get_context_string(self) -> str:
        """Get conversation context as string for LLM."""
        if not self.conversation_history:
            return "No recent conversation."
        recent = self.conversation_history[-7:]  # Last 7 exchanges
        return "\n".join(
            f"User: {h['user']}\nJARVIS: {h['jarvis']}" for h in recent
        )

    def get_full_history_text(self) -> str:
        """Get full conversation history as readable text."""
        if not self.conversation_history:
            return "No conversation history yet, sir."
        lines = []
        for h in self.conversation_history[-10:]:
            lines.append(f"User: {h['user']}")
            lines.append(f"JARVIS: {h['jarvis']}")
            lines.append("")
        return "\n".join(lines)

    def get_recent_commands(self, count: int = 5) -> List[str]:
        """Get recent commands."""
        return [h["user"] for h in self.conversation_history[-count:]]

    def get_recent_responses(self, count: int = 5) -> List[str]:
        """Get recent responses."""
        return [h["jarvis"] for h in self.conversation_history[-count:]]

    def get_topics(self) -> List[str]:
        """Get recent conversation topics."""
        return self._topics.copy()

    def get_preferences(self) -> Dict[str, Any]:
        """Get learned user preferences."""
        return self._user_preferences.copy()

    def clear(self) -> None:
        """Clear all context."""
        self.last_command = ""
        self.last_response = ""
        self.last_action = None
        self.pending_confirmation = None
        self.conversation_history.clear()
        self._topics.clear()
        self._user_preferences.clear()
        self._save_context()

    def _load_context(self) -> Dict:
        """Load context from file."""
        try:
            with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_context(self) -> None:
        """Save context to file."""
        try:
            data = {
                "history": self.conversation_history,
                "topics": self._topics,
                "preferences": self._user_preferences,
            }
            with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error("Failed to save context: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "history_length": len(self.conversation_history),
            "topics": self._topics,
            "has_pending_confirmation": self.pending_confirmation is not None,
        }
