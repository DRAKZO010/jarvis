"""LLM MCP - Large Language Model integration via OpenRouter."""

from __future__ import annotations
import logging
import os
import time
from typing import Any, Dict, List, Optional

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.llm")


class LLMMCP(MCPBase):
    """LLM integration MCP module."""

    name = "llm"
    description = "Large Language Model integration via OpenRouter"
    commands = [
        "ask",
        "think",
        "analyze",
        "explain",
        "suggest",
    ]

    def __init__(self):
        super().__init__()
        self.model = os.getenv("LLM_MODEL", "qwen/qwen-2.5-7b-instruct")
        self.cooldown = 1.5
        self._last_call = 0.0
        # Absolute time before which every call is refused (rate limiting).
        self._penalty_until = 0.0
        self._context = None
        self._client = None
        self._client_error: Optional[str] = None
        self._soul_cache = None
        self._user_cache = None
        self._conversation_history: List[Dict[str, str]] = []
        self._max_history = 10

    def initialize(self) -> bool:
        """Initialize LLM module."""
        self._initialized = True
        if not os.environ.get("OPENROUTER_API_KEY", "").strip():
            log.warning("OPENROUTER_API_KEY not set - LLM reasoning disabled.")
        else:
            log.info("LLM MCP initialized with model: %s", self.model)
        return True

    @property
    def available(self) -> bool:
        """True when an API key is configured and the client imports."""
        return bool(os.environ.get("OPENROUTER_API_KEY", "").strip()) and self._import_ok()

    def _import_ok(self) -> bool:
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        """Get or create the cached OpenAI client, or None if unusable."""
        if self._client is not None:
            return self._client

        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            self._client_error = "OPENROUTER_API_KEY is not set"
            return None

        try:
            from openai import OpenAI
        except ImportError:
            self._client_error = "the 'openai' package is not installed"
            log.error("LLM unavailable: %s. Run: pip install openai", self._client_error)
            return None

        try:
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                timeout=20.0,
            )
            self._client_error = None
        except Exception as e:
            self._client_error = str(e)
            log.error("LLM client init failed: %s", e)
            self._client = None
        return self._client

    def _offline_message(self) -> str:
        return (
            "My AI core is offline, sir. "
            f"Reason: {self._client_error or 'not configured'}. "
            "Free-form questions are unavailable until that is fixed."
        )

    def _rate_limited(self, err_str: str) -> bool:
        return "429" in err_str or "rate limit" in err_str.lower()

    def _out_of_credit(self, err_str: str) -> bool:
        return (
            "402" in err_str
            or "credit" in err_str.lower()
            or "payment" in err_str.lower()
            or "insufficient" in err_str.lower()
        )

    def set_context(self, context) -> None:
        """Set the conversation context."""
        self._context = context

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle LLM commands."""
        if command in ("ask", "think", "analyze", "explain", "suggest"):
            if args:
                return self.ask(args)
            return "What would you like me to think about, sir?"
        return None

    def _build_messages(self, prompt: str, system_prompt: str, use_history: bool) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if use_history and self._conversation_history:
            messages.extend(self._conversation_history[-self._max_history:])
        messages.append({"role": "user", "content": prompt})
        return messages

    def _record_exchange(self, prompt: str, content: str) -> None:
        self._conversation_history.append({"role": "user", "content": prompt})
        self._conversation_history.append({"role": "assistant", "content": content.strip()})
        if len(self._conversation_history) > self._max_history * 2:
            self._conversation_history = self._conversation_history[-self._max_history * 2:]

    def _check_cooldown(self, skip: bool) -> Optional[str]:
        """Return a spoken refusal if the cooldown or penalty is active."""
        if skip:
            return None
        now = time.monotonic()
        if now < self._penalty_until:
            return "My AI core is rate limited. I'll be ready in a moment, sir."
        if now - self._last_call < self.cooldown:
            remaining = self.cooldown - (now - self._last_call)
            return f"Please wait {remaining:.0f} seconds before the next query, sir."
        return None

    def ask(self, prompt: str, system_prompt: str = "", use_history: bool = True, skip_cooldown: bool = False) -> str:
        """Ask the LLM a question with conversation history."""
        refusal = self._check_cooldown(skip_cooldown)
        if refusal:
            return refusal

        client = self._get_client()
        if not client:
            return self._offline_message()

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(prompt, system_prompt, use_history),
                temperature=0.8,
                max_tokens=200,
            )

            self._last_call = time.monotonic()
            content = resp.choices[0].message.content
            if not content:
                return "My AI core returned an empty response. Please try again, sir."

            self._record_exchange(prompt, content)
            return content.strip()

        except Exception as e:
            err_str = str(e)
            log.error("LLM error: %s", e)
            if self._rate_limited(err_str):
                # Push the gate forward instead of shortening it.
                self._penalty_until = time.monotonic() + 20
                return "My AI core is rate limited. Please wait a moment and try again, sir."
            if self._out_of_credit(err_str):
                return "My AI core needs credits. Please add funds at openrouter.ai, sir."
            return "I encountered an error with my AI core."

    def ask_stream(self, prompt: str, system_prompt: str = "", use_history: bool = True):
        """Stream an LLM response token by token for lower first-token latency."""
        refusal = self._check_cooldown(False)
        if refusal:
            yield refusal
            return

        client = self._get_client()
        if not client:
            yield self._offline_message()
            return

        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(prompt, system_prompt, use_history),
                temperature=0.8,
                max_tokens=200,
                stream=True,
            )

            parts: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    parts.append(token)
                    yield token

            self._last_call = time.monotonic()
            content = "".join(parts).strip()
            if content:
                self._record_exchange(prompt, content)

        except Exception as e:
            err_str = str(e)
            log.error("LLM stream error: %s", e)
            if self._rate_limited(err_str):
                self._penalty_until = time.monotonic() + 20
                yield "My AI core is rate limited. Please wait a moment, sir."
            elif self._out_of_credit(err_str):
                yield "My AI core needs credits. Please add funds at openrouter.ai, sir."
            else:
                yield "I encountered an error with my AI core."

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()
        log.info("Conversation history cleared.")

    def build_system_prompt(self, identity: Dict, facts_str: str, tasks_str: str, context_str: str) -> str:
        """Build the system prompt for the LLM."""
        soul = self._load_soul()
        user_profile = self._load_user_profile()

        return (
            f"{soul}\n\n"
            f"USER PROFILE:\n{user_profile}\n\n"
            f"CAPABILITIES:\n"
            f"- Open applications, websites, and search the web\n"
            f"- Open YouTube, Gmail, Maps, and social media\n"
            f"- Read and summarize web content\n"
            f"- Manage tasks (add, complete, delete) and reminders\n"
            f"- Remember and forget facts the user tells you\n"
            f"- Execute system commands\n"
            f"- Provide system status, weather, and more\n"
            f"- Set alarms and reminders\n"
            f"- Control system volume and brightness\n"
            f"- Search for files and manage them\n\n"
            f"Stored memories:\n{facts_str}\n\n"
            f"Pending tasks:\n{tasks_str}\n\n"
            f"Recent conversation:\n{context_str}\n\n"
            f"Always address the user as 'sir' naturally. Never break character.\n"
            f"Keep responses concise (1-3 sentences). Be helpful and proactive."
        )

    def _load_soul(self) -> str:
        """Load SOUL.md personality file (cached)."""
        if self._soul_cache is not None:
            return self._soul_cache
        from pathlib import Path
        soul_path = Path(__file__).resolve().parent.parent / "SOUL.md"
        try:
            with open(soul_path, "r", encoding="utf-8") as f:
                self._soul_cache = f.read()
        except Exception:
            self._soul_cache = "You are J.A.R.V.I.S., a sophisticated AI assistant."
        return self._soul_cache

    def _load_user_profile(self) -> str:
        """Load USER.md profile file (cached)."""
        if self._user_cache is not None:
            return self._user_cache
        from pathlib import Path
        user_path = Path(__file__).resolve().parent.parent / "USER.md"
        try:
            with open(user_path, "r", encoding="utf-8") as f:
                self._user_cache = f.read()
        except Exception:
            self._user_cache = "User profile not available."
        return self._user_cache

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "model": self.model,
            "available": self.available,
            "error": self._client_error,
            "penalized": time.monotonic() < self._penalty_until,
            "history_length": len(self._conversation_history),
        }
