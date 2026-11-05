"""Automatic Long-Term Memory Fact Extractor — adapted from OpenJarvis.

Extracts durable, long-term facts about the user from conversation turns
in the background, without requiring manual 'remember that...' commands.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, List, Optional

log = logging.getLogger("jarvis.mcp.memory_extractor")

DEFAULT_EXTRACTOR_SYSTEM_PROMPT = (
    "You extract durable, long-term facts about the user from a single "
    "conversation exchange. A good fact is stable over time and useful in "
    "future conversations: preferences, identity, goals, ongoing projects, "
    "constraints, or relationships. Ignore one-off task details, small talk, "
    "and anything the assistant said about itself.\n\n"
    "Respond with ONLY a JSON array of short fact strings (each under 200 "
    "characters). If there is nothing worth remembering, respond with []."
)


class FactExtractor:
    """Asynchronously distills long-term facts from dialogue turns."""

    def __init__(
        self,
        llm_caller: Optional[Callable[[str, str], str]] = None,
        memory_store: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm_caller = llm_caller
        self.memory_store = memory_store
        self.system_prompt = system_prompt or DEFAULT_EXTRACTOR_SYSTEM_PROMPT
        self._lock = threading.Lock()

    def extract_from_turn(self, user_text: str, assistant_text: str = "") -> List[str]:
        """Extract durable facts from an exchange synchronously."""
        user_text = (user_text or "").strip()
        if not user_text or not self.llm_caller:
            return []

        # Skip trivial single-word utterances or standard commands
        if len(user_text.split()) < 3 and not any(k in user_text.lower() for k in ("i like", "my name", "i am", "i live", "i work")):
            return []

        prompt = (
            f"{self.system_prompt}\n\n"
            f"User: {user_text}\n"
            f"Assistant: {(assistant_text or '').strip()}\n\n"
            "Facts JSON:"
        )

        try:
            raw = self.llm_caller(prompt, "You are a fact distillation engine. Respond with a JSON array.")
            if not raw:
                return []
            # Extract json array
            raw = raw.strip()
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                facts = json.loads(raw[start : end + 1])
                if isinstance(facts, list):
                    clean_facts = [str(f).strip() for f in facts if str(f).strip()]
                    return clean_facts[:5]
        except Exception as e:
            log.debug("Fact extraction encountered non-fatal error: %s", e)

        return []

    def process_turn_async(self, user_text: str, assistant_text: str = "") -> None:
        """Run fact extraction in a detached background thread."""
        thread = threading.Thread(
            target=self._run_async,
            args=(user_text, assistant_text),
            daemon=True,
            name="jarvis-fact-extractor",
        )
        thread.start()

    def _run_async(self, user_text: str, assistant_text: str) -> None:
        try:
            facts = self.extract_from_turn(user_text, assistant_text)
            if facts and self.memory_store:
                with self._lock:
                    for fact in facts:
                        # Check if memory_store has remember method
                        if hasattr(self.memory_store, "remember"):
                            self.memory_store.remember(f"{fact} [auto-learned]")
                            log.info("Auto-learned user fact: %s", fact)
        except Exception as e:
            log.debug("Async fact extraction background thread error: %s", e)

