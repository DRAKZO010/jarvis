"""Memory MCP - Remember, recall, and forget facts."""

from __future__ import annotations
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.memory")

MEMORY_PATH = Path(__file__).resolve().parent.parent / "jarvis_memory.json"


class MemoryMCP(MCPBase):
    """Memory management MCP module."""

    name = "memory"
    description = "Remember, recall, and forget facts"
    commands = [
        "remember",
        "save",
        "store",
        "recall",
        "what do you remember",
        "what do you know",
        "memories",
        "forget",
        "remove",
        "delete",
    ]

    def __init__(self):
        super().__init__()
        self._memory: Dict = {"secure_identity": {}, "known_facts": [], "pending_tasks": []}
        self._responses = {
            "remembered": [
                "Understood. I'll remember that.",
                "Noted, sir.",
                "I'll keep that in mind.",
                "Consider it committed to memory.",
            ],
            "forgotten": [
                "Consider it forgotten, sir.",
                "Memory erased.",
                "I've purged that from my records.",
            ],
        }

    def initialize(self) -> bool:
        """Initialize memory module."""
        self._memory = self._load_memory()
        self._initialized = True
        log.info("Memory MCP initialized with %d facts.", len(self._memory.get("known_facts", [])))
        return True

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle memory commands."""
        if command in ("remember", "save", "store"):
            if args:
                return self.remember(args)
            return "What should I remember, sir?"

        if command in ("recall", "what do you remember", "what do you know", "memories"):
            return self.recall(args)

        if command in ("forget", "remove", "delete"):
            if args:
                return self.forget(args)
            return "What should I forget, sir?"

        return None

    def remember(self, fact: str) -> str:
        """Remember a fact."""
        if "known_facts" not in self._memory:
            self._memory["known_facts"] = []

        # Check for duplicates
        fact_lower = fact.lower().strip()
        for existing in self._memory["known_facts"]:
            if existing.get("fact", "").lower().strip() == fact_lower:
                return "I already know that, sir."

        entry = {"fact": fact, "timestamp": datetime.now().isoformat()}
        self._memory["known_facts"].append(entry)
        self._save_memory()
        return random.choice(self._responses["remembered"])

    def recall(self, query: str = "") -> str:
        """Recall facts matching a query with improved search."""
        facts = self._memory.get("known_facts", [])
        if not facts:
            return "I don't have any stored memories yet, sir."

        if query:
            query_lower = query.lower()
            query_words = query_lower.split()
            
            # Score each fact by relevance
            scored_matches = []
            for f in facts:
                fact_lower = f["fact"].lower()
                score = 0
                
                # Exact match (highest score)
                if query_lower in fact_lower:
                    score = 100
                else:
                    # Word-level matching
                    for word in query_words:
                        if word in fact_lower:
                            score += 10
                        # Fuzzy matching for common typos
                        if len(word) > 3:
                            for fact_word in fact_lower.split():
                                if word[:3] == fact_word[:3]:
                                    score += 2
                
                if score > 0:
                    scored_matches.append((score, f["fact"]))
            
            if scored_matches:
                # Sort by score, return top matches
                scored_matches.sort(key=lambda x: x[0], reverse=True)
                matches = [m[1] for m in scored_matches[:5]]
                return "Here's what I recall: " + "; ".join(matches)
            
            return f"I don't have any memories about {query}, sir."

        recent = [f["fact"] for f in facts[-10:]]
        return "My recent memories: " + "; ".join(recent)

    def forget(self, query: str) -> str:
        """Forget facts matching a query."""
        facts = self._memory.get("known_facts", [])
        before = len(facts)
        self._memory["known_facts"] = [
            f for f in facts if query.lower() not in f["fact"].lower()
        ]
        removed = before - len(self._memory["known_facts"])
        self._save_memory()

        if removed:
            return random.choice(self._responses["forgotten"])
        return f"I couldn't find any memories about {query}, sir."

    def get_identity(self) -> Dict[str, str]:
        """Get the user's identity."""
        return self._memory.get("secure_identity", {})

    def set_identity(self, **kwargs) -> None:
        """Set identity fields."""
        if "secure_identity" not in self._memory:
            self._memory["secure_identity"] = {}
        self._memory["secure_identity"].update(kwargs)
        self._save_memory()

    def get_facts_for_prompt(self) -> str:
        """Get facts formatted for LLM prompt."""
        facts = self._memory.get("known_facts", [])
        if not facts:
            return "  (none)"
        return "\n".join(f"  - {f['fact']}" for f in facts[-15:])

    def get_facts_count(self) -> int:
        """Get count of stored facts."""
        return len(self._memory.get("known_facts", []))

    def _load_memory(self) -> Dict:
        """Load memory from file."""
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {"secure_identity": {}, "known_facts": [], "pending_tasks": []}

    def _save_memory(self) -> None:
        """Save memory to file."""
        try:
            with open(MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(self._memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error("Failed to save memory: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "facts_count": self.get_facts_count(),
            "identity": self.get_identity(),
        }
