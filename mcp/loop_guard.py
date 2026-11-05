"""Agent Loop Guard & Context Compressor — adapted from OpenJarvis.

Prevents runaway / degenerate tool loops (repeated calls, A-B-A-B ping-ponging)
and handles 4-stage context compression to protect LLM context windows.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

log = logging.getLogger("jarvis.mcp.loop_guard")


@dataclass(slots=True)
class LoopGuardConfig:
    """Configuration for loop detection and context bounds."""

    enabled: bool = True
    max_identical_calls: int = 3  # SHA-256 of (tool_name, arguments)
    ping_pong_window: int = 6  # Detect A-B-A-B alternating calls
    poll_tool_budget: int = 8  # Max calls allowed for polling / monitoring tools
    max_context_messages: int = 24  # Trigger compression above this length
    warn_before_block: bool = True  # Warn once before terminating loop


@dataclass(slots=True)
class LoopVerdict:
    """Verdict returned by loop guard check."""

    blocked: bool = False
    reason: str = ""
    warned: bool = False


class LoopGuard:
    """Detect and stop degenerate loops and context overflow."""

    def __init__(
        self,
        config: Optional[LoopGuardConfig] = None,
        on_trigger: Optional[Callable[[str, str], None]] = None,
    ):
        self._config = config or LoopGuardConfig()
        self._on_trigger = on_trigger
        self._call_counts: Dict[str, int] = {}
        self._tool_sequence: deque[str] = deque(maxlen=self._config.ping_pong_window * 2)
        self._per_tool_counts: Dict[str, int] = {}
        self._warned_cycles: set[str] = set()

    def check_call(self, tool_name: str, arguments: str = "") -> LoopVerdict:
        """Check whether a tool call should proceed, warn, or be blocked."""
        if not self._config.enabled:
            return LoopVerdict()

        # 1. Identical call hash tracking (SHA-256)
        call_hash = hashlib.sha256(f"{tool_name}:{arguments}".encode("utf-8")).hexdigest()[:16]
        self._call_counts[call_hash] = self._call_counts.get(call_hash, 0) + 1
        if self._call_counts[call_hash] > self._config.max_identical_calls:
            self._notify("identical_call", tool_name)
            verdict = LoopVerdict(
                blocked=True,
                reason=(
                    f"Identical call to '{tool_name}' repeated "
                    f"{self._call_counts[call_hash]} times (max {self._config.max_identical_calls})."
                ),
            )
            return self._apply_warn_policy(verdict, cycle_key=f"identical:{tool_name}")

        # 2. Per-tool budget
        self._per_tool_counts[tool_name] = self._per_tool_counts.get(tool_name, 0) + 1
        if self._per_tool_counts[tool_name] > self._config.poll_tool_budget:
            self._notify("poll_budget", tool_name)
            verdict = LoopVerdict(
                blocked=True,
                reason=(
                    f"Tool '{tool_name}' exceeded max call budget "
                    f"({self._config.poll_tool_budget})."
                ),
            )
            return self._apply_warn_policy(verdict, cycle_key=f"poll:{tool_name}")

        # 3. Ping-pong cycle detection (A-B-A-B or A-B-C-A-B-C)
        self._tool_sequence.append(tool_name)
        if len(self._tool_sequence) >= self._config.ping_pong_window:
            if self._detect_ping_pong():
                self._notify("ping_pong", tool_name)
                verdict = LoopVerdict(
                    blocked=True,
                    reason="Repetitive tool-calling pattern detected (ping-pong loop).",
                )
                return self._apply_warn_policy(verdict, cycle_key=f"ping_pong:{tool_name}")

        return LoopVerdict()

    def _apply_warn_policy(self, verdict: LoopVerdict, cycle_key: str = "") -> LoopVerdict:
        if verdict.blocked and self._config.warn_before_block:
            key = cycle_key or verdict.reason
            if key not in self._warned_cycles:
                self._warned_cycles.add(key)
                log.warning("LoopGuard warning: %s", verdict.reason)
                return LoopVerdict(blocked=False, warned=True, reason=verdict.reason)
        return verdict

    def _detect_ping_pong(self) -> bool:
        seq = list(self._tool_sequence)
        n = len(seq)
        for period in (2, 3):
            if n >= period * 2:
                tail = seq[-period * 2 :]
                pattern = tail[:period]
                if all(tail[i] == pattern[i % period] for i in range(len(tail))):
                    return True
        return False

    def _notify(self, reason_type: str, tool_name: str) -> None:
        log.warning("LoopGuard triggered (%s) on tool '%s'", reason_type, tool_name)
        if self._on_trigger:
            try:
                self._on_trigger(reason_type, tool_name)
            except Exception as e:
                log.debug("LoopGuard trigger callback error: %s", e)

    def reset(self) -> None:
        """Reset call counters and pattern sequences."""
        self._call_counts.clear()
        self._tool_sequence.clear()
        self._per_tool_counts.clear()
        self._warned_cycles.clear()

    # --- 4-stage Context Overflow Compression --------------------------------

    @staticmethod
    def _get_role(msg: Any) -> str:
        if isinstance(msg, dict):
            return str(msg.get("role", "")).lower()
        return str(getattr(msg, "role", "")).lower()

    @staticmethod
    def _is_system(msg: Any) -> bool:
        return LoopGuard._get_role(msg) == "system"

    @staticmethod
    def _is_tool(msg: Any) -> bool:
        return LoopGuard._get_role(msg) in ("tool", "function")

    def compress_context(self, messages: Sequence[Any]) -> List[Any]:
        """Apply 4-stage context overflow recovery.

        Stage 1: Truncate old tool output messages (keep most recent).
        Stage 2: Sliding window keeping system prompt + recent exchanges.
        Stage 3: Drop middle exchanges.
        Stage 4: Emergency truncation (system prompt + last 2 exchanges).
        """
        if not messages or len(messages) <= self._config.max_context_messages:
            return list(messages)

        msgs = list(messages)

        # Stage 1: Truncate older tool results
        threshold = len(msgs) // 2
        stage1: List[Any] = []
        for i, msg in enumerate(msgs):
            if i < threshold and self._is_tool(msg):
                if isinstance(msg, dict):
                    truncated_msg = dict(msg)
                    truncated_msg["content"] = "[Tool result truncated]"
                    stage1.append(truncated_msg)
                else:
                    try:
                        setattr(msg, "content", "[Tool result truncated]")
                        stage1.append(msg)
                    except Exception:
                        stage1.append(msg)
            else:
                stage1.append(msg)

        if len(stage1) <= self._config.max_context_messages:
            return stage1

        # Stage 2: Sliding window keeping system messages + recent non-system messages
        system_msgs = [m for m in stage1 if self._is_system(m)]
        non_system = [m for m in stage1 if not self._is_system(m)]
        budget = max(1, self._config.max_context_messages - len(system_msgs))
        if len(non_system) > budget:
            non_system = non_system[-budget:]
        stage2 = system_msgs + non_system

        if len(stage2) <= self._config.max_context_messages:
            return stage2

        # Stage 3: Drop middle exchanges
        keep_start = max(len(system_msgs), len(stage2) // 10)
        keep_end = max(2, len(stage2) // 2)
        stage3 = stage2[:keep_start] + stage2[-keep_end:]

        if len(stage3) <= self._config.max_context_messages:
            return stage3

        # Stage 4: Extreme emergency truncation — system + last 4 items (2 turns)
        tail = [m for m in stage2 if not self._is_system(m)]
        return system_msgs + tail[-4:]
