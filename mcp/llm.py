"""LLM MCP - Large Language Model integration via Google Gemini."""

from __future__ import annotations
import importlib.util
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.llm")

# Gemini's no-billing tier caps requests per model, per project, per day. Each
# model therefore carries its own allowance, and one model using up its day is
# not a reason for the assistant to go quiet while another still has room.
# Verified to work with this key; unlisted models 404 and are skipped at runtime.
DEFAULT_FALLBACK_MODELS = (
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)

# Requests the free tier allows per model per day. Tracked locally so command
# routing cannot spend the day's allowance before the user asks anything.
# Other clients sharing the project draw from the same cap, so this is a guard
# rail rather than a promise, and the server's 429 is still the authority.
FREE_TIER_DAILY_LIMIT = 20

# Requests held back for answering. Routing runs on nearly every utterance and
# is only a convenience, so it gives way before questions do.
DAILY_ANSWER_RESERVE = 3

# A burst limit is measured in seconds. A daily cap is not, which is why the two
# have to be told apart rather than sharing one short penalty.
BURST_PENALTY_S = 20

# The daily cap is reported as a 429 against a per-day quota, and it carries the
# same "retry in Ns" hint as a burst limit. The status code cannot tell them
# apart, so the violated quota is what identifies the cap. The provider has been
# observed naming it three ways, so all of them are matched: the quotaId, the
# human-readable metric, and the quotaMetric. A per-minute free tier cap reports
# as PerMinute and none of these match, which is what should happen.
_DAILY_QUOTA_MARKERS = (
    "perday",
    "per_day",
    "per day per project",
    "free_tier_requests",
)

# This key cannot use the model at all, e.g. retired or not released to the
# account. Retrying later will not help, but another model will.
_UNAVAILABLE_MARKERS = (
    "is not found for api version",
    "no longer available",
    "is not supported for",
)


def _next_local_midnight() -> float:
    """POSIX time of the next local midnight.

    The free tier's cap resets on the calendar day, so the gate on an exhausted
    model has to be timed against the calendar rather than a fixed offset.
    """
    now = datetime.now()
    midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return midnight.timestamp()


def _is_daily_quota_error(err_str: str) -> bool:
    """True when a 429 is the free tier's daily cap rather than a burst limit."""
    low = err_str.lower()
    if "429" not in err_str and "resource_exhausted" not in low:
        return False
    return any(marker in low for marker in _DAILY_QUOTA_MARKERS)


def _is_model_unavailable(err_str: str) -> bool:
    """True when the model cannot be used with this key at all."""
    low = err_str.lower()
    return any(marker in low for marker in _UNAVAILABLE_MARKERS)


def _is_transient_error(err_str: str) -> bool:
    """True when the provider is busy or the connection dropped.

    Gemini reports a busy backend as 503 and the OpenAI client reports a stalled
    stream as a timeout. Neither says anything about the model's quota, so the
    model stays usable and the next one in the chain is worth trying.
    """
    low = err_str.lower()
    if any(code in err_str for code in ("502", "503", "504")):
        return True
    if "error code: 500" in low or "500 internal" in low:
        return True
    return any(marker in low for marker in (
        "overloaded", "timed out", "timeout", "temporarily unavailable",
        "connection error", "connection reset", "server error",
    ))


def _retry_delay(err_str: str, default: float) -> float:
    """Seconds the provider asked us to wait, when it says."""
    match = re.search(r"retryDelay[^0-9]*([\d.]+)", err_str)
    if match:
        try:
            return max(0.0, float(match.group(1)))
        except ValueError:
            pass
    return default


class LLMMCP(MCPBase):
    """LLM integration MCP module."""

    name = "llm"
    description = "Large Language Model integration via Google Gemini"
    commands = [
        "ask",
        "think",
        "analyze",
        "explain",
        "suggest",
    ]

    def __init__(self):
        super().__init__()
        # LLM provider: "gemini" (default) or "ollama" for local models.
        self.provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        self.model = os.getenv("LLM_MODEL", "gemini-flash-latest")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "800"))
        self.cooldown = 1.5
        self._last_call = 0.0
        # Absolute time before which every call is refused (burst rate limit).
        self._penalty_until = 0.0
        # Common state, always initialized regardless of provider.
        self._exhausted_until: Dict[str, float] = {}
        self._unusable: set = set()
        self._daily_spend: Dict[str, int] = {}
        self._spend_day: Optional[str] = None
        self._active_model: Optional[str] = None
        self._last_error: str = ""
        self._last_error_kind: str = ""
        self._context = None
        self._client = None
        self._client_error: Optional[str] = None
        self._soul_cache = None
        self._user_cache = None
        self._conversation_history: List[Dict[str, str]] = []
        self._max_history = 10
        self._budget_path = Path(os.getenv("LLM_BUDGET_PATH", str(Path(__file__).resolve().parent.parent / "jarvis_llm_budget.json")))
        self._load_budget()

        # Provider-specific configuration.
        if self.provider == "ollama":
            # Ollama's OpenAI-compatible API lives at {base_url}/v1.
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            if not ollama_url.endswith("/v1"):
                ollama_url = ollama_url + "/v1"
            self.base_url = ollama_url
            # Ollama does not require an API key; the OpenAI client still
            # wants a non-empty string, so use a placeholder.
            self.api_key = "ollama"
            # Local models have no daily quota or fallback chain.
            self.fallback_models = ()
            self.models: List[str] = [self.model]
            self.daily_limit = 10**9  # effectively unlimited locally
            self.answer_reserve = 0
        else:
            # Gemini's OpenAI-compatible endpoint. v1beta is required: the v1
            # alias returns 404 for the current flash models.
            self.base_url = os.getenv(
                "LLM_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self.api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
            # Models to try in order, the configured one first.
            self.fallback_models = tuple(
                m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()
            )
            self.models: List[str] = list(
                dict.fromkeys([self.model, *(self.fallback_models or DEFAULT_FALLBACK_MODELS)])
            )
            self.daily_limit = int(os.getenv("LLM_DAILY_LIMIT", str(FREE_TIER_DAILY_LIMIT)))
            self.answer_reserve = int(os.getenv("LLM_ANSWER_RESERVE", str(DAILY_ANSWER_RESERVE)))

    def _load_budget(self) -> None:
        try:
            raw = json.loads(self._budget_path.read_text(encoding="utf-8"))
            self._exhausted_until = {k: float(v) for k, v in (raw.get("exhausted_until") or {}).items()}
            self._daily_spend = {k: int(v) for k, v in (raw.get("daily_spend") or {}).items()}
            self._spend_day = raw.get("spend_day")
        except Exception:
            self._exhausted_until = {}
            self._daily_spend = {}
            self._spend_day = None

    def _save_budget(self) -> None:
        try:
            self._budget_path.parent.mkdir(parents=True, exist_ok=True)
            self._budget_path.write_text(json.dumps({
                "spend_day": self._spend_day,
                "daily_spend": self._daily_spend,
                "exhausted_until": self._exhausted_until,
            }), encoding="utf-8")
        except Exception as e:
            log.debug("LLM budget persist failed: %s", e)

    def initialize(self) -> bool:
        """Initialize LLM module."""
        self._initialized = True
        if not os.environ.get("GOOGLE_API_KEY", "").strip():
            log.warning("GOOGLE_API_KEY not set - LLM reasoning disabled.")
        else:
            log.info("LLM MCP initialized with model: %s", self.model)
        return True

    @property
    def available(self) -> bool:
        """True when an API key is configured and the client imports."""
        if self.provider == "ollama":
            return self._import_ok()
        return bool(os.environ.get("GOOGLE_API_KEY", "").strip()) and self._import_ok()

    def _import_ok(self) -> bool:
        try:
            return importlib.util.find_spec("openai") is not None
        except (ImportError, ValueError):
            return False

    def _get_client(self):
        """Get or create the cached OpenAI client, or None if unusable."""
        if self._client is not None:
            return self._client

        if not self._import_ok():
            self._client_error = "the 'openai' package is not installed"
            log.error("LLM unavailable: %s. Run: pip install openai", self._client_error)
            return None

        api_key = self.api_key if self.provider == "ollama" else os.environ.get("GOOGLE_API_KEY", "").strip()
        if self.provider != "ollama" and not api_key:
            self._client_error = "GOOGLE_API_KEY is not set"
            return None

        try:
            from openai import OpenAI
        except ImportError:
            self._client_error = "the 'openai' package is not installed"
            log.error("LLM unavailable: %s. Run: pip install openai", self._client_error)
            return None

        try:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=api_key,
                timeout=60.0,
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

    # --- daily quota tracking ------------------------------------------------

    def _spend_for(self, model: str) -> int:
        """Requests sent to ``model`` today, rolled over on the calendar day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._spend_day != today:
            self._spend_day = today
            self._daily_spend = {}
        return self._daily_spend.get(model, 0)

    def _record_spend(self, model: str) -> None:
        """Record a request against the daily allowance (no-op for Ollama)."""
        if self.provider == "ollama":
            return
        self._spend_for(model)
        self._daily_spend[model] = self._daily_spend.get(model, 0) + 1
        self._save_budget()

    def _spend_today(self) -> int:
        return sum(self._spend_for(model) for model in self.models)

    def _usable_models(self) -> List[str]:
        """Models worth trying, skipping those already known to be spent.

        Quota-exhausted models are skipped until their day rolls over; models
        the key cannot use are skipped outright. The local request count is not
        a reason to skip a model, because the provider is the authority on
        whether anything is left. Local Ollama models have no quota tracking.
        """
        if self.provider == "ollama":
            return list(self.models)
        now = time.time()
        return [
            model
            for model in self.models
            if self._exhausted_until.get(model, 0.0) <= now and model not in self._unusable
        ]

    def _routing_budget_gate(self) -> str:
        """Refuse to spend the last of the day's allowance on command routing.

        Routing fires on nearly every utterance, so without this it quietly
        consumes the free tier and leaves nothing for the question the user
        actually asked. Local Ollama models have no quota, so this is a no-op.
        """
        if self.provider == "ollama":
            return ""
        if not self._usable_models():
            return self._exhausted_message()
        if self._spend_today() < max(0, self.daily_limit - self.answer_reserve):
            return ""
        return (
            "Routing is paused so the last of today's free requests stay "
            "available for answers, sir."
        )

    # --- failure classification ---------------------------------------------

    def _note_failure(self, model: str, err_str: str) -> str:
        """Classify a failure and record what it means for later calls.

        Returns the kind of failure, which tells the caller whether another
        model in the chain is worth trying.
        """
        self._last_error = err_str
        kind = self._classify(err_str)
        self._last_error_kind = kind
        if kind == "daily":
            # Nothing about this model improves until the day rolls over.
            reset = _next_local_midnight()
            self._exhausted_until[model] = reset
            self._save_budget()
            log.warning(
                "%s has used its daily request allowance; next usable at %s",
                model,
                datetime.fromtimestamp(reset).isoformat(timespec="minutes"),
            )
        elif kind == "unavailable":
            self._unusable.add(model)
            log.warning("%s is not available to this key; skipping it", model)
        elif kind == "burst":
            # A burst limit is normally project-wide, so a different model is no
            # way around it. Keep the gate short, and let the provider set it.
            self._penalty_until = time.monotonic() + _retry_delay(err_str, BURST_PENALTY_S)
        return kind

    def _classify(self, err_str: str) -> str:
        """Name the kind of failure, which is what tells callers what to do.

        Checked in this order: the daily cap and unavailable-model cases both
        arrive as 429 or 404 but are specific to one model, so they must be
        recognised before the broader status-code tests.
        """
        if _is_daily_quota_error(err_str):
            return "daily"
        if _is_model_unavailable(err_str):
            return "unavailable"
        if self._rate_limited(err_str):
            return "burst"
        if self._out_of_credit(err_str):
            return "credit"
        if _is_transient_error(err_str):
            return "transient"
        return "other"

    def _failure_message(self, kind: str) -> str:
        if kind == "daily":
            return self._exhausted_message()
        if kind == "unavailable":
            return "None of my configured AI models is available to this key, sir."
        if kind == "burst":
            return "My AI core is rate limited. Please wait a moment and try again, sir."
        if kind == "credit":
            return "My AI core is out of quota. Please check your Gemini API key, sir."
        if kind == "transient":
            return "My AI core is overloaded right now. Please try again in a moment, sir."
        return "My AI core hit an error. Please try again, sir."

    def _exhausted_message(self) -> str:
        """Say what actually happened, and when it ends.

        The provider's own hint says to retry in under a minute. That is true
        and useless, because the cap is per day, and repeating it every twenty
        seconds is what made this look like a permanent fault rather than an
        exhausted allowance.
        """
        pending = [t for t in self._exhausted_until.values() if t > time.time()]
        if not pending:
            return "My AI core has used up today's free requests, sir."
        reset = datetime.fromtimestamp(min(pending))
        if reset.date() == datetime.now().date():
            return "My AI core has used up today's free requests, sir."
        return (
            "My AI core has used up today's free requests. The quota resets at "
            f"{reset.strftime('%H:%M')} tomorrow, sir."
        )

    # --- chat completion -----------------------------------------------------

    def _ask_model(self, client, model: str, messages: list, temperature: float):
        """Single completion attempt, with the request counted against the day."""
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        self._record_spend(model)
        return resp

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

    def _penalty_refusal(self) -> Optional[str]:
        """Spoken refusal while the burst penalty is up, else ``None``."""
        if time.monotonic() < self._penalty_until:
            return "My AI core is rate limited. I'll be ready in a moment, sir."
        return None

    def _check_cooldown(self, skip: bool) -> Optional[str]:
        """Return a spoken refusal if the cooldown or penalty is active."""
        if skip:
            return None
        refusal = self._penalty_refusal()
        if refusal:
            return refusal
        now = time.monotonic()
        if now - self._last_call < self.cooldown:
            remaining = self.cooldown - (now - self._last_call)
            return f"Please wait {remaining:.0f} seconds before the next query, sir."
        return None

    def ask(self, prompt: str, system_prompt: str = "", use_history: bool = True,
            skip_cooldown: bool = False, temperature: float = 0.8,
            purpose: str = "answer") -> str:
        """Ask the LLM a question with conversation history.

        ``temperature`` is lowered by callers that want a deterministic answer,
        such as command routing, where the same utterance must always resolve
        to the same action.

        ``purpose`` is ``"route"`` for command routing and ``"answer"`` for
        everything else. Routing is held back once the day's allowance runs
        low, because it is a convenience and an answer is the point.

        The reply is retried on the next model in the chain when a model is
        spent or unavailable, since Gemini's daily cap is per model.
        """
        refusal = self._check_cooldown(skip_cooldown)
        if refusal:
            return refusal

        if purpose == "route":
            gated = self._routing_budget_gate()
            if gated:
                return gated

        client = self._get_client()
        if not client:
            return self._offline_message()

        models = self._usable_models()
        if not models:
            return self._exhausted_message()

        messages = self._build_messages(prompt, system_prompt, use_history)
        last_kind = "other"
        saw_empty = False

        for model in models:
            try:
                resp = self._ask_model(client, model, messages, temperature)
            except Exception as e:
                err_str = str(e)
                log.error("LLM error on %s: %s", model, e)
                last_kind = self._note_failure(model, err_str)
                if last_kind in ("daily", "unavailable", "transient"):
                    # These are specific to one model or to the provider's
                    # current load, so the next model is worth trying.
                    continue
                return self._failure_message(last_kind)

            self._last_call = time.monotonic()
            self._active_model = model
            content = resp.choices[0].message.content
            if not content:
                # A reasoning model can spend the whole token budget before it
                # emits any text. The next model in the chain may not.
                log.warning("%s returned an empty response", model)
                saw_empty = True
                continue

            self._record_exchange(prompt, content)
            return content.strip()

        if saw_empty:
            return "My AI core returned an empty response. Please try again, sir."
        return self._failure_message(last_kind)

    def ask_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        use_history: bool = True,
        skip_cooldown: bool = False,
        purpose: str = "answer",
        temperature: float = 0.8,
    ):
        """Stream an LLM response token by token for lower first-token latency.

        Mirrors the :meth:`ask` guardrails for the streaming path.
        """
        refusal = self._check_cooldown(skip_cooldown)
        if refusal:
            yield refusal
            return

        if purpose == "route":
            gated = self._routing_budget_gate()
            if gated:
                yield gated
                return

        client = self._get_client()
        if not client:
            yield self._offline_message()
            return

        models = self._usable_models()
        if not models:
            yield self._exhausted_message()
            return

        messages = self._build_messages(prompt, system_prompt, use_history)
        last_kind = "other"

        for model in models:
            # The penalty is provider-wide, so a model fired after it started
            # would hit the same wall the one before it did. Re-checked per
            # model because failures inside this loop move the penalty.
            refusal = self._penalty_refusal()
            if refusal:
                yield refusal
                return
            if purpose == "route":
                gated = self._routing_budget_gate()
                if gated:
                    yield gated
                    return
            parts: List[str] = []
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                self._record_spend(model)
                self._active_model = model
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
                    return
                # Empty stream: same reasoning-token problem as ask(), and the
                # next model may answer.
                log.warning("%s streamed an empty response", model)
                last_kind = "empty"

            except Exception as e:
                err_str = str(e)
                log.error("LLM stream error on %s: %s", model, e)
                last_kind = self._note_failure(model, err_str)
                if last_kind == "daily":
                    # The daily cap is shared per project, not per model, so
                    # every model left in the chain would fail the same way.
                    # Stop instead of hammering the cap model by model.
                    yield self._failure_message(last_kind)
                    return
                if last_kind == "unavailable":
                    continue
                if last_kind == "transient" and not parts:
                    # Nothing of this reply has been spoken yet, so the next
                    # model can answer cleanly instead of giving up.
                    continue
                yield self._failure_message(last_kind)
                return

        yield (
            "My AI core returned an empty response. Please try again, sir."
            if last_kind == "empty"
            else self._failure_message(last_kind)
        )

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
            self._soul_cache = "You are Jarvis, a sophisticated AI assistant."
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
        """Get module status.

        ``available`` accounts for quota, not just configuration. Reporting a
        key and an import as enough is what let the health endpoint claim the AI
        core was fine while every call failed.
        """
        now = time.time()
        status = {
            "name": self.name,
            "initialized": self._initialized,
            "provider": self.provider,
            "model": self.model,
            "active_model": self._active_model,
            "models": list(self.models),
            "available": self.available and bool(self._usable_models()),
            "error": self._client_error,
            "penalized": time.monotonic() < self._penalty_until,
            "exhausted_models": sorted(
                model for model, until in self._exhausted_until.items() if until > now
            ),
            "unusable_models": sorted(self._unusable),
            # The raw error is a long JSON dump. This endpoint is polled, so it
            # carries the classified cause plus enough of the payload to identify
            # the failure without the whole thing.
            "last_error_kind": self._last_error_kind or None,
            "last_error": self._last_error[:200] or None,
            "history_length": len(self._conversation_history),
        }
        if self.provider != "ollama":
            status["daily_requests_used"] = self._spend_today()
            status["daily_limit"] = self.daily_limit
        return status
