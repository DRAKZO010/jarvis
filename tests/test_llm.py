"""LLM resilience tests.

Every test here is offline: a fake client stands in for the OpenAI SDK so no
network call, API key, or credit is required.
"""

from __future__ import annotations

import time
import types

import pytest

from mcp.llm import LLMMCP


class _FakeStream:
    """Yields chunk objects shaped like the OpenAI streaming API."""

    def __init__(self, text):
        # Split into word tokens but keep the separators, so the chunks
        # reassemble to exactly the original text.
        import re as _re
        parts = [p for p in _re.split(r"(\s+)", text) if p]
        self._chunks = [
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=part))]
            )
            for part in parts
        ]

    def __iter__(self):
        return iter(self._chunks)


class _FakeCompletions:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        if kwargs.get("stream"):
            return _FakeStream(self.behaviour)
        message = types.SimpleNamespace(content=self.behaviour)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    m = LLMMCP()
    m.initialize()
    m._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_FakeCompletions("PONG"))
    )
    m._last_call = 0.0
    m._penalty_until = 0.0
    return m


def _swap_behaviour(llm, behaviour):
    llm._client.chat.completions = _FakeCompletions(behaviour)


# --- happy path ------------------------------------------------------------

def test_ask_returns_content(llm):
    assert llm.ask("ping") == "PONG"


def test_ask_records_history(llm):
    llm.ask("first")
    llm._last_call = 0.0
    llm.ask("second")
    assert [m["content"] for m in llm._conversation_history] == [
        "first", "PONG", "second", "PONG",
    ]


def test_history_is_bounded(llm):
    """Each exchange is two messages, so the cap is _max_history * 2."""
    llm._max_history = 4
    for i in range(10):
        llm._last_call = 0.0
        llm.ask(f"q{i}")
    assert len(llm._conversation_history) <= llm._max_history * 2
    # Trimming must not split an exchange, or the API rejects the prompt.
    roles = [m["role"] for m in llm._conversation_history]
    assert roles[0::2] == ["user"] * (len(roles) // 2)
    assert roles[1::2] == ["assistant"] * (len(roles) // 2)


def test_clear_history(llm):
    llm.ask("ping")
    llm.clear_history()
    assert llm._conversation_history == []


def test_use_history_false_omits_previous_turns(llm):
    llm.ask("first")
    llm._last_call = 0.0
    llm.ask("second", use_history=False)
    sent = llm._client.chat.completions.calls[-1]["messages"]
    assert [m["content"] for m in sent] == ["second"]


def test_empty_response_is_reported(llm):
    _swap_behaviour(llm, "")
    assert "empty" in llm.ask("ping").lower()


# --- configuration failures ------------------------------------------------

def test_missing_key_yields_offline_message_not_crash(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    m = LLMMCP()
    m.initialize()
    msg = m.ask("ping")
    assert "offline" in msg.lower()
    assert "OPENROUTER_API_KEY" in msg


def test_missing_openai_package_is_handled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    m = LLMMCP()
    m.initialize()
    m._client = None
    # Force the ImportError branch of _get_client.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert not m.available, "available must be False without the openai package"
    assert "openai" in m.ask("ping").lower()


def test_client_error_is_cached_and_reported(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    m = LLMMCP()
    m.initialize()
    assert m._get_client() is None
    assert m._get_client() is None  # cached error path, must not raise
    assert m._client_error


# --- error classification --------------------------------------------------

def test_429_pushes_penalty_forward(llm):
    _swap_behaviour(llm, Exception("Error code: 429 - rate limit exceeded"))
    before = time.monotonic()
    msg = llm.ask("ping")
    assert llm._penalty_until >= before + 19, "penalty must move into the future"
    assert "rate limited" in msg.lower()


def test_rate_limited_call_is_then_refused_without_calling_api(llm):
    _swap_behaviour(llm, Exception("429"))
    llm.ask("ping")
    calls = len(llm._client.chat.completions.calls)
    refusal = llm.ask("ping")
    assert len(llm._client.chat.completions.calls) == calls, "penalised call hit the API"
    assert "rate limited" in refusal.lower()


def test_credit_error_reports_credits(llm):
    _swap_behaviour(llm, Exception("Error code: 402 - insufficient credits"))
    assert "credit" in llm.ask("ping").lower()


@pytest.mark.parametrize("err", [
    "Error code: 402",
    "insufficient funds",
    "add payment method",
    "out of credit",
])
def test_credit_variants(llm, err):
    _swap_behaviour(llm, Exception(err))
    assert "credit" in llm.ask("ping").lower()


def test_unknown_error_is_generic(llm):
    _swap_behaviour(llm, Exception("kaboom"))
    msg = llm.ask("ping")
    assert "error" in msg.lower()
    assert "kaboom" not in msg, "internal detail leaked to the user"


# --- cooldown --------------------------------------------------------------

def test_cooldown_blocks_immediate_second_call(llm):
    llm.ask("ping")
    msg = llm.ask("ping")
    assert "wait" in msg.lower()


def test_skip_cooldown_bypasses_the_gate(llm):
    llm.ask("ping")
    assert llm.ask("ping", skip_cooldown=True) == "PONG"


def test_failed_call_does_not_start_the_cooldown(llm):
    """A penalty is not a cooldown: do not block retries after a hard error."""
    _swap_behaviour(llm, Exception("kaboom"))
    llm.ask("ping")
    assert llm._last_call == 0.0


def test_penalty_expires(llm):
    llm._penalty_until = time.monotonic() - 1
    assert llm.ask("ping") == "PONG"


# --- streaming and status --------------------------------------------------

def test_ask_stream_yields_content(llm):
    _swap_behaviour(llm, "streamed answer")
    out = list(llm.ask_stream("ping"))
    assert "".join(out).strip() == "streamed answer"


def test_ask_stream_yields_offline_message(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    m = LLMMCP()
    m.initialize()
    assert "offline" in "".join(m.ask_stream("ping")).lower()


def test_status_shape(llm):
    llm.ask("ping")
    status = llm.get_status()
    assert isinstance(status, dict)
    assert "available" in status


def test_max_tokens_is_bounded(llm):
    """A runaway reply must not flood the HUD transcript."""
    llm.ask("ping")
    assert llm._client.chat.completions.calls[-1]["max_tokens"] <= 512
