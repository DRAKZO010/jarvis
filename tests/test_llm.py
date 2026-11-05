"""LLM resilience tests.

Every test here is offline: a fake client stands in for the OpenAI SDK so no
network call, API key, or credit is required.
"""

from __future__ import annotations

import tempfile
import time
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mcp.llm import LLMMCP

# The provider reports a spent daily allowance as a 429 whose violated quota is
# a per-day one. Copied from a real response so the classification is tested
# against the actual shape of the error rather than a paraphrase.
DAILY_429 = (
    "Error code: 429 - {'error': {'code': 429, 'message': \"Quota exceeded for quota "
    "metric: 'Generate requests per day per project per model (free tier)'\", "
    "'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}]}]}}"
)

UNAVAILABLE_404 = (
    "Error code: 404 - models/primary-model is not found for API version v1main, "
    "or is not supported for this API version"
)

# The same daily cap reported without a quotaId, naming only the quota metric.
# If classification keyed on one field name, this variant would be mistaken for
# a burst limit and retried forever.
DAILY_429_BY_METRIC = (
    "Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your "
    "current quota, please check your plan and billing details.', 'status': "
    "'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': "
    "'generativelanguage.googleapis.com/generate_content_free_tier_requests', "
    "'quotaValue': '20'}]}]}}]"
)

# A per-minute free tier cap must stay a burst, or a 20-second gate replaces a
# day-long one and the core hammers the endpoint all evening.
BURST_429_PER_MINUTE = (
    "Error code: 429 - [{'error': {'code': 429, 'message': 'Resource has been "
    "exhausted', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '15'}]}]}}]"
)


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
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    m = LLMMCP()
    m.initialize()
    m._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_FakeCompletions("PONG"))
    )
    m._last_call = 0.0
    m._penalty_until = 0.0
    m._budget_path = Path(tempfile.gettempdir()) / "jarvis_llm_budget_test.json"
    m._exhausted_until = {}
    m._daily_spend = {}
    m._spend_day = None
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
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    m = LLMMCP()
    m.initialize()
    msg = m.ask("ping")
    assert "offline" in msg.lower()
    assert "GOOGLE_API_KEY" in msg


def test_missing_openai_package_is_handled(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    m = LLMMCP()
    m.initialize()
    m._client = None
    # _import_ok probes with importlib rather than importing, so the absence of
    # the package is simulated by hiding the spec.
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def patched(name, *a, **kw):
        if name == "openai":
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", patched)
    assert not m.available, "available must be False without the openai package"
    assert "openai" in m.ask("ping").lower()


def test_client_error_is_cached_and_reported(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
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


def test_credit_error_reports_quota(llm):
    _swap_behaviour(llm, Exception("Error code: 402 - insufficient credits"))
    assert "quota" in llm.ask("ping").lower()


@pytest.mark.parametrize("err", [
    "Error code: 402",
    "insufficient funds",
    "add payment method",
    "out of credit",
])
def test_credit_variants(llm, err):
    _swap_behaviour(llm, Exception(err))
    assert "quota" in llm.ask("ping").lower()


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
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    m = LLMMCP()
    m.initialize()
    assert "offline" in "".join(m.ask_stream("ping")).lower()


def test_status_shape(llm):
    llm.ask("ping")
    status = llm.get_status()
    assert isinstance(status, dict)
    assert "available" in status


def test_max_tokens_is_bounded(llm):
    """A runaway reply must not flood the HUD transcript.

    The ceiling has to clear Gemini's thinking tokens, which are drawn from the
    same budget, but it still has to be a bound rather than "unlimited".
    """
    llm.ask("ping")
    sent = llm._client.chat.completions.calls[-1]["max_tokens"]
    assert sent == llm.max_tokens
    assert 0 < sent <= 1024


# --- per-model quota and fallback -------------------------------------------


class _PerModelCompletions:
    """Dispatches per model and records which models were actually tried."""

    def __init__(self, behaviours):
        self.behaviours = behaviours
        self.tried = []

    def create(self, **kwargs):
        model = kwargs.get("model")
        self.tried.append(model)
        if model not in self.behaviours:
            raise AssertionError("unexpected model tried: %r" % model)
        behaviour = self.behaviours[model]
        if isinstance(behaviour, Exception):
            raise behaviour
        if kwargs.get("stream"):
            return _FakeStream(behaviour)
        message = types.SimpleNamespace(content=behaviour)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


@pytest.fixture
def chained_llm(monkeypatch):
    """A model chain with pinned names, so tests do not depend on the .env."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LLM_MODEL", "primary-model")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "second-model,third-model")
    m = LLMMCP()
    m.initialize()
    m._last_call = 0.0
    m._penalty_until = 0.0
    m._budget_path = Path(tempfile.gettempdir()) / "jarvis_llm_budget_test.json"
    m._exhausted_until = {}
    m._daily_spend = {}
    m._spend_day = None
    return m


def _swap_models(llm, behaviours):
    fake = _PerModelCompletions(behaviours)
    llm._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=fake))
    return fake


def test_daily_quota_falls_back_to_the_next_model(chained_llm):
    """A model's day being over is not a reason for the assistant to go quiet."""
    fake = _swap_models(chained_llm, {
        "primary-model": Exception(DAILY_429),
        "second-model": "answer from the fallback",
    })
    assert chained_llm.ask("ping") == "answer from the fallback"
    assert fake.tried == ["primary-model", "second-model"]


def test_all_models_spent_says_so_and_does_not_blame_a_moment(chained_llm):
    _swap_models(chained_llm, {
        "primary-model": Exception(DAILY_429),
        "second-model": Exception(DAILY_429),
        "third-model": Exception(DAILY_429),
    })
    msg = chained_llm.ask("ping")
    low = msg.lower()
    assert "today" in low and "free requests" in low, msg
    assert "wait a moment" not in low, "must not describe a daily cap as a pause"


def test_daily_quota_named_only_by_metric_still_falls_back(chained_llm):
    fake = _swap_models(chained_llm, {
        "primary-model": Exception(DAILY_429_BY_METRIC),
        "second-model": "answer from the fallback",
    })
    assert chained_llm.ask("ping") == "answer from the fallback"
    assert "primary-model" in chained_llm._exhausted_until


def test_per_minute_cap_stays_a_burst(chained_llm):
    """A PerMinute cap must not be parked until midnight."""
    _swap_models(chained_llm, {"primary-model": Exception(BURST_429_PER_MINUTE)})
    chained_llm.ask("ping")
    assert "primary-model" not in chained_llm._exhausted_until
    assert chained_llm._penalty_until > time.monotonic()


def test_spent_model_is_not_tried_again(chained_llm):
    fake = _swap_models(chained_llm, {
        "primary-model": Exception(DAILY_429),
        "second-model": "from the fallback",
    })
    chained_llm.ask("ping")
    fake.tried.clear()
    chained_llm._last_call = 0.0
    chained_llm.ask("ping")
    assert "primary-model" not in fake.tried, "a spent model was retried"


def test_daily_exhaustion_expires_at_local_midnight(chained_llm):
    _swap_models(chained_llm, {"primary-model": Exception(DAILY_429)})
    chained_llm.ask("ping")
    until = chained_llm._exhausted_until["primary-model"]
    midnight = (datetime.now() + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()
    assert abs(until - midnight) < 1, "the cap resets on the calendar day"


def test_burst_429_does_not_walk_the_chain(chained_llm):
    """A burst limit is project-wide, so switching models cannot evade it."""
    fake = _swap_models(chained_llm, {
        "primary-model": Exception("Error code: 429 - rate limit exceeded"),
    })
    assert "rate limited" in chained_llm.ask("ping").lower()
    assert fake.tried == ["primary-model"], "wasted requests on other models"


def test_burst_429_uses_the_delay_the_provider_gave(chained_llm):
    _swap_models(chained_llm, {"primary-model": Exception(
        "Error code: 429 - {'error': {'status': 'RESOURCE_EXHAUSTED', 'details': "
        "[{'retryDelay': '37s'}]}}"
    )})
    before = time.monotonic()
    chained_llm.ask("ping")
    assert chained_llm._penalty_until >= before + 36


def test_unavailable_model_is_skipped_for_the_session(chained_llm):
    fake = _swap_models(chained_llm, {
        "primary-model": Exception(UNAVAILABLE_404),
        "second-model": "still working",
    })
    assert chained_llm.ask("ping") == "still working"
    assert "primary-model" in chained_llm._unusable
    fake.tried.clear()
    chained_llm._last_call = 0.0
    chained_llm.ask("ping")
    assert "primary-model" not in fake.tried


def test_empty_reply_falls_through_to_the_next_model(chained_llm):
    """A reasoning model can burn the token budget before emitting any text."""
    _swap_models(chained_llm, {"primary-model": "", "second-model": "the real answer"})
    assert chained_llm.ask("ping") == "the real answer"


def test_empty_reply_from_every_model_is_reported(chained_llm):
    _swap_models(chained_llm, {
        "primary-model": "", "second-model": "", "third-model": "",
    })
    assert "empty" in chained_llm.ask("ping").lower()


# --- budget guard -----------------------------------------------------------


def test_routing_yields_the_last_of_the_daily_budget(chained_llm):
    _swap_models(chained_llm, {"primary-model": "PONG"})
    chained_llm.daily_limit = 3
    chained_llm.answer_reserve = 1
    for _ in range(2):
        chained_llm._last_call = 0.0
        chained_llm.ask("route me", purpose="route")
    chained_llm._last_call = 0.0
    assert "Routing is paused" in chained_llm.ask("route me", purpose="route")
    chained_llm._last_call = 0.0
    assert chained_llm.ask("answer me") == "PONG", "answers must still get through"


def test_daily_spend_is_counted_per_model(chained_llm):
    _swap_models(chained_llm, {"primary-model": "PONG"})
    chained_llm.ask("ping")
    assert chained_llm._spend_for("primary-model") == 1
    assert chained_llm._spend_today() == 1


def test_daily_spend_rolls_over_with_the_day(chained_llm):
    _swap_models(chained_llm, {"primary-model": "PONG"})
    chained_llm.ask("ping")
    chained_llm._spend_day = "1999-01-01"
    assert chained_llm._spend_for("primary-model") == 0


# --- honest status ----------------------------------------------------------


def test_status_reports_spent_models_and_claims_nothing(chained_llm):
    _swap_models(chained_llm, {
        "primary-model": Exception(DAILY_429),
        "second-model": Exception(DAILY_429),
        "third-model": Exception(DAILY_429),
    })
    chained_llm.ask("ping")
    status = chained_llm.get_status()
    assert status["available"] is False, "health must not claim a usable core"
    assert status["exhausted_models"] == [
        "primary-model", "second-model", "third-model",
    ]
    assert status["last_error_kind"] == "daily"


def test_status_names_the_model_that_answered(chained_llm):
    _swap_models(chained_llm, {
        "primary-model": Exception(DAILY_429),
        "second-model": "from the fallback",
    })
    chained_llm.ask("ping")
    status = chained_llm.get_status()
    assert status["active_model"] == "second-model"
    assert status["available"] is True, "a working model keeps the core available"
    assert status["exhausted_models"] == ["primary-model"]


# --- transient provider failures --------------------------------------------

TRANSIENT_503 = (
    "Error code: 503 - {'error': {'code': 503, 'message': 'The model is "
    "overloaded. Please try again later.', 'status': 'UNAVAILABLE'}}"
)
TRANSIENT_TIMEOUT = "Request timed out."


def test_busy_model_falls_back_to_the_next_model(chained_llm):
    """A 503 means the provider is busy, not that the model is spent."""
    fake = _swap_models(chained_llm, {
        "primary-model": Exception(TRANSIENT_503),
        "second-model": "answer from the fallback",
    })
    assert chained_llm.ask("ping") == "answer from the fallback"
    assert fake.tried == ["primary-model", "second-model"]
    assert chained_llm._exhausted_until == {}, "a busy model is not spent"
    assert chained_llm._unusable == set()
    assert chained_llm._penalty_until == 0.0, "a busy model is not rate limited"


def test_timeout_falls_back_on_the_streaming_path(chained_llm):
    fake = _swap_models(chained_llm, {
        "primary-model": Exception(TRANSIENT_TIMEOUT),
        "second-model": "streamed fallback",
    })
    assert "".join(chained_llm.ask_stream("ping")) == "streamed fallback"
    assert fake.tried == ["primary-model", "second-model"]


def test_every_model_busy_is_reported_as_overload(chained_llm):
    _swap_models(chained_llm, {
        "primary-model": Exception(TRANSIENT_503),
        "second-model": Exception(TRANSIENT_503),
        "third-model": Exception(TRANSIENT_503),
    })
    low = chained_llm.ask("ping").lower()
    assert "overloaded" in low, low
    assert "try again" in low, low


def test_timeout_is_not_a_daily_cap_or_a_burst(chained_llm):
    assert chained_llm._classify(TRANSIENT_TIMEOUT) == "transient"
    assert chained_llm._classify(TRANSIENT_503) == "transient"


# --- streaming path gates ----------------------------------------------------
# ask_stream is the path the HUD chat uses, and it must refuse the same
# requests ask() refuses: walking the whole chain after a per-project signal
# is what turned one daily cap into a retry storm in the logs.


def test_stream_routing_yields_the_last_of_the_daily_budget(chained_llm):
    # The streaming path must gate routing the way ask() does.
    _swap_models(chained_llm, {"primary-model": "PONG"})
    chained_llm.daily_limit = 3
    chained_llm.answer_reserve = 1
    for _ in range(2):
        chained_llm._last_call = 0.0
        assert "".join(chained_llm.ask_stream("route me", purpose="route")) == "PONG"
    chained_llm._last_call = 0.0
    assert "Routing is paused" in "".join(chained_llm.ask_stream("route me", purpose="route"))
    chained_llm._last_call = 0.0
    assert "".join(chained_llm.ask_stream("answer me")) == "PONG", "answers must still get through"



def test_stream_stops_at_the_daily_cap_instead_of_walking_the_chain(chained_llm):
    fake = _swap_models(chained_llm, {
        "primary-model": Exception(DAILY_429),
        "second-model": "never spoken",
        "third-model": "never spoken",
    })
    spoken = "".join(chained_llm.ask_stream("ping"))
    assert fake.tried == ["primary-model"], "wasted requests after the cap was known"
    assert "free requests" in spoken.lower(), spoken


def test_stream_refuses_while_a_burst_penalty_is_active(chained_llm):
    fake = _swap_models(chained_llm, {"primary-model": "never spoken"})
    chained_llm._penalty_until = time.monotonic() + 30
    spoken = "".join(chained_llm.ask_stream("ping"))
    assert fake.tried == [], "fired into an active penalty"
    assert "rate limited" in spoken.lower(), spoken


def test_stream_walks_the_chain_for_a_model_specific_signal(chained_llm):
    """An unavailable model is one model's problem; the chain is still worth it."""
    fake = _swap_models(chained_llm, {
        "primary-model": Exception(UNAVAILABLE_404),
        "second-model": "streamed fallback",
    })
    assert "".join(chained_llm.ask_stream("ping")) == "streamed fallback"
    assert fake.tried == ["primary-model", "second-model"]


