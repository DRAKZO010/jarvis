"""Capability catalogue and LLM routing tests.

The pattern table runs first and is unaffected by anything here. What these
tests protect is the second path: when speech does not match a pattern, the
model is asked to choose an action from a described catalogue, and its choice
must pass a deny-by-default allowlist. The dangerous failure mode is a model
(or an utterance that talks it into one) selecting ``execute`` or
``delete_file``, so that case is asserted explicitly rather than left to
inspection.
"""

from __future__ import annotations

import pytest

from mcp.capabilities import (
    ACTION_CATALOGUE,
    LLM_ROUTABLE_ACTIONS,
    blocked_actions,
    build_routing_prompt,
    describe_actions,
    is_routable,
)

# Actions that must never be selectable through the model: they run code,
# synthesise input, destroy data, or change the machine's power state.
MUST_BE_BLOCKED = {
    "execute",
    "kill_process",
    "type_text",
    "press_key",
    "hotkey",
    "mouse_click",
    "mouse_move",
    "delete_file",
    "delete_screenshots",
    "delete_last_screenshot",
    "lock_pc",
    "sleep_pc",
    "hibernate_pc",
    "shutdown_pc",
    "restart_pc",
    "fix_error",
}


# --------------------------------------------------------------------------- #
# Catalogue integrity
# --------------------------------------------------------------------------- #

def test_every_table_action_has_a_catalogue_description(app_module):
    """A new pattern must not be invisible to the router."""
    table = {name for _, name in app_module._ACTION_PATTERNS}
    missing = sorted(table - set(ACTION_CATALOGUE))
    assert not missing, f"actions with no catalogue description: {missing}"


def test_catalogue_has_no_dead_entries(app_module):
    """Only 'chat' is allowed to be described without a table row."""
    table = {name for _, name in app_module._ACTION_PATTERNS}
    dead = sorted(set(ACTION_CATALOGUE) - table - {"chat"})
    assert not dead, f"catalogue describes actions that do not exist: {dead}"


def test_every_description_is_meaningful():
    """Bare snake_case names are what the router was missing to begin with."""
    for action, description in ACTION_CATALOGUE.items():
        assert description.strip(), f"{action} has an empty description"
        assert len(description.strip()) > 10, f"{action} description is too thin"
        assert "_" not in description, f"{action} description is not prose"


def test_routable_and_blocked_partition_the_catalogue():
    """Deny-by-default: the two sets must cover everything, with no overlap."""
    routable, blocked = set(LLM_ROUTABLE_ACTIONS), blocked_actions()
    assert routable | blocked == set(ACTION_CATALOGUE)
    assert not routable & blocked, f"both routable and blocked: {routable & blocked}"


def test_dangerous_actions_are_blocked_from_the_model():
    assert MUST_BE_BLOCKED <= blocked_actions(), (
        "the model can choose: "
        f"{sorted(MUST_BE_BLOCKED - blocked_actions())}"
    )


def test_chat_is_routable():
    """Blocking 'chat' would make unmatched speech unreachable."""
    assert is_routable("chat")


def test_blocked_actions_are_not_routable():
    for action in sorted(blocked_actions()):
        assert not is_routable(action), f"{action} must not be routable"


# --------------------------------------------------------------------------- #
# Prompt contents
# --------------------------------------------------------------------------- #

def test_prompt_describes_actions_rather_than_listing_bare_names():
    """The model needs intent information, not just the action vocabulary."""
    prompt = build_routing_prompt("can you remind me about lunch")
    assert "add_task: " in prompt, "catalogue is not being sent as descriptions"
    assert "open_app: " in prompt


def test_prompt_names_blocked_actions_so_the_model_does_not_choose_them():
    prompt = build_routing_prompt("shut down")
    for action in ("execute", "delete_file", "shutdown_pc"):
        assert action in prompt, f"{action} is not named as blocked in the prompt"
    assert "Never choose these" in prompt


def test_prompt_includes_the_utterance_and_json_contract():
    prompt = build_routing_prompt("play some jazz")
    assert "play some jazz" in prompt
    assert '"action"' in prompt and '"payload"' in prompt
    assert "chat" in prompt


def test_describe_actions_omits_blocked_entries():
    rendered = describe_actions()
    for action in sorted(blocked_actions()):
        assert f"\n  {action}:" not in rendered, f"{action} leaked into the menu"


def test_describe_actions_is_sorted_and_covers_routable():
    lines = [line.strip().split(":")[0] for line in describe_actions().splitlines() if line.strip()]
    assert lines == sorted(lines), "catalogue order should be stable for caching"
    assert set(lines) == set(LLM_ROUTABLE_ACTIONS)


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "response",
    [
        '{"action":"open_app","payload":"notepad"}',
        '  {"action":"open_app","payload":"notepad"}  ',
        '```json\n{"action":"open_app","payload":"notepad"}\n```',
        '```\n{"action":"open_app","payload":"notepad"}\n```',
        'Sure! {"action":"open_app","payload":"notepad"}',
    ],
    ids=["plain", "padded", "fenced-json", "fenced-bare", "prose-wrapped"],
)
def test_extract_route_json_accepts_expected_shapes(app_module, response):
    data = app_module._extract_route_json(response)
    assert data == {"action": "open_app", "payload": "notepad"}


@pytest.mark.parametrize(
    "response",
    ["", "   ", "no json here", "[1, 2, 3]", '{"action": ', "{not json}"],
    ids=["empty", "blank", "prose", "list", "truncated", "malformed"],
)
def test_extract_route_json_rejects_unusable_responses(app_module, response):
    assert app_module._extract_route_json(response) is None


# --------------------------------------------------------------------------- #
# Routing behaviour
# --------------------------------------------------------------------------- #

def _fake_ask(app_module, monkeypatch, reply, recorded=None):
    """Replace the provider call. ``recorded`` may be a dict or a list."""

    def ask(prompt, **kwargs):
        if isinstance(recorded, dict):
            recorded.update(prompt=prompt, **kwargs)
        elif recorded is not None:
            recorded.append(prompt)
        return reply

    monkeypatch.setattr(app_module.llm_mcp, "ask", ask)


def test_routing_is_deterministic(app_module, monkeypatch):
    """Classification must not vary run to run, so temperature is pinned to 0."""
    recorded = {}
    _fake_ask(app_module, monkeypatch, '{"action":"chat","payload":""}', recorded)
    app_module._llm_parse_command("why is the sky blue")
    assert recorded["temperature"] == 0.0
    assert recorded["use_history"] is False


def test_routing_returns_the_chosen_action_and_payload(app_module, monkeypatch):
    _fake_ask(app_module, monkeypatch, '{"action":"add_task","payload":"buy milk"}')
    assert app_module._llm_parse_command("i need milk") == ("add_task", "buy milk")


def test_routing_tolerates_a_fenced_response(app_module, monkeypatch):
    _fake_ask(
        app_module, monkeypatch,
        '```json\n{"action":"set_timer","payload":"10 minutes"}\n```',
    )
    assert app_module._llm_parse_command("count down for a bit") == ("set_timer", "10 minutes")


def test_empty_payload_becomes_empty_string(app_module, monkeypatch):
    _fake_ask(app_module, monkeypatch, '{"action":"list_tasks","payload":""}')
    assert app_module._llm_parse_command("what is on my plate") == ("list_tasks", "")


def test_routing_rejects_a_blocked_action(app_module, monkeypatch):
    """The model asking for a dangerous action must not be obeyed."""
    _fake_ask(app_module, monkeypatch, '{"action":"execute","payload":"format c:"}')
    assert app_module._llm_parse_command("please tidy the disk") is None


def test_routing_rejects_a_destructive_file_action(app_module, monkeypatch):
    _fake_ask(app_module, monkeypatch, '{"action":"delete_file","payload":"everything.docx"}')
    assert app_module._llm_parse_command("tidy up my documents") is None


def test_routing_rejects_a_power_action(app_module, monkeypatch):
    _fake_ask(app_module, monkeypatch, '{"action":"shutdown_pc","payload":""}')
    assert app_module._llm_parse_command("i am heading out") is None


def test_routing_rejects_an_unknown_action(app_module, monkeypatch):
    _fake_ask(app_module, monkeypatch, '{"action":"launch_missiles","payload":""}')
    assert app_module._llm_parse_command("do something clever") is None


def test_routing_survives_unusable_model_output(app_module, monkeypatch):
    _fake_ask(app_module, monkeypatch, "I am not sure what you mean.")
    assert app_module._llm_parse_command("hmm") is None


def test_routing_survives_a_provider_error(app_module, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("provider exploded")
    monkeypatch.setattr(app_module.llm_mcp, "ask", boom)
    assert app_module._llm_parse_command("anything") is None


# --------------------------------------------------------------------------- #
# Quota protection
# --------------------------------------------------------------------------- #

def test_repeat_utterance_is_served_from_cache(app_module, monkeypatch):
    """A stable decision must not cost a second daily-quota request."""
    calls = []

    def ask(prompt, **kwargs):
        calls.append(prompt)
        return '{"action":"add_task","payload":"buy milk"}'

    monkeypatch.setattr(app_module.llm_mcp, "ask", ask)
    app_module._ROUTE_CACHE.clear()

    first = app_module._llm_parse_command("remind me to buy milk")
    second = app_module._llm_parse_command("remind me to buy milk")

    assert first == second == ("add_task", "buy milk")
    assert len(calls) == 1, f"expected one provider call, made {len(calls)}"


def test_cache_ignores_case_and_extra_whitespace(app_module, monkeypatch):
    calls = []

    def ask(prompt, **kwargs):
        calls.append(prompt)
        return '{"action":"set_timer","payload":"10 minutes"}'

    monkeypatch.setattr(app_module.llm_mcp, "ask", ask)
    app_module._ROUTE_CACHE.clear()

    app_module._llm_parse_command("put a timer on for ten")
    app_module._llm_parse_command("  Put   a timer on for ten  ")
    assert len(calls) == 1, "whitespace and case should normalise to the same key"


def test_failures_are_not_cached(app_module, monkeypatch):
    """A failed call must stay retryable rather than poisoning the cache."""
    calls = []

    def ask(prompt, **kwargs):
        calls.append(prompt)
        return "no json here"

    monkeypatch.setattr(app_module.llm_mcp, "ask", ask)
    app_module._ROUTE_CACHE.clear()

    assert app_module._llm_parse_command("something odd") is None
    assert app_module._llm_parse_command("something odd") is None
    assert len(calls) == 2, "a failure should not be remembered as a decision"


def test_blocked_choices_are_not_cached(app_module, monkeypatch):
    """A rejected decision must be re-evaluated, not remembered as chat."""
    replies = [
        '{"action":"execute","payload":"format c:"}',
        '{"action":"chat","payload":""}',
    ]
    calls = []

    def ask(prompt, **kwargs):
        calls.append(prompt)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(app_module.llm_mcp, "ask", ask)
    app_module._ROUTE_CACHE.clear()

    assert app_module._llm_parse_command("tidy the disk") is None
    assert app_module._llm_parse_command("tidy the disk") == ("chat", "")


def test_route_cache_stays_bounded(app_module, monkeypatch):
    """The cache must not grow without limit during a long session."""
    monkeypatch.setattr(
        app_module.llm_mcp, "ask",
        lambda prompt, **k: '{"action":"chat","payload":""}',
    )
    app_module._ROUTE_CACHE.clear()
    for i in range(app_module._ROUTE_CACHE_MAX + 25):
        app_module._llm_parse_command(f"utterance number {i}")
    assert len(app_module._ROUTE_CACHE) <= app_module._ROUTE_CACHE_MAX


def test_blocked_choice_falls_through_to_conversation(app_module, monkeypatch):
    """End to end: a blocked pick becomes chat, not silence and not execution."""
    _fake_ask(app_module, monkeypatch, '{"action":"execute","payload":"del /q"}')
    action, _payload = app_module._detect_action("wipe the slate clean")
    assert action == "chat", f"expected conversation fallback, got {action!r}"


# --------------------------------------------------------------------------- #
# Questions must not pay for a routing request they cannot benefit from
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text",
    [
        "how do i reverse a linked list",
        "what is the capital of peru",
        "why is the sky blue",
        "who was the first president of kenya",
        "when did the war end",
        "where is the nearest pharmacy",
        "which editor do you prefer",
        "is that safe to do",
        "do you know any good books",
        "can you explain recursion",
        "tell me a joke",
        "explain quantum tunnelling",
        "describe the water cycle",
        "give me a summary of the news",
        "do you think it will rain",
    ],
)
def test_questions_skip_the_routing_request(app_module, monkeypatch, text):
    """A question costs one request, not two: no routing call, then answer."""
    calls = []
    _fake_ask(app_module, monkeypatch, '{"action":"chat","payload":""}', calls)
    app_module._ROUTE_CACHE.clear()
    action, _payload = app_module._detect_action(text)
    assert action == "chat"
    assert not calls, f"spent a routing request on the question {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "put the kettle on for 4 minutes",
        "pull up the weather for lisbon",
        "jot down that the garage door is broken",
        "track down that tax document",
        "any news on the stock market",
    ],
)
def test_commands_still_consult_the_model(app_module, monkeypatch, text):
    """The shortcut must only catch questions, not swallow commands.

    These phrasings deliberately miss the pattern table, so they are the ones
    that actually depend on the model.
    """
    calls = []
    _fake_ask(
        app_module, monkeypatch,
        '{"action":"add_task","payload":"something"}', calls,
    )
    app_module._ROUTE_CACHE.clear()
    action, _payload = app_module._detect_action(text)
    assert action == "add_task"
    assert calls, f"expected a routing request for the command {text!r}"


def test_table_matches_still_never_reach_the_model(app_module, monkeypatch):
    """A phrase the table already handles must not spend a request."""
    calls = []
    _fake_ask(app_module, monkeypatch, '{"action":"shutdown_pc","payload":""}', calls)
    app_module._ROUTE_CACHE.clear()
    for text, expected in (("remind me to buy milk", "set_reminder"),
                           ("open the window", "open_app")):
        assert app_module._detect_action(text)[0] == expected
    assert not calls, "the model was consulted for a table-matched command"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("how do i reverse a linked list", True),
        ("what time is it", True),
        ("is python better than java?", True),
        ("remind me to buy milk", False),
        ("put the kettle on", False),
        ("open the window", False),
        ("", False),
        ("   ", False),
    ],
)
def test_question_detection(app_module, text, expected):
    assert app_module._looks_like_question(text) is expected


def test_punctuated_question_is_still_a_question(app_module, monkeypatch):
    calls = []
    _fake_ask(app_module, monkeypatch, '{"action":"chat","payload":""}', calls)
    app_module._ROUTE_CACHE.clear()
    assert app_module._detect_action("how do i reverse a linked list?")[0] == "chat"
    assert not calls


def test_conversation_choice_is_honoured(app_module, monkeypatch):
    _fake_ask(app_module, monkeypatch, '{"action":"chat","payload":""}')
    action, _payload = app_module._detect_action("why is the sky blue")
    assert action == "chat"


# --------------------------------------------------------------------------- #
# The regex path must be unaffected
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,expected",
    [
        ("shut down", "shutdown_pc"),
        ("restart the computer", "restart_pc"),
        ("lock the computer", "lock_pc"),
        ("delete the file old.txt", "delete_file"),
        ("type hello world", "type_text"),
        ("click at 50 60", "mouse_click"),
    ],
)
def test_exact_destructive_commands_still_work(app_module, text, expected):
    """Blocking the model must not remove these from the explicit patterns."""
    matched = app_module._match_pattern(text)
    assert matched is not None, f"no pattern matched {text!r}"
    assert matched[0] == expected


def test_pattern_table_still_takes_priority_over_the_model(app_module, monkeypatch):
    """A recognised command must never reach the model at all."""
    calls = []
    _fake_ask(app_module, monkeypatch, '{"action":"shutdown_pc","payload":""}', calls)
    action, _payload = app_module._detect_action("what time is it")
    assert action == "time"
    assert not calls, "the model was consulted for a command the table recognises"
