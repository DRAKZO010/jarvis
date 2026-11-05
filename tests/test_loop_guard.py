"""Unit tests for LoopGuard & Context Compression."""

from mcp.loop_guard import LoopGuard, LoopGuardConfig, LoopVerdict


def test_loop_guard_allows_normal_calls():
    guard = LoopGuard(LoopGuardConfig(max_identical_calls=3))
    v = guard.check_call("open_app", "notepad")
    assert not v.blocked
    assert not v.warned


def test_loop_guard_catches_identical_calls():
    config = LoopGuardConfig(max_identical_calls=2, warn_before_block=True)
    guard = LoopGuard(config)

    v1 = guard.check_call("search_web", "weather")
    assert not v1.blocked and not v1.warned

    v2 = guard.check_call("search_web", "weather")
    assert not v2.blocked and not v2.warned

    # Third identical call triggers warning
    v3 = guard.check_call("search_web", "weather")
    assert not v3.blocked
    assert v3.warned

    # Fourth call is blocked
    v4 = guard.check_call("search_web", "weather")
    assert v4.blocked
    assert "Identical call" in v4.reason


def test_loop_guard_catches_ping_pong():
    config = LoopGuardConfig(ping_pong_window=4, warn_before_block=False)
    guard = LoopGuard(config)

    # Alternate A, B, A, B
    guard.check_call("get_tasks", "")
    guard.check_call("get_weather", "")
    guard.check_call("get_tasks", "")
    v = guard.check_call("get_weather", "")
    assert v.blocked
    assert "ping-pong" in v.reason


def test_loop_guard_context_compression():
    config = LoopGuardConfig(max_context_messages=6)
    guard = LoopGuard(config)

    messages = [
        {"role": "system", "content": "You are Jarvis"},
        {"role": "user", "content": "turn 1"},
        {"role": "tool", "content": "giant tool output " * 50},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
        {"role": "tool", "content": "more output"},
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "turn 3"},
        {"role": "assistant", "content": "reply 3"},
    ]

    compressed = guard.compress_context(messages)
    assert len(compressed) <= 6
    assert compressed[0]["role"] == "system"
    assert compressed[-1]["content"] == "reply 3"

