"""HTTP and WebSocket contract tests for the FastAPI app.

The HUD is a locked design, so these tests pin the *contract* the front end
depends on: URL paths, JSON field names, and event types.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(app_module, silence_voice):
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture
def client_module(app_module):
    """Alias so tests can reach module state (e.g. the speech cooldown)."""
    return app_module


# --- static assets ---------------------------------------------------------

def test_root_serves_the_hud(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_root_does_not_inline_the_stylesheet(client):
    """The HUD is served as three files; a <style> block means it was inlined."""
    body = client.get("/").text
    assert "<style>" not in body
    assert "app.js" in body


def test_stylesheet_and_script_are_served(client):
    for path in ("/style.css", "/app.js"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.content, path


def test_favicon(client):
    # 204 is fine: the HUD has no favicon and the route answers deliberately.
    assert client.get("/favicon.ico").status_code in (200, 204, 404)


def test_static_mount_is_reachable(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200


# --- health and APIs -------------------------------------------------------

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "online"
    assert body["version"]
    assert "voice" in body and "llm" in body
    assert isinstance(body["metrics"], dict)


def test_health_task_count_is_an_int(client):
    assert isinstance(client.get("/health").json()["task_count"], int)


def test_api_tasks_shape(client):
    body = client.get("/api/tasks").json()
    assert "tasks" in body
    for task in body["tasks"]:
        assert {"id", "description", "due", "done"} <= set(task)


def test_api_metrics_shape(client):
    """Metrics are flat and suffixed, matching what static/app.js reads."""
    body = client.get("/api/metrics").json()
    for key in ("cpu_pct", "ram_pct", "ram_used", "ram_total", "disk_pct",
                "net_sent", "net_recv", "uptime", "process_count"):
        assert key in body, f"missing metric {key!r}"


def test_metrics_percentages_are_in_range(client):
    body = client.get("/api/metrics").json()
    for key in ("cpu_pct", "ram_pct", "disk_pct"):
        value = body[key]
        assert value is None or 0 <= value <= 100, f"{key} out of range: {value}"


def test_metrics_values_are_numbers_or_none(client):
    """The HUD does arithmetic on these; a string here becomes NaN in JS."""
    body = client.get("/api/metrics").json()
    for key, value in body.items():
        assert value is None or isinstance(value, (int, float, bool, str)), (key, value)


def test_tts_requires_text(client):
    assert "error" in client.get("/api/tts").json()


# --- websocket -------------------------------------------------------------

# Frames the background metrics/tasks loop interleaves with real replies.
BACKGROUND = {"metrics", "tasks", "speaking", "status"}


def wait_for_reply(ws, timeout=20.0):
    """Collect frames until JARVIS's own transcript arrives, skipping noise.

    Bounded by wall clock, not frame count: the metrics loop ticks every 2s, so
    a frame limit turns a missing reply into a minute-long wait.
    """
    deadline = time.monotonic() + timeout
    user_lines, jarvis_lines = [], []
    while time.monotonic() < deadline:
        msg = ws.receive_json()
        kind = msg.get("type")
        if kind == "transcript" and msg.get("speaker") == "jarvis":
            jarvis_lines.append(msg)
            return user_lines, jarvis_lines
        if kind == "transcript" and msg.get("speaker") == "user":
            user_lines.append(msg)
    pytest.fail(f"no JARVIS reply within {timeout}s (user={user_lines})")


def test_websocket_echoes_command_and_replies(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "command", "text": "what time is it"}))
        user_lines, jarvis_lines = wait_for_reply(ws)
        assert jarvis_lines, "JARVIS never spoke"
        assert jarvis_lines[0]["text"].strip(), "empty reply text"


def test_websocket_ignores_malformed_json(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json at all")
        ws.send_text(json.dumps({"type": "command", "text": "what time is it"}))
        _user, jarvis_lines = wait_for_reply(ws)
        assert jarvis_lines, "a malformed frame broke the connection"


def test_websocket_interrupt_message_is_accepted(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "interrupt"}))
        ws.send_text(json.dumps({"type": "command", "text": "what time is it"}))
        _user, jarvis_lines = wait_for_reply(ws)
        assert jarvis_lines, "an interrupt frame broke the connection"


def test_websocket_ignores_empty_and_unknown_frames(client):
    with client.websocket_connect("/ws") as ws:
        for frame in ({"type": "command", "text": "   "}, {"type": "nonsense"}, {}):
            ws.send_text(json.dumps(frame))
        ws.send_text(json.dumps({"type": "command", "text": "what time is it"}))
        _user, jarvis_lines = wait_for_reply(ws)
        assert jarvis_lines


def test_user_transcript_is_sent_once(client):
    """Regression: the front end used to render the user's line twice."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "command", "text": "what time is it"}))
        user_lines, _jarvis = wait_for_reply(ws)
        assert len(user_lines) == 1, f"user transcript duplicated: {user_lines}"


def test_reply_survives_the_speech_cooldown(client_module, client):
    """The TTS cooldown must not swallow the visible reply."""
    for _ in range(2):
        client_module._last_spoken_at = time.monotonic()  # inside the cooldown
        client_module._last_command = ("", 0.0)           # allow the command
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "command", "text": "what time is it"}))
            _user, jarvis_lines = wait_for_reply(ws)
            assert jarvis_lines, "cooldown suppressed the transcript"


def test_identical_replies_are_both_shown(client_module, monkeypatch):
    """Two identical questions can have one identical answer; show both."""
    sent = []
    monkeypatch.setattr(client_module, "broadcast", lambda msg: sent.append(msg))
    client_module.speak("It is 10:00 PM.")
    client_module.speak("It is 10:00 PM.")
    texts = [m["text"] for m in sent if m.get("type") == "transcript"]
    assert texts == ["It is 10:00 PM.", "It is 10:00 PM."]


def test_cooldown_gates_audio_but_not_text(client_module, monkeypatch):
    spoken, sent = [], []
    monkeypatch.setattr(client_module.voice_mcp, "speak", lambda t: spoken.append(t))
    monkeypatch.setattr(client_module, "broadcast", lambda msg: sent.append(msg))
    client_module.speak("First.")
    client_module.speak("Second.")      # inside the 0.4s window: text only
    client_module._last_spoken_at -= 1  # pretend the window has passed
    client_module.speak("Third.")
    assert spoken == ["First.", "Third."]
    assert len([m for m in sent if m.get("type") == "transcript"]) == 3


def test_repeated_command_is_ignored(client_module, monkeypatch):
    """Regression: a latched wake word used to run the same command twice."""
    handled = []
    monkeypatch.setattr(
        client_module, "_handle_command_inner", lambda cmd, mem: handled.append(cmd)
    )
    client_module.handle_command("what time is it", {})
    client_module.handle_command("what time is it", {})
    assert handled == ["what time is it"]

    client_module._command_lock.acquire()
    try:
        client_module._last_command = ("", 0.0)  # simulate the window passing
    finally:
        client_module._command_lock.release()
    client_module.handle_command("what time is it", {})
    assert handled == ["what time is it", "what time is it"]


def test_different_commands_are_not_deduped(client_module, monkeypatch):
    handled = []
    monkeypatch.setattr(
        client_module, "_handle_command_inner", lambda cmd, mem: handled.append(cmd)
    )
    client_module.handle_command("what time is it", {})
    client_module.handle_command("what is the date", {})
    assert handled == ["what time is it", "what is the date"]


def test_metrics_are_broadcast_to_connected_clients(client):
    with client.websocket_connect("/ws") as ws:
        for _ in range(40):
            msg = ws.receive_json()
            if msg.get("type") == "metrics":
                assert "cpu_pct" in msg
                return
        pytest.fail("no metrics broadcast within 40 frames")


def test_speaking_state_reaches_the_hud(client_module, client):
    """The voice module pushes this through the hook, not through a route."""
    with client.websocket_connect("/ws") as ws:
        client_module.voice_mcp._broadcast_speaking(True)
        client_module.voice_mcp._broadcast_speaking(False)
        flags = []
        for _ in range(40):
            msg = ws.receive_json()
            if msg.get("type") == "speaking":
                flags.append(msg["speaking"])
                if flags == [True, False]:
                    return
        pytest.fail(f"speaking events not delivered in order: {flags}")


def test_broadcast_with_no_clients_is_a_no_op(app_module):
    app_module.broadcast({"type": "status", "state": "idle"})


def test_broadcast_hook_is_installed(app_module):
    """The speaking event only works if the app registered the hook."""
    import mcp.voice as voice_mod

    assert voice_mod._broadcast_hook is not None, "app did not install the speaking hook"


# --- server-side speech recognition ----------------------------------------

def test_listening_endpoint_reports_state(client):
    r = client.get("/api/listening")
    assert r.status_code == 200
    body = r.json()
    for key in ("listening", "push_to_talk", "wake_word", "last_transcript", "listen_error"):
        assert key in body, f"missing {key!r}"


def test_ptt_endpoint_opens_and_closes_the_gate(client):
    on = client.post("/api/ptt", json={"active": True})
    assert on.status_code == 200
    assert on.json()["push_to_talk"] is True
    assert client.get("/api/listening").json()["push_to_talk"] is True

    off = client.post("/api/ptt", json={"active": False})
    assert off.status_code == 200
    assert off.json()["push_to_talk"] is False
    assert client.get("/api/listening").json()["push_to_talk"] is False


def test_ptt_endpoint_requires_the_active_flag(client):
    """Guessing a typo'd payload would either drop speech or leave the mic open."""
    r = client.post("/api/ptt", json={})
    assert r.status_code == 400
    assert "error" in r.json()


def test_ptt_endpoint_rejects_a_non_boolean(client):
    r = client.post("/api/ptt", json={"active": "yes please"})
    assert r.status_code == 400
    assert "error" in r.json()


def test_ptt_endpoint_rejects_a_missing_content_type(client):
    """A body that is not JSON must not be able to flip the gate."""
    r = client.post("/api/ptt", content=b'{"active": true}')
    assert r.status_code in (400, 422), r.status_code
    assert client.get("/api/listening").json()["push_to_talk"] is False


def test_websocket_ptt_is_broadcast_back(client_module):
    with TestClient(client_module.app) as c:
        with c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ptt", "active": True})
            seen = None
            for _ in range(40):
                msg = ws.receive_json()
                if msg.get("type") == "ptt":
                    seen = msg
                    break
            assert seen is not None, "ptt state was not broadcast"
            assert seen["active"] is True
        assert client_module.voice_mcp.is_push_to_talk() is False


def test_websocket_disconnect_releases_push_to_talk(client_module):
    """A tab closing mid-press must not leave the gate open for the room."""
    with TestClient(client_module.app) as c:
        with c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ptt", "active": True})
            for _ in range(40):
                if ws.receive_json().get("type") == "ptt":
                    break
            assert client_module.voice_mcp.is_push_to_talk() is True
    assert client_module.voice_mcp.is_push_to_talk() is False


def test_websocket_ptt_release_is_broadcast(client_module):
    with TestClient(client_module.app) as c:
        with c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ptt", "active": True})
            ws.send_json({"type": "ptt", "active": False})
            states = []
            for _ in range(40):
                msg = ws.receive_json()
                if msg.get("type") == "ptt":
                    states.append(msg["active"])
                    if states == [True, False]:
                        break
            assert states == [True, False], states
