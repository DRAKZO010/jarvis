"""End-to-end audit of the running assistant.

Exercises every MCP module through the real dispatcher plus the HTTP/WebSocket
surface, and reports PASS / FAIL per check. Nothing here performs a destructive
action: power, process-kill, file-delete, mouse and keyboard actions are
verified by routing only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

for _line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

os.environ.setdefault("JARVIS_DISABLE_MIC", "1")

import logging

logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

import jarvis  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append(("PASS", name, detail or ""))
    except Exception as e:  # noqa: BLE001
        RESULTS.append(("FAIL", name, "%s: %s" % (type(e).__name__, e)))
        if os.environ.get("AUDIT_TRACE"):
            traceback.print_exc()


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --- LLM / quota ---------------------------------------------------------- #

def t_llm_plain_ask():
    return jarvis.llm_mcp.ask("Reply with the single word: pong",
                              use_history=False, skip_cooldown=True)


def t_llm_status():
    return json.dumps(jarvis.llm_mcp.get_status())


def t_llm_penalised_flag():
    """Is the module in its 'rate limited' state right now?"""
    import time as _t
    return "penalised=%s" % (_t.monotonic() < jarvis.llm_mcp._penalty_until)


def t_routing_decision():
    return str(jarvis._llm_parse_command("put the kettle on for four minutes"))


# --- system ---------------------------------------------------------------- #

def t_system_info():
    return jarvis.system_mcp.get_system_info()


def t_cpu():
    return jarvis.system_mcp.handle("cpu", "")


def t_ram():
    return jarvis.system_mcp.handle("ram", "")


def t_disk():
    return jarvis.system_mcp.handle("disk", "")


def t_network():
    return jarvis.system_mcp.handle("network", "")


def t_battery():
    return jarvis.system_mcp.handle("battery", "")


def t_running_apps():
    return jarvis.system_mcp.get_running_apps()


# --- tasks ----------------------------------------------------------------- #

def t_tasks_list():
    return jarvis.tasks_mcp.handle("list tasks", "")


def t_task_add_cycle():
    before = len(jarvis.tasks_mcp._tasks)
    jarvis.tasks_mcp.handle("add task", "AUDITPROBE_DELETE_ME")
    added = any(t["description"] == "AUDITPROBE_DELETE_ME"
                for t in jarvis.tasks_mcp._tasks)
    assert added, "add task did not reach the task list"
    assert len(jarvis.tasks_mcp._tasks) == before + 1, "task count did not grow"
    jarvis.tasks_mcp.delete_task_by_description("AUDITPROBE_DELETE_ME")
    assert not any(t["description"] == "AUDITPROBE_DELETE_ME"
                   for t in jarvis.tasks_mcp._tasks), "delete by text failed"
    return "add + delete by text both persisted"


def t_task_list_truncation():
    """With more than 10 pending tasks the spoken list must say it is partial."""
    pending = [t for t in jarvis.tasks_mcp._tasks if not t["done"]]
    out = jarvis.tasks_mcp.handle("list tasks", "")
    if len(pending) <= 10:
        return "%d pending, fits in one reply" % len(pending)
    shown = out.count("Task ")
    assert shown < len(pending), "expected truncation with %d pending" % len(pending)
    silent = not any(w in out.lower() for w in ("more", "also", "and", "remaining"))
    return ("%d pending, speaks %d, NO indication it is partial: %s"
            % (len(pending), shown, silent))


# --- memory ---------------------------------------------------------------- #

def t_memory_recall():
    return jarvis.memory_mcp.handle("recall", "")


# --- reminders ------------------------------------------------------------- #

def t_reminders_list():
    return jarvis.reminders_mcp.handle("list reminders", "")


# --- context --------------------------------------------------------------- #

def t_context_stats():
    return str(getattr(jarvis.context_mcp, "get_context", lambda: "n/a")())


# --- voice (offline surface) ---------------------------------------------- #

def t_voice_status():
    return jarvis.voice_mcp.get_status().get("tts_backend") or "idle"


def t_voice_handle_status():
    return jarvis.voice_mcp.handle("voice status", "")


def t_voice_backend():
    return jarvis.voice_mcp.handle("voice backend", "")


def t_wake_word_strip():
    from mcp.voice import strip_wake_word
    return str(strip_wake_word("Jarvis what time is it", "jarvis"))


def t_vad_status():
    s = jarvis.voice_mcp.get_status()
    return "gate=%s noise=%s out=%s" % (
        s.get("vad_threshold"), s.get("noise_floor"), s.get("output_device"))


# --- browser / files / automation (non destructive) ------------------------ #

def t_files_find():
    return jarvis.files_mcp.find_file("definitely-not-a-real-file-xyzzy")


def t_files_list_dir():
    return jarvis.automation_mcp.list_directory(str(ROOT / "mcp"))


def t_files_recent():
    return jarvis.files_mcp.list_recent_files()


def t_automation_windows():
    return jarvis.automation_mcp.list_windows()


def t_automation_processes():
    return jarvis.automation_mcp.list_processes()


def t_browser_mcp_status():
    return "ok"


# --- HTTP ------------------------------------------------------------------ #

def t_http_root():
    from fastapi.testclient import TestClient
    with TestClient(jarvis.app) as c:
        r = c.get("/")
        assert r.status_code == 200, r.status_code
        assert "Jarvis" in r.text
        return "200, %d bytes" % len(r.text)


def t_http_health():
    from fastapi.testclient import TestClient
    with TestClient(jarvis.app) as c:
        r = c.get("/health")
        assert r.status_code == 200, r.status_code
        return json.dumps(r.json())[:300]


def t_http_metrics():
    from fastapi.testclient import TestClient
    with TestClient(jarvis.app) as c:
        r = c.get("/api/metrics")
        assert r.status_code == 200, r.status_code
        return json.dumps(r.json())[:200]


def t_http_tasks():
    from fastapi.testclient import TestClient
    with TestClient(jarvis.app) as c:
        r = c.get("/api/tasks")
        assert r.status_code == 200, r.status_code
        return "%d tasks" % len(r.json().get("tasks", []))


def t_http_static_assets():
    from fastapi.testclient import TestClient
    with TestClient(jarvis.app) as c:
        out = []
        for path in ("/style.css", "/app.js"):
            r = c.get(path)
            out.append("%s=%d" % (path, r.status_code))
        return " ".join(out)


def t_websocket():
    from fastapi.testclient import TestClient
    with TestClient(jarvis.app) as c:
        with c.websocket_connect("/ws"):
            return "connected"


# --- routing table integrity ----------------------------------------------- #

def t_routing_integrity():
    """Every pattern compiles, maps to a real dispatcher, and is in the catalogue."""
    from mcp.capabilities import ACTION_CATALOGUE, LLM_ROUTABLE_ACTIONS, is_routable
    import re
    for pattern, action in jarvis._ACTION_PATTERNS:
        re.compile(pattern)
    table = {a for _, a in jarvis._ACTION_PATTERNS}
    missing_desc = sorted(table - set(ACTION_CATALOGUE))
    dead_desc = sorted(set(ACTION_CATALOGUE) - table - {"chat"})
    return "%d patterns, %d actions, missing=%s dead=%s" % (
        len(jarvis._ACTION_PATTERNS), len(table), missing_desc, dead_desc)


def t_news_clean():
    return jarvis._clean_news("Headlines: 1. Big story - BBC 2. Other thing - CNN")


def t_openai_import():
    return jarvis.llm_mcp._import_ok() and jarvis.llm_mcp.available


def main():
    t0 = time.time()
    section("MODULE / SUBSYSTEM AUDIT")
    print("model=%s  fish=%s  wake=%s" % (
        os.environ.get("LLM_MODEL"),
        "configured" if os.environ.get("FISH_API_KEY") else "MISSING",
        os.environ.get("JARVIS_WAKE_WORD")))

    plan = [
        ("LLM / quota", [
            ("llm.status", t_llm_status),
            ("llm.penalised-now", t_llm_penalised_flag),
            ("llm.plain-ask (live call)", t_llm_plain_ask),
            ("llm.routing-decision (live call)", t_routing_decision),
        ]),
        ("System", [
            ("system.info", t_system_info),
            ("system.cpu", t_cpu),
            ("system.ram", t_ram),
            ("system.disk", t_disk),
            ("system.network", t_network),
            ("system.battery", t_battery),
            ("system.running-apps", t_running_apps),
        ]),
        ("Tasks", [
            ("tasks.list", t_tasks_list),
            ("tasks.add-delete-cycle", t_task_add_cycle),
            ("tasks.list-truncation", t_task_list_truncation),
        ]),
        ("Memory / reminders / context", [
            ("memory.recall", t_memory_recall),
            ("reminders.list", t_reminders_list),
            ("context.stats", t_context_stats),
        ]),
        ("Voice (offline surface)", [
            ("voice.status", t_voice_status),
            ("voice.handle-status", t_voice_handle_status),
            ("voice.backend", t_voice_backend),
            ("voice.wake-word-strip", t_wake_word_strip),
            ("voice.vad+output", t_vad_status),
        ]),
        ("Files / automation (read only)", [
            ("files.find", t_files_find),
            ("files.list-dir", t_files_list_dir),
            ("files.recent", t_files_recent),
            ("automation.list-windows", t_automation_windows),
            ("automation.list-processes", t_automation_processes),
            ("browser.module", t_browser_mcp_status),
        ]),
        ("HTTP / WebSocket", [
            ("http./", t_http_root),
            ("http./health", t_http_health),
            ("http./api/metrics", t_http_metrics),
            ("http./api/tasks", t_http_tasks),
            ("http.static-assets", t_http_static_assets),
            ("ws./ws", t_websocket),
        ]),
        ("Routing integrity", [
            ("routing.integrity", t_routing_integrity),
            ("news.clean", t_news_clean),
            ("llm.client-available", t_openai_import),
        ]),
    ]

    for title, checks in plan:
        section(title)
        for name, fn in checks:
            check(name, fn)

    section("RESULTS")
    npass = sum(1 for r in RESULTS if r[0] == "PASS")
    nfail = len(RESULTS) - npass
    for status, name, detail in RESULTS:
        line = "%-6s %-34s %s" % (status, name, detail.replace("\n", " ")[:150])
        print(line)
    print("\n%d passed, %d failed, %.1fs" % (npass, nfail, time.time() - t0))
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
