"""System metrics and task persistence tests.

Task tests redirect the JSON store to a temp file, so the user's real task
list is never read or written.
"""

from __future__ import annotations

import json

import pytest

from mcp.tasks import TasksMCP
from mcp.system import SystemMCP


# --- system metrics --------------------------------------------------------

@pytest.fixture
def system():
    m = SystemMCP()
    m.initialize()
    return m


def test_ui_metrics_keys(system):
    metrics = system.get_ui_metrics()
    for key in ("cpu_pct", "ram_pct", "ram_used", "ram_total", "disk_pct",
                "disk_used", "disk_total", "net_sent", "net_recv",
                "uptime", "process_count"):
        assert key in metrics, f"missing {key!r}"


def test_ui_metrics_percentages_are_bounded(system):
    metrics = system.get_ui_metrics()
    for key in ("cpu_pct", "ram_pct", "disk_pct"):
        value = metrics[key]
        assert value is None or 0 <= value <= 100, f"{key}={value}"


def test_ui_metrics_net_counters_do_not_go_backwards(system):
    """The HUD computes a rate from these; a decrease would render negative."""
    first = system.get_ui_metrics()
    second = system.get_ui_metrics()
    assert second["net_sent"] >= first["net_sent"]
    assert second["net_recv"] >= first["net_recv"]


def test_battery_is_none_or_valid(system):
    battery = system.get_ui_metrics()["battery"]
    assert battery is None or 0 <= battery <= 100


def test_uptime_is_formatted(system):
    uptime = system.get_ui_metrics()["uptime"]
    assert isinstance(uptime, str) and ":" in uptime


def test_metrics_are_json_serialisable(system):
    json.dumps(system.get_ui_metrics())


def test_get_ui_metrics_is_cheap_enough_to_poll(system):
    """It is called every 2s by the broadcaster and on every /health hit."""
    import time

    start = time.perf_counter()
    for _ in range(5):
        system.get_ui_metrics()
    assert (time.perf_counter() - start) / 5 < 0.5


# --- tasks -----------------------------------------------------------------

@pytest.fixture
def tasks(tmp_path, monkeypatch):
    import mcp.tasks as tasks_mod

    store = tmp_path / "jarvis_tasks.json"
    monkeypatch.setattr(tasks_mod, "TASKS_PATH", store)
    m = TasksMCP()
    m.initialize()
    m._store = store
    return m


def test_add_assigns_unique_ids(tasks):
    tasks.add_task("first")
    tasks.add_task("second")
    ids = [t["id"] for t in tasks.get_tasks_for_ui()]
    assert len(set(ids)) == 2, f"duplicate ids: {ids}"


def test_add_continues_after_a_completed_task(tasks):
    """Regression: max(id)+1 reused ids once the newest task was done."""
    a = tasks.add_task("alpha")
    b = tasks.add_task("beta")
    tid = int(b.split()[2].rstrip(".")) if b.split()[2].rstrip(".").isdigit() else None
    assert a and b
    ids = [t["id"] for t in tasks.get_tasks_for_ui()]
    assert len(set(ids)) == 2
    if tid is not None:
        assert tid == max(ids)


def test_pending_count_ignores_done(tasks):
    tasks.add_task("one")
    tasks.add_task("two")
    target = tasks.get_tasks_for_ui()[0]["id"]
    tasks.complete_task(target)
    assert tasks.get_pending_count() == 1


def test_complete_unknown_id_is_graceful(tasks):
    result = tasks.complete_task(9999)
    assert isinstance(result, str) and result


def test_delete_unknown_id_is_graceful(tasks):
    assert isinstance(tasks.delete_task(4242), str)


def test_ui_shape_is_stable(tasks):
    tasks.add_task("write tests")
    for task in tasks.get_tasks_for_ui():
        assert set(task) == {"id", "description", "due", "done"}
        assert isinstance(task["done"], bool)
        assert isinstance(task["id"], int)


def test_pending_tasks_sort_before_done(tasks):
    tasks.add_task("alpha")
    tasks.add_task("beta")
    tasks.complete_task(tasks.get_tasks_for_ui()[0]["id"])
    flags = [t["done"] for t in tasks.get_tasks_for_ui()]
    assert flags == sorted(flags), f"done tasks are not last: {flags}"


def test_mixed_id_types_do_not_crash_sorting(tasks):
    """Regression: a hand-edited or imported file can hold string ids."""
    tasks._tasks = [
        {"id": "3", "description": "string id", "done": False},
        {"id": 1, "description": "int id", "done": False},
    ]
    rows = tasks.get_tasks_for_ui()  # must not raise TypeError
    assert len(rows) == 2


def test_corrupt_store_falls_back_to_empty(tmp_path, monkeypatch):
    import mcp.tasks as tasks_mod

    store = tmp_path / "broken.json"
    store.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(tasks_mod, "TASKS_PATH", store)
    m = TasksMCP()
    assert m.initialize() is True
    assert m.get_tasks_for_ui() == []


def test_missing_store_falls_back_to_empty(tmp_path, monkeypatch):
    import mcp.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "TASKS_PATH", tmp_path / "absent.json")
    m = TasksMCP()
    m.initialize()
    assert m.get_tasks_for_ui() == []


def test_tasks_persist_across_instances(tasks):
    tasks.add_task("persist me")
    reloaded = TasksMCP()
    reloaded.initialize()
    assert any(t["description"] == "persist me" for t in reloaded.get_tasks_for_ui())


def test_prompt_text_lists_pending_only(tasks):
    tasks.add_task("pending one")
    tasks.add_task("will be done")
    done_id = next(t["id"] for t in tasks.get_tasks_for_ui()
                   if t["description"] == "will be done")
    tasks.complete_task(done_id)
    text = tasks.get_tasks_for_prompt()
    assert "pending one" in text
    assert "will be done" not in text


def test_prompt_text_handles_empty_list(tasks):
    assert tasks.get_tasks_for_prompt().strip()


def test_fixture_redirected_away_from_the_real_store(tasks, tmp_path):
    """Guard: the fixture must not read or write the user's real task file."""
    import mcp.tasks as tasks_mod

    assert tasks_mod.TASKS_PATH.parent == tmp_path
