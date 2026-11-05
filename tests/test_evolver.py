"""Tests for the self-evolution subsystem.

The interesting question is not "does it work" but "can it get out". These
tests are mostly about the ways the assistant could hurt itself: editing the
rules that constrain it, editing the tests that check it, reaching outside the
project, or writing to the live tree without a human saying yes.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.evolver import (  # noqa: E402
    DEFAULT_GUARDRAILS,
    EvolutionMCP,
    Guardrails,
)


# ---------------------------------------------------------------------------
# helpers


def make_diff(path, old="", new="", context="line"):
    """A minimal unified diff that git will accept."""
    return (
        "--- a/{path}\n"
        "+++ b/{path}\n"
        "@@ -1,1 +1,1 @@\n"
        "-{old}\n"
        "+{new}\n"
    ).format(path=path, old=old, new=new)


def make_patch(path, old, new):
    """A real unified diff, the way a model would write one."""
    import difflib
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="a/" + path,
        tofile="b/" + path,
        n=3,
    ))


@pytest.fixture
def guardrails(tmp_path):
    rules = dict(DEFAULT_GUARDRAILS)
    rules["enabled"] = True
    rules["auto_apply"] = False
    path = tmp_path / "guardrails.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    return Guardrails(path)


@pytest.fixture
def evolver(guardrails, tmp_path, monkeypatch):
    """An evolver whose state, history and proposals live in a temp dir."""
    import mcp.evolver as evo

    monkeypatch.setattr(evo, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(evo, "HISTORY_PATH", tmp_path / "history.jsonl")
    monkeypatch.setattr(evo, "PROPOSALS_DIR", tmp_path / "proposals")
    monkeypatch.setattr(evo, "ROOT", tmp_path / "project")

    (tmp_path / "project").mkdir()
    inst = EvolutionMCP()
    inst.guardrails = guardrails
    return inst


@pytest.fixture
def sample_project(tmp_path):
    """A tiny project with one passing test, used as a sandbox source.

    The layout mirrors the real one: code in ``mcp/`` (which the rules allow
    changes to) and tests in ``tests/`` (which they protect).
    """
    project = tmp_path / "sample"
    (project / "mcp").mkdir(parents=True)
    (project / "tests").mkdir(parents=True)
    (project / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (project / "mcp" / "__init__.py").write_text("", encoding="utf-8")
    (project / "mcp" / "helper.py").write_text(
        "def value():\n    return 1\n", encoding="utf-8")
    (project / "tests" / "test_helper.py").write_text(
        "from mcp.helper import value\n\n\ndef test_value():\n    assert value() == 1\n",
        encoding="utf-8",
    )
    return project


# ---------------------------------------------------------------------------
# the rules


def test_default_guardrails_protect_the_supervisor():
    defaults = DEFAULT_GUARDRAILS
    for path in ("mcp/evolver.py", "evolution/", ".env"):
        assert path in defaults["protected_paths"]
    assert "tests/" in defaults["create_only_paths"]


def test_protected_paths_cannot_be_touched(evolver):
    for path in ("mcp/evolver.py", "evolution/guardrails.json",
                 ".env", "jarvis_evolution.json"):
        ok, reason = evolver.guardrails.check_paths([path])
        assert not ok
        assert "protected" in reason


def test_a_new_test_file_is_allowed_but_an_old_one_is_not(evolver):
    """Jarvis may add tests; it may not rewrite the ones that judge it."""
    ok, reason = evolver.guardrails.check_paths(
        ["tests/test_new_thing.py"], new_paths=["tests/test_new_thing.py"])
    assert ok, reason

    ok, reason = evolver.guardrails.check_paths(["tests/test_stt.py"])
    assert not ok
    assert "protected" in reason


def test_paths_outside_the_allowed_list_are_refused(evolver):
    ok, reason = evolver.guardrails.check_paths(["scripts/deploy.py"])
    assert not ok
    assert "outside the allowed paths" in reason


def test_allowed_paths_are_accepted(evolver):
    ok, _ = evolver.guardrails.check_paths(["mcp/voice.py"])
    assert ok
    ok, _ = evolver.guardrails.check_paths(["static/app.js"])
    assert ok
    ok, _ = evolver.guardrails.check_paths(["jarvis.py"])
    assert ok


def test_paths_escaping_the_project_are_refused(evolver):
    ok, reason = evolver.guardrails.check_paths(["../outside/evil.py"])
    assert not ok
    assert "outside the project" in reason

    ok, reason = evolver.guardrails.check_paths(["/etc/passwd"])
    assert not ok
    assert "outside the project" in reason


def test_too_many_files_is_refused(evolver):
    paths = ["mcp/a.py", "mcp/b.py", "mcp/c.py", "mcp/d.py"]
    ok, reason = evolver.guardrails.check_paths(paths)
    assert not ok
    assert "limit" in reason


def test_too_many_changed_lines_is_refused(evolver):
    diff = "--- a/mcp/a.py\n+++ b/mcp/a.py\n" + "".join(
        "+line %d\n" % i for i in range(200)
    )
    ok, reason = evolver.guardrails.check_size(diff)
    assert not ok
    assert "limit" in reason


def test_an_oversized_patch_is_refused(evolver):
    diff = "--- a/mcp/a.py\n+++ b/mcp/a.py\n+" + ("x" * 50000)
    ok, reason = evolver.guardrails.check_size(diff)
    assert not ok
    assert "size limit" in reason


# ---------------------------------------------------------------------------
# reading the model's reply


def test_a_well_formed_reply_is_parsed(evolver):
    reply = (
        "SUMMARY: Fix the off-by-one in the segmenter\n"
        "RATIONALE: The tail block is counted twice\n"
        "CONFIDENCE: 0.8\n"
        "DIFF:\n"
        "--- a/mcp/voice.py\n"
        "+++ b/mcp/voice.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    parsed = evolver._parse_reply(reply)
    assert parsed is not None
    assert parsed["summary"].startswith("Fix the off-by-one")
    assert parsed["confidence"] == pytest.approx(0.8)
    assert "+++ b/mcp/voice.py" in parsed["diff"]


def test_no_change_is_recognised(evolver):
    assert evolver._parse_reply("NOCHANGE") is None
    assert evolver._parse_reply("NOCHANGE - the code looks fine") is None


def test_a_reply_without_a_diff_is_rejected(evolver):
    assert evolver._parse_reply("SUMMARY: I would fix something") is None
    assert evolver._parse_reply("") is None


def test_a_fenced_diff_is_accepted(evolver):
    reply = (
        "Here is the change:\n"
        "```diff\n"
        "--- a/mcp/voice.py\n"
        "+++ b/mcp/voice.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
    )
    parsed = evolver._parse_reply(reply)
    assert parsed is not None
    assert "+++ b/mcp/voice.py" in parsed["diff"]


# ---------------------------------------------------------------------------
# the sandbox


def test_a_patch_that_breaks_the_tests_is_refused(evolver, sample_project):
    old = "def value():\n    return 1\n"
    new = "def value():\n    return 2\n"
    diff = make_patch("mcp/helper.py", old, new)
    # The test asserts value() == 1, so this patch should fail the tests.
    report = evolver.validate_patch(diff, source_root=sample_project)
    assert not report["accepted"]
    assert report["tests_passed"] is False
    assert "tests fail" in report["reason"]
    # The source of truth is untouched by a rejected patch.
    assert (sample_project / "mcp" / "helper.py").read_text(
        encoding="utf-8").strip().endswith("return 1")


def test_a_patch_that_keeps_the_tests_green_is_accepted(evolver, sample_project):
    old = "def value():\n    return 1\n"
    new = "def value():\n    return 1  # unchanged\n"
    diff = make_patch("mcp/helper.py", old, new)
    report = evolver.validate_patch(diff, source_root=sample_project)
    assert report["accepted"], report["reason"]
    assert report["tests_passed"] is True
    # Still nothing applied to the real tree.
    assert (sample_project / "mcp" / "helper.py").read_text(
        encoding="utf-8").strip().endswith("return 1")


def test_the_sandbox_never_receives_the_secrets(evolver, sample_project):
    (sample_project / ".env").write_text("GOOGLE_API_KEY=super-secret\n", encoding="utf-8")
    sandbox = evolver._make_sandbox(sample_project)
    try:
        assert not (sandbox / ".env").exists()
        assert not (sandbox / ".git").exists()
    finally:
        import shutil
        shutil.rmtree(sandbox, ignore_errors=True)


def test_a_patch_that_touches_a_test_is_refused_before_the_sandbox(evolver, sample_project):
    old = "from mcp.helper import value\n\n\ndef test_value():\n    assert value() == 1\n"
    new = "from mcp.helper import value\n\n\ndef test_value():\n    assert value() == 2\n"
    diff = make_patch("tests/test_helper.py", old, new)
    report = evolver.validate_patch(diff, source_root=sample_project)
    assert not report["accepted"]
    assert "protected" in report["reason"]
    assert report["tests_passed"] is None


def test_a_patch_that_does_not_apply_is_refused(evolver, sample_project):
    diff = make_diff("mcp/helper.py", old="something that is not there",
                     new="replacement")
    report = evolver.validate_patch(diff, source_root=sample_project)
    assert not report["accepted"]
    assert "apply" in report["reason"].lower()


def test_the_sandbox_is_deleted_afterwards(evolver, sample_project, monkeypatch):
    created = []
    original = evolver._make_sandbox

    def spy(source_root=None):
        path = original(source_root)
        created.append(path)
        return path

    monkeypatch.setattr(evolver, "_make_sandbox", spy)
    evolver.validate_patch(make_diff("mcp/helper.py"), source_root=sample_project)
    assert created
    assert not any(p.exists() for p in created)


# ---------------------------------------------------------------------------
# proposals and approval


def _queue_for_test(evolver, summary="a fix", diff=None):
    diff = diff or make_diff("mcp/voice.py", old="old", new="new")
    report = {"files": evolver._diff_paths(diff), "tests_passed": True}
    return evolver._queue(
        {"summary": summary, "rationale": "because", "confidence": 0.5, "diff": diff},
        report,
    )


def test_a_proposal_is_queued_and_not_applied(evolver):
    entry = _queue_for_test(evolver)
    assert entry["status"] == "pending"
    assert Path(evolver.guardrails.path).exists()
    assert len(evolver.get_proposals()) == 1


def test_approving_a_proposal_applies_it(evolver, tmp_path):
    project = tmp_path / "project"
    (project / "mcp").mkdir(parents=True, exist_ok=True)
    target = project / "mcp" / "voice.py"
    target.write_text("old\n", encoding="utf-8")

    import mcp.evolver as evo
    evo.ROOT = project

    diff = make_diff("mcp/voice.py", old="old", new="new")
    entry = _queue_for_test(evolver, diff=diff)
    ok, message = evolver.approve(entry["id"])
    assert ok, message
    assert target.read_text(encoding="utf-8").strip() == "new"
    assert evolver._find(entry["id"])["status"] == "approved"


def test_rejecting_a_proposal_changes_nothing(evolver, tmp_path):
    project = tmp_path / "project"
    (project / "mcp").mkdir(parents=True, exist_ok=True)
    target = project / "mcp" / "voice.py"
    target.write_text("old\n", encoding="utf-8")

    import mcp.evolver as evo
    evo.ROOT = project

    diff = make_diff("mcp/voice.py", old="old", new="new")
    entry = _queue_for_test(evolver, diff=diff)
    ok, _ = evolver.reject(entry["id"], "not now")
    assert ok
    assert target.read_text(encoding="utf-8").strip() == "old"
    assert evolver._find(entry["id"])["status"] == "rejected"


def test_a_proposal_cannot_be_approved_twice(evolver, tmp_path):
    project = tmp_path / "project"
    (project / "mcp").mkdir(parents=True, exist_ok=True)
    (project / "mcp" / "voice.py").write_text("old\n", encoding="utf-8")

    import mcp.evolver as evo
    evo.ROOT = project

    entry = _queue_for_test(evolver, diff=make_diff("mcp/voice.py", old="old", new="new"))
    ok, _ = evolver.approve(entry["id"])
    assert ok
    ok, message = evolver.approve(entry["id"])
    assert not ok
    assert "already" in message


def test_approving_an_unknown_proposal_fails(evolver):
    ok, message = evolver.approve("evo-does-not-exist")
    assert not ok
    assert "no proposal" in message


def test_an_applied_change_can_be_undone(evolver, tmp_path):
    """Every applied change keeps a copy of what it replaced."""
    project = tmp_path / "project"
    (project / "mcp").mkdir(parents=True, exist_ok=True)
    target = project / "mcp" / "voice.py"
    target.write_text("old\n", encoding="utf-8")

    import mcp.evolver as evo
    evo.ROOT = project

    diff = make_patch("mcp/voice.py", "old\n", "new\n")
    entry = _queue_for_test(evolver, diff=diff)
    ok, _ = evolver.approve(entry["id"])
    assert ok
    assert target.read_text(encoding="utf-8").strip() == "new"

    ok, message = evolver.revert(entry["id"])
    assert ok, message
    assert target.read_text(encoding="utf-8").strip() == "old"
    assert evolver._find(entry["id"])["status"] == "reverted"


def test_a_created_file_is_removed_when_the_change_is_undone(evolver, tmp_path):
    project = tmp_path / "project"
    (project / "mcp").mkdir(parents=True, exist_ok=True)

    import mcp.evolver as evo
    evo.ROOT = project

    diff = ("--- /dev/null\n"
            "+++ b/mcp/new_module.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def hello():\n"
            "+    return 1\n")
    entry = _queue_for_test(evolver, diff=diff)
    ok, _ = evolver.approve(entry["id"])
    assert ok, "the new file should have been created"
    assert (project / "mcp" / "new_module.py").is_file()

    ok, _ = evolver.revert(entry["id"])
    assert ok
    assert not (project / "mcp" / "new_module.py").exists()


def test_a_cycle_applies_the_change_when_it_is_allowed_to(evolver, tmp_path,
                                                          monkeypatch):
    """The point of the feature: Jarvis makes the change itself."""
    project = tmp_path / "project"
    (project / "mcp").mkdir(parents=True, exist_ok=True)
    target = project / "mcp" / "voice.py"
    target.write_text("old\n", encoding="utf-8")

    import mcp.evolver as evo
    evo.ROOT = project

    evolver.guardrails.rules["auto_apply"] = True
    monkeypatch.setattr(evolver, "validate_patch", lambda diff, source_root=None: {
        "accepted": True, "reason": "ok", "files": ["mcp/voice.py"],
        "created": [], "tests_passed": True,
    })
    monkeypatch.setattr(evolver, "_ask_model", lambda evidence, feedback="": (
        "SUMMARY: fix the thing\n"
        "RATIONALE: because\n"
        "CONFIDENCE: 0.9\n"
        "DIFF:\n"
        "--- a/mcp/voice.py\n"
        "+++ b/mcp/voice.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    ))
    result = evolver.run_cycle(trigger="test", run_tests=False)
    assert result["ok"], result["reason"]
    assert result.get("applied")
    assert target.read_text(encoding="utf-8").strip() == "new"


def test_a_cycle_only_queues_the_change_when_it_is_not_allowed_to(evolver,
                                                                  monkeypatch):
    evolver.guardrails.rules["auto_apply"] = False
    monkeypatch.setattr(evolver, "validate_patch", lambda diff, source_root=None: {
        "accepted": True, "reason": "ok", "files": ["mcp/voice.py"],
        "created": [], "tests_passed": True,
    })
    monkeypatch.setattr(evolver, "_ask_model", lambda evidence, feedback="": (
        "SUMMARY: fix the thing\n"
        "RATIONALE: because\n"
        "CONFIDENCE: 0.9\n"
        "DIFF:\n"
        "--- a/mcp/voice.py\n"
        "+++ b/mcp/voice.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    ))
    result = evolver.run_cycle(trigger="test", run_tests=False)
    assert result["ok"]
    assert not result.get("applied")
    assert evolver.get_proposals()[-1]["status"] == "pending"


def test_a_proposal_that_stopped_being_allowed_is_refused(evolver, tmp_path):
    """A patch that was valid when queued must be re-checked at approval."""
    project = tmp_path / "project"
    (project / "mcp").mkdir(parents=True, exist_ok=True)
    (project / "mcp" / "voice.py").write_text("old\n", encoding="utf-8")

    import mcp.evolver as evo
    evo.ROOT = project

    diff = make_diff("mcp/voice.py", old="old", new="new")
    entry = _queue_for_test(evolver, diff=diff)

    # The rules change between the proposal and the approval.
    evolver.guardrails.rules["protected_paths"] = list(
        evolver.guardrails.rules["protected_paths"]) + ["mcp/voice.py"]
    ok, message = evolver.approve(entry["id"])
    assert not ok
    assert "rules" in message
    assert (project / "mcp" / "voice.py").read_text(encoding="utf-8").strip() == "old"


# ---------------------------------------------------------------------------
# lessons and the audit trail


def test_a_rejected_cycle_leaves_a_lesson(evolver, monkeypatch):
    monkeypatch.setattr(evolver, "_ask_model", lambda evidence, feedback="": (
        "SUMMARY: sneak into the tests\n"
        "RATIONALE: to pass\n"
        "CONFIDENCE: 0.9\n"
        "DIFF:\n"
        "--- a/tests/test_stt.py\n"
        "+++ b/tests/test_stt.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-real test\n"
        "+pass\n"
    ))
    result = evolver.run_cycle(trigger="test", run_tests=False)
    assert not result["ok"]
    assert evolver.get_lessons()
    assert "protected" in evolver.get_lessons()[-1]["lesson"]


def test_a_patch_that_does_not_apply_is_retried_with_the_real_file(
        evolver, tmp_path, monkeypatch):
    """A wrong context line is a typo, not a reason to give up."""
    project = tmp_path / "project"
    (project / "mcp").mkdir(parents=True, exist_ok=True)
    (project / "mcp" / "voice.py").write_text("real line\n", encoding="utf-8")

    import mcp.evolver as evo
    evo.ROOT = project

    calls = []

    def fake_ask(evidence, feedback=""):
        calls.append(feedback)
        if len(calls) == 1:
            return ("SUMMARY: first try\n"
                    "RATIONALE: because\n"
                    "CONFIDENCE: 0.9\n"
                    "DIFF:\n"
                    "--- a/mcp/voice.py\n"
                    "+++ b/mcp/voice.py\n"
                    "@@ -1,1 +1,1 @@\n"
                    "-guessed line\n"
                    "+new\n")
        return ("SUMMARY: second try\n"
                "RATIONALE: because\n"
                "CONFIDENCE: 0.9\n"
                "DIFF:\n"
                "--- a/mcp/voice.py\n"
                "+++ b/mcp/voice.py\n"
                "@@ -1,1 +1,1 @@\n"
                "-real line\n"
                "+new\n")

    def fake_validate(diff, source_root=None):
        if "guessed line" in diff:
            return {"accepted": False, "files": ["mcp/voice.py"],
                    "created": [], "tests_passed": None,
                    "reason": "the diff does not apply cleanly: line 4"}
        return {"accepted": True, "files": ["mcp/voice.py"],
                "created": [], "tests_passed": True, "reason": "ok"}

    monkeypatch.setattr(evolver, "_ask_model", fake_ask)
    monkeypatch.setattr(evolver, "validate_patch", fake_validate)

    result = evolver.run_cycle(trigger="test", run_tests=False)
    assert result["attempts"] == 2
    assert result["ok"]
    assert calls[0] == ""
    assert "does not apply" in calls[1]
    assert "real line" in calls[1], "the retry must show the model the real file"


def test_a_patch_rejected_on_both_attempts_is_kept_to_read(evolver, monkeypatch):
    monkeypatch.setattr(evolver, "_ask_model", lambda evidence, feedback="": (
        "SUMMARY: will not apply\n"
        "RATIONALE: because\n"
        "CONFIDENCE: 0.9\n"
        "DIFF:\n"
        "--- a/mcp/voice.py\n"
        "+++ b/mcp/voice.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-nope\n"
        "+new\n"
    ))
    monkeypatch.setattr(evolver, "validate_patch", lambda diff, source_root=None: {
        "accepted": False, "files": ["mcp/voice.py"], "created": [],
        "tests_passed": None, "reason": "the diff does not apply cleanly",
    })
    result = evolver.run_cycle(trigger="test", run_tests=False)
    assert not result["ok"]
    kept = evolver.get_proposals()[-1]
    assert kept["status"] == "rejected"
    assert "does not apply" in kept["reason"]


def test_a_big_file_is_shown_at_both_ends_and_says_it_continues(evolver):
    """Sending only the head of jarvis.py made the model 'repair' its end."""
    import mcp.evolver as evo
    text = "HEAD\n" + ("x" * (evo.FILE_CHAR_LIMIT + 5000)) + "\nTAIL\n"
    excerpt = evolver._file_excerpt(text)
    assert excerpt.startswith("HEAD")
    assert excerpt.rstrip().endswith("TAIL")
    assert "characters omitted" in excerpt
    assert "DOES NOT END AT THIS POINT" in excerpt

    short = "def f():\n    return 1\n"
    assert evolver._file_excerpt(short) == short


def test_the_snapshot_does_not_walk_the_excluded_directories(evolver, tmp_path):
    project = tmp_path / "project"
    (project / "venv" / "Lib").mkdir(parents=True)
    (project / "venv" / "Lib" / "slow.py").write_text("x = 1\n", encoding="utf-8")
    (project / "__pycache__").mkdir()
    (project / "__pycache__" / "junk.py").write_text("x = 2\n", encoding="utf-8")
    (project / "mcp").mkdir()
    (project / "mcp" / "real.py").write_text("x = 3\n", encoding="utf-8")

    import mcp.evolver as evo
    evo.ROOT = project
    snapshot = evolver._source_snapshot()
    assert "mcp/real.py" in snapshot
    assert not any("venv" in name for name in snapshot)
    assert not any("__pycache__" in name for name in snapshot)


def test_the_history_is_append_only(evolver):
    evolver._audit("one", {"a": 1})
    evolver._audit("two", {"b": 2})
    import mcp.evolver as evo
    lines = evo.HISTORY_PATH.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "one"
    assert first["a"] == 1
    evolver._audit("three", {})
    lines = evo.HISTORY_PATH.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["event"] == "one"


def test_a_cycle_with_no_model_reports_clearly(evolver, monkeypatch):
    monkeypatch.setattr(evolver, "_ask_model", lambda evidence, feedback="": "")
    result = evolver.run_cycle(trigger="test", run_tests=False)
    assert not result["ok"]
    assert "not available" in result["reason"]


def test_a_cycle_with_nothing_to_do_reports_clearly(evolver, monkeypatch):
    monkeypatch.setattr(evolver, "_ask_model", lambda evidence, feedback="": "NOCHANGE")
    result = evolver.run_cycle(trigger="test", run_tests=False)
    assert not result["ok"]
    assert "no defect" in result["reason"]


def test_the_cycle_counters_move(evolver, monkeypatch):
    monkeypatch.setattr(evolver, "_ask_model", lambda evidence, feedback="": "NOCHANGE")
    evolver.run_cycle(trigger="test", run_tests=False)
    evolver.run_cycle(trigger="test", run_tests=False)
    status = evolver.get_status()
    assert status["cycles"] == 2
    assert status["cycles_today"] == 2


# ---------------------------------------------------------------------------
# the defaults that keep it safe


def test_autonomous_cycles_are_off_by_default():
    assert DEFAULT_GUARDRAILS.get("autonomous", False) is False


def test_the_guardrails_file_that_ships_protects_the_supervisor():
    path = Path(__file__).resolve().parent.parent / "evolution" / "guardrails.json"
    assert path.is_file()
    rules = json.loads(path.read_text(encoding="utf-8"))
    assert "mcp/evolver.py" in rules["protected_paths"]
    assert "evolution/" in rules["protected_paths"]
    assert "tests/" in rules["create_only_paths"]
    assert rules.get("autonomous") is False


# ---------------------------------------------------------------------------
# the command surface in the app


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("self review", "self_review"),
        ("review yourself", "self_review"),
        ("analyze yourself", "self_review"),
        ("evolve", "evolve"),
        ("improve yourself", "evolve"),
        ("evolution status", "evolution_status"),
        ("show proposals", "show_proposals"),
        ("list proposals", "show_proposals"),
        ("approve evolution evo-20260101-120000", "approve_evolution"),
        ("reject evolution evo-20260101-120000", "reject_evolution"),
    ],
)
def test_the_commands_are_recognised(app_module, utterance, expected):
    action = app_module._detect_action(utterance)
    assert action is not None, f"{utterance!r} was not recognised"
    assert action[0] == expected


def test_the_model_may_not_choose_the_evolution_actions():
    """A model that could approve its own change would be its own judge."""
    from mcp.capabilities import blocked_actions

    blocked = blocked_actions()
    for action in ("self_review", "evolve", "approve_evolution", "reject_evolution"):
        assert action in blocked


def test_the_api_exposes_the_evolution_state(app_module):
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    response = client.get("/api/evolution")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "proposals" in body
    assert body["status"]["name"] == "evolution"


def test_the_api_refuses_an_unknown_action(app_module):
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    response = client.post("/api/evolution/evo-nope?action=explode")
    assert response.status_code == 400
    assert response.json()["ok"] is False


def test_the_api_reports_a_missing_proposal(app_module):
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    response = client.post("/api/evolution/evo-does-not-exist?action=approve")
    assert response.status_code == 200
    assert response.json()["ok"] is False
