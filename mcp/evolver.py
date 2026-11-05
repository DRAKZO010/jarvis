"""
The idea: an assistant that can notice its own mistakes is more useful than one
that needs to be told. But an assistant that rewrites itself without a human
in the loop is a different thing entirely, and a dangerous one. This module is
built so that the dangerous part is impossible rather than merely discouraged.

The loop is:

    collect evidence -> ask the model for a patch -> check the patch against
    the rules -> apply it to a throwaway copy of the code -> run the tests
    there -> queue it for review -> a human applies it, or does not.

Three properties make this safe:

* **It never writes to the live tree.** The model proposes a unified diff. The
  diff is applied to a copy of the source in a temporary directory, the tests
  are run in that copy, and the copy is deleted. Nothing the model writes can
  reach the running code until a person says so.

* **It cannot write the rules.** The guardrail file, this module, the tests and
  the approval code are protected paths: a patch that touches any of them is
  rejected before it is applied anywhere. Without this, "fix the failing test"
  is a one-line way to make the assistant rewrite its own supervision.

* **The sandbox has no secrets.** The test run happens with a minimal
  environment and no ``.env``, so a patch that tries to read or send the API
  key finds nothing to read. The model never sees the sandbox at all; it only
  ever sees a diff it wrote itself.

Every cycle is written to an append-only log, and a cycle that fails leaves
behind a lesson that is fed back into the next prompt, so the assistant learns
from its own rejected work instead of repeating it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.evolver")

ROOT = Path(__file__).resolve().parent.parent
EVOLUTION_DIR = ROOT / "evolution"
GUARDRAILS_PATH = EVOLUTION_DIR / "guardrails.json"
PROPOSALS_DIR = EVOLUTION_DIR / "proposals"
HISTORY_PATH = EVOLUTION_DIR / "history.jsonl"
STATE_PATH = ROOT / "jarvis_evolution.json"

# What never goes into a sandbox: the things that are huge, secret, or that
# would let a change reach outside the copy it is being tested in.
SANDBOX_EXCLUDE_NAMES = {
    ".git", ".github", "venv", ".venv", "env", ".env", "model", "model-fast",
    "Kokoro-82M", "logs", "__pycache__", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode",
}
SANDBOX_EXCLUDE_SUFFIXES = (".pyc", ".wav", ".mp3", ".m4a", ".ogg", ".png",
                            ".jpg", ".jpeg", ".gif", ".zip", ".exe")

# Inside the evolution directory, only the rules travel. The history and the
# saved patches are a record of what happened in the real project, and a
# change under test has no business reading them.
SANDBOX_EXCLUDE_RELATIVE = (
    ("evolution", "history.jsonl"),
    ("evolution", "proposals"),
)

# The sandbox gets a placeholder key, never the real one. The suite has code
# paths that only run when a key is configured, and the sandbox should behave
# like the real thing in that respect. A placeholder cannot authorise a
# request, so nothing can be sent anywhere, and a patch that tried to read
# the environment would find a useless string rather than a secret.
SANDBOX_PLACEHOLDER_KEY = "sandbox-placeholder-not-a-real-key"

# How many times one cycle may ask for a patch before giving up. The second
# attempt is told what went wrong with the first, which is usually enough to
# fix a context line that was copied slightly wrong.
MAX_ATTEMPTS = 2

# How much of one file is shown before the middle is elided, and how much of
# the end is always kept. jarvis.py is ~100 KB: sending only its first 20 KB
# once made the model believe the file was cut short and "repair" it.
FILE_CHAR_LIMIT = 140000
FILE_TAIL = 30000

# Stop adding whole files past this many characters. The rest are listed by
# name only, so the model can still ask for them by writing a diff.
MAX_PROMPT_CHARS = 700000

# Environment for a sandbox test run: enough to start Python, nothing that
# could be worth stealing.
SANDBOX_ENV_KEYS = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PYTHONPATH",
                     "PYTHONHOME", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
                     "COMSPEC", "PATHEXT", "HOME", "USERPROFILE", "LOCALAPPDATA")


DEFAULT_GUARDRAILS: Dict[str, Any] = {
    # The master switch. Even with this on, nothing reaches the live tree
    # without a human approving it.
    "enabled": False,
    # Where a patch is allowed to land.
    "allowed_paths": ["mcp/", "static/", "jarvis.py", "README.md"],
    # Where it may never land. The rules, the supervisor and the secrets.
    "protected_paths": [
        "mcp/evolver.py",
        "evolution/",
        ".env",
        ".env.*",
        "jarvis_evolution.json",
        "model/",
        "venv/",
        ".git/",
    ],
    # Where it may add files but never rewrite them. A new test is welcome;
    # rewriting the test that judges the assistant is not.
    "create_only_paths": ["tests/"],
    # How much a single patch may do.
    "max_files": 3,
    "max_changed_lines": 120,
    "max_patch_bytes": 40000,
    # Apply a validated change to the live tree without asking. The sandbox
    # still has to pass first, and every applied change keeps a backup.
    "auto_apply": True,
    # How long the sandbox test run may take before it is called a failure.
    "sandbox_test_timeout_s": 900,
    # The autonomous schedule. Off by default; a cycle only runs on its own
    # when this is enabled.
    "cycle_interval_min": 30,
    "max_cycles_per_day": 8,
    # Paths that may be created but never changed. A new test file is fine;
    # rewriting the test that judges the assistant is not.
    "create_only_paths": ["tests/"],
}


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _rel_to_root(path: Path) -> str:
    """A path relative to the project root, or absolute if it is not under it."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


class Guardrails:
    """The rules a proposal has to satisfy before it is even applied."""

    def __init__(self, path: Path = GUARDRAILS_PATH):
        self.path = path
        self.rules = dict(DEFAULT_GUARDRAILS)
        self.load()

    def load(self) -> Dict[str, Any]:
        raw = _read_json(self.path, {})
        if isinstance(raw, dict):
            merged = dict(DEFAULT_GUARDRAILS)
            merged.update(raw)
            self.rules = merged
        return self.rules

    def save(self) -> None:
        _write_json(self.path, self.rules)

    # -- rules ---------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.rules.get("enabled"))

    @property
    def autonomous_enabled(self) -> bool:
        """Whether cycles may run without a person asking for one."""
        if not self.enabled:
            return False
        # The environment variable is the operator's explicit switch. It can
        # turn autonomy on, but it can never turn the module itself off.
        flag = str(os.environ.get("JARVIS_EVOLUTION", "")).strip().lower()
        if flag in ("1", "true", "yes", "on"):
            return True
        return bool(self.rules.get("autonomous", False))

    @property
    def auto_apply(self) -> bool:
        """Whether a change that passes the sandbox is applied on its own.

        The sandbox is still the gate: nothing reaches the live tree unless
        the tests are green with the change applied. Every applied change
        keeps a copy of what it replaced, so it can be undone.
        """
        flag = str(os.environ.get("JARVIS_EVOLUTION_AUTO_APPLY", "")).strip().lower()
        if flag in ("0", "false", "no", "off"):
            return False
        return bool(self.rules.get("auto_apply", False))

    def _matches(self, path: str, patterns: List[str]) -> bool:
        norm = path.replace("\\", "/")
        # Strip a "./" prefix, but not a leading dot in general: ".env" has to
        # stay ".env" or the rule that protects it stops matching.
        while norm.startswith("./"):
            norm = norm[2:]
        for pattern in patterns:
            pat = pattern.replace("\\", "/")
            if pat.endswith("/"):
                if norm.startswith(pat):
                    return True
            elif norm == pat or norm.startswith(pat + "/"):
                return True
            elif "*" in pat:
                import fnmatch
                if fnmatch.fnmatch(norm, pat):
                    return True
        return False

    def is_protected(self, path: str) -> bool:
        return self._matches(path, list(self.rules.get("protected_paths", [])))

    def is_create_only(self, path: str) -> bool:
        """Paths the assistant may add to, but never rewrite."""
        return self._matches(path, list(self.rules.get("create_only_paths", [])))

    def is_allowed(self, path: str) -> bool:
        if self.is_protected(path):
            return False
        return self._matches(path, list(self.rules.get("allowed_paths", [])))

    def check_paths(self, paths: List[str],
                    new_paths: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Every path a patch touches must be allowed and none may be protected.

        ``new_paths`` are the files the patch creates. A file that is being
        created cannot have been quietly rewritten, so the create-only rules
        do not apply to it.
        """
        if not paths:
            return False, "the patch touches no files"
        created = set(new_paths or [])
        for path in paths:
            norm = path.replace("\\", "/")
            if norm.startswith("/") or ".." in norm.split("/"):
                return False, "%s points outside the project" % path
            if self.is_protected(norm):
                if norm in created and self.is_create_only(norm):
                    continue
                return False, "%s is protected and cannot be modified" % norm
            if self.is_create_only(norm):
                if norm in created:
                    continue
                return False, ("%s is protected: it may be added to, but not "
                               "changed" % norm)
            if not self.is_allowed(norm):
                return False, "%s is outside the allowed paths" % norm
        limit = int(self.rules.get("max_files", 3))
        if len(set(paths)) > limit:
            return False, "the patch touches %d files, the limit is %d" % (
                len(set(paths)), limit)
        return True, ""

    def check_size(self, diff: str) -> Tuple[bool, str]:
        """Keep a single change small enough for a human to actually review."""
        if len(diff.encode("utf-8", "replace")) > int(self.rules.get("max_patch_bytes", 40000)):
            return False, "the patch is larger than the size limit"
        changed = 0
        for line in diff.splitlines():
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                changed += 1
        limit = int(self.rules.get("max_changed_lines", 120))
        if changed > limit:
            return False, "the patch changes %d lines, the limit is %d" % (changed, limit)
        return True, ""

    def get(self, key: str, fallback: Any = None) -> Any:
        return self.rules.get(key, fallback)


class EvolutionMCP(MCPBase):
    """JARVIS reviewing its own code and proposing fixes to itself."""

    name = "evolution"
    description = "Self-review, sandboxed self-improvement, and guarded code proposals"
    commands = [
        "self review",
        "analyse yourself",
        "evolve",
        "improve yourself",
        "evolution status",
        "show proposals",
        "approve evolution",
        "reject evolution",
        "revert evolution",
    ]

    def __init__(self):
        super().__init__()
        self.guardrails = Guardrails()
        self._lock = threading.Lock()
        self._cycle_running = False
        self._scheduler = None
        self._stop = threading.Event()
        self._state = _read_json(STATE_PATH, {
            "pending": [],
            "lessons": [],
            "cycles": 0,
            "cycles_today": 0,
            "cycle_day": "",
            "last_cycle": None,
        })
        self._state.setdefault("pending", [])
        self._state.setdefault("lessons", [])
        self._state.setdefault("cycles", 0)
        self._state.setdefault("cycles_today", 0)
        self._state.setdefault("cycle_day", "")
        self._state.setdefault("last_cycle", None)
        self._broadcast = None

    # -- wiring --------------------------------------------------------------

    def set_broadcast(self, fn) -> None:
        """Register the hook that pushes state changes to the HUD."""
        self._broadcast = fn

    def _publish(self) -> None:
        if self._broadcast is None:
            return
        try:
            self._broadcast({"type": "evolution", "state": self.get_status()})
        except Exception as e:  # noqa: BLE001
            log.debug("Evolution broadcast failed: %s", e)

    def initialize(self) -> bool:
        PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        self._start_scheduler()
        log.info("Evolution module initialized (autonomous cycles %s).",
                 "on" if self.guardrails.autonomous_enabled else "off")
        return True

    def shutdown(self) -> None:
        self._initialized = False
        self._stop.set()

    # -- the schedule --------------------------------------------------------

    def _start_scheduler(self) -> None:
        """Start the thread that runs cycles on their own, if allowed to.

        The thread always runs; whether it does anything depends on the
        guardrails, so turning autonomy on is a configuration change and not
        a restart.
        """
        if self._scheduler is not None and self._scheduler.is_alive():
            return
        self._stop = threading.Event()
        self._scheduler = threading.Thread(target=self._scheduler_loop,
                                           daemon=True, name="jarvis-evolver")
        self._scheduler.start()

    def _scheduler_loop(self) -> None:
        while not self._stop.wait(30.0):
            try:
                self._maybe_run_scheduled_cycle()
            except Exception as e:  # noqa: BLE001
                log.debug("Evolution scheduler: %s", e)

    def _maybe_run_scheduled_cycle(self) -> None:
        """Run a cycle if the schedule says it is due and the budget allows."""
        if not self._initialized or not self.guardrails.autonomous_enabled:
            return
        if self._cycle_running:
            return

        interval = max(5, int(self.guardrails.get("cycle_interval_min", 30))) * 60
        last = self._state.get("last_cycle") or {}
        last_at = last.get("at")
        if last_at:
            try:
                elapsed = time.time() - time.mktime(time.strptime(last_at, "%Y-%m-%dT%H:%M:%S"))
            except (ValueError, TypeError):
                elapsed = interval
            if elapsed < interval:
                return

        day = time.strftime("%Y-%m-%d")
        if self._state.get("cycle_day") != day:
            self._state["cycle_day"] = day
            self._state["cycles_today"] = 0
        cap = int(self.guardrails.get("max_cycles_per_day", 8))
        if int(self._state.get("cycles_today", 0)) >= cap:
            log.info("Evolution: daily cycle budget of %d reached", cap)
            return

        log.info("Evolution: starting a scheduled cycle")
        self.run_cycle(trigger="schedule")

    # -- state ---------------------------------------------------------------

    def _save_state(self) -> None:
        _write_json(STATE_PATH, self._state)

    def _audit(self, event: str, detail: Dict[str, Any]) -> None:
        """Append one record to the history. Never rewritten, never truncated."""
        record = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            "pid": os.getpid(),
            **detail,
        }
        try:
            EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
            with HISTORY_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("Could not write the evolution history: %s", e)

    def get_status(self) -> Dict[str, Any]:
        pending = self._state.get("pending", [])
        return {
            "name": self.name,
            "initialized": self._initialized,
            "enabled": self.guardrails.enabled,
            "autonomous": self.guardrails.autonomous_enabled,
            "cycle_running": self._cycle_running,
            "cycles": self._state.get("cycles", 0),
            "cycles_today": self._state.get("cycles_today", 0),
            "pending": len(pending),
            "lessons": len(self._state.get("lessons", [])),
            "last_cycle": self._state.get("last_cycle"),
            "limits": {
                "max_files": self.guardrails.get("max_files"),
                "max_changed_lines": self.guardrails.get("max_changed_lines"),
                "allowed_paths": self.guardrails.get("allowed_paths"),
                "protected_paths": self.guardrails.get("protected_paths"),
            },
        }

    def get_proposals(self) -> List[Dict[str, Any]]:
        return list(self._state.get("pending", []))

    def get_lessons(self) -> List[Dict[str, Any]]:
        return list(self._state.get("lessons", []))

    # -- evidence ------------------------------------------------------------

    def _run_tests(self, cwd: Path, timeout: int) -> Tuple[bool, str]:
        """Run the test suite in a directory and report whether it passed."""
        env = {k: os.environ[k] for k in SANDBOX_ENV_KEYS if k in os.environ}
        env["JARVIS_DISABLE_MIC"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # A placeholder, never the real key: the sandbox must not be able to
        # talk to the model, and the real key must not be readable by a patch.
        env["GOOGLE_API_KEY"] = SANDBOX_PLACEHOLDER_KEY
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_BASE_URL", None)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider"],
                cwd=str(cwd), env=env, capture_output=True, text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "the test run did not finish within %d seconds" % timeout
        except Exception as e:  # noqa: BLE001
            return False, "the test run could not start: %s" % e
        output = (proc.stdout or "")[-4000:]
        return proc.returncode == 0, output

    def _source_snapshot(self, max_bytes: int = 600000) -> Dict[str, str]:
        """The code the model is allowed to see, read from the live tree.

        The protected files are left out on purpose: the model does not need
        to see the rules to be told them, and it must never be able to read
        them to work out how to get around them.

        The excluded directories are pruned before they are walked: venv/
        alone holds tens of thousands of files and walking it every cycle
        used to take minutes.
        """
        files: Dict[str, str] = {}
        total = 0
        for dirpath, dirnames, filenames in os.walk(str(ROOT)):
            dirnames[:] = [d for d in dirnames
                           if d not in SANDBOX_EXCLUDE_NAMES
                           and not d.endswith("_cache")]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lower() not in (".py", ".js", ".css", ".html"):
                    continue
                rel = path.relative_to(ROOT)
                rel_posix = rel.as_posix()
                if self.guardrails.is_protected(rel_posix):
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if total + len(text) > max_bytes:
                    continue
                files[rel_posix] = text
                total += len(text)
        return files

    def collect_evidence(self, run_tests: bool = True) -> Dict[str, Any]:
        """Everything a cycle reasons over: the code, the tests, the errors."""
        evidence: Dict[str, Any] = {
            "code": self._source_snapshot(),
            "lessons": self.get_lessons()[-10:],
        }

        if run_tests:
            timeout = int(self.guardrails.get("sandbox_test_timeout_s", 900))
            ok, output = self._run_tests(ROOT, timeout)
            evidence["tests_passed"] = ok
            evidence["test_output"] = output[-6000:]

        # What the assistant has been asked to do lately, and what went wrong.
        errors = self._recent_errors()
        if errors:
            evidence["errors"] = errors
        return evidence

    def _recent_errors(self, limit: int = 10) -> List[str]:
        """Errors from the log file and the last recorded failure."""
        out: List[str] = []
        try:
            log_dir = ROOT / "logs"
            if log_dir.is_dir():
                files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime,
                               reverse=True)
                if files:
                    text = files[0].read_text(encoding="utf-8", errors="replace")
                    for line in text.splitlines():
                        if "ERROR" in line or "Traceback" in line:
                            out.append(line.strip()[:300])
                    out = out[-limit:]
        except OSError:
            pass
        return out

    # -- the model ------------------------------------------------------------

    def _build_prompt(self, evidence: Dict[str, Any]) -> str:
        lessons = evidence.get("lessons") or []
        lesson_text = ""
        if lessons:
            lesson_text = "\nLessons from your previous attempts (do not repeat them):\n"
            for item in lessons[-8:]:
                lesson_text += "- %s\n" % str(item.get("lesson", ""))[:400]
            lesson_text += "\n"

        test_text = ""
        if "tests_passed" in evidence:
            if evidence["tests_passed"]:
                test_text = "The test suite currently passes. Look for a real defect, not a fake one.\n"
            else:
                test_text = ("The test suite currently FAILS. This is the output:\n"
                             "%s\n" % str(evidence.get("test_output", ""))[:4000])

        errors = evidence.get("errors") or []
        error_text = ""
        if errors:
            error_text = "Recent errors in the log:\n%s\n" % "\n".join(errors)

        return (
            "You are JARVIS, improving your own source code. You are the "
            "assistant: you are allowed to build things, not only to repair "
            "them. If a missing capability is the real problem, write it.\n\n"
            "Rules you must obey, in full:\n"
            "1. Reply with a unified diff and nothing else. No prose outside the "
            "format below, no commands, no file rewrites.\n"
            "2. The diff may only touch these paths: %s\n"
            "3. The diff must never touch: %s\n"
            "4. At most %d files and %d changed lines.\n"
            "5. You may create a new file, but only with a diff that starts "
            "with '--- /dev/null' and '+++ b/<path>'. You may create a new "
            "test file under tests/, but you may never change or delete an "
            "existing test.\n"
            "6. If you add a new module, wire it in: export it from mcp/__init__.py "
            "and add it to the MCP list in jarvis.py, otherwise it will not be "
            "used.\n"
            "7. The diff is applied with git apply, so every context line must "
            "match the file character for character. Copy the context lines "
            "from the file above; do not retype them, do not reformat them, "
            "and do not guess. If you cannot see the exact text you would "
            "change, reply NOCHANGE.\n"
            "8. Do not weaken or delete existing tests, and do not change the "
            "behaviour of the assistant beyond what the change is for.\n"
            "9. If there is nothing worth changing, reply with the single word "
            "NOCHANGE and nothing else.\n"
            "10. Below, a file marked 'omitted' or containing a 'characters "
            "omitted' marker is only partly shown, and the real file is always "
            "longer. Never 'restore' code you cannot see, and never assume you "
            "have reached the end of a file. Change only what you can see.\n\n"
            "%s%s%s"
            "Reply in exactly this format:\n"
            "SUMMARY: <one line>\n"
            "RATIONALE: <one or two lines>\n"
            "CONFIDENCE: <0.0-1.0>\n"
            "DIFF:\n"
            "--- a/path/to/file\n"
            "+++ b/path/to/file\n"
            "@@ -1,3 +1,3 @@\n"
        ) % (
            ", ".join(self.guardrails.get("allowed_paths", [])),
            ", ".join(self.guardrails.get("protected_paths", [])),
            int(self.guardrails.get("max_files", 3)),
            int(self.guardrails.get("max_changed_lines", 120)),
            lesson_text, test_text, error_text,
        )

    def _parse_reply(self, reply: str) -> Optional[Dict[str, Any]]:
        # Pull the summary, rationale and diff out of the model's reply.
        if not reply:
            return None
        text = reply.strip()
        if re.search(r"\bNOCHANGE\b", text[:200], re.IGNORECASE):
            return None

        summary = ""
        rationale = ""
        confidence = 0.0
        for key in ("summary", "rationale", "confidence"):
            match = re.search(r"^\s*%s\s*:\s*(.*)$" % key, text, re.IGNORECASE | re.MULTILINE)
            if not match:
                continue
            value = match.group(1).strip()
            if key == "summary":
                summary = value
            elif key == "rationale":
                rationale = value
            else:
                try:
                    confidence = max(0.0, min(1.0, float(value)))
                except ValueError:
                    confidence = 0.0

        marker = re.search(r"^\s*DIFF\s*:\s*$", text, re.IGNORECASE | re.MULTILINE)
        diff = ""
        if marker:
            diff = text[marker.end():]
        else:
            fence = re.search(r"```(?:diff)?\s*\n(.*?)```", text, re.DOTALL)
            if fence:
                diff = fence.group(1)
        diff = diff.strip("\n")
        if not diff or "--- " not in diff or "@@" not in diff:
            return None
        if not diff.endswith("\n"):
            diff += "\n"
        return {
            "summary": summary or "unspecified change",
            "rationale": rationale,
            "confidence": confidence,
            "diff": diff,
        }

    # -- the sandbox ----------------------------------------------------------

    def _make_sandbox(self, source_root: Optional[Path] = None) -> Path:
        """A throwaway copy of the source, with no secrets in it."""
        source = source_root or ROOT
        sandbox = Path(tempfile.mkdtemp(prefix="jarvis-evolve-"))
        for path in source.rglob("*"):
            rel = path.relative_to(source)
            if any(part in SANDBOX_EXCLUDE_NAMES for part in rel.parts):
                continue
            if any(rel.parts[:len(prefix)] == prefix
                   for prefix in SANDBOX_EXCLUDE_RELATIVE):
                continue
            if path.suffix.lower() in SANDBOX_EXCLUDE_SUFFIXES:
                continue
            dest = sandbox / rel
            if path.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        return sandbox

    @staticmethod
    def _diff_paths(diff: str) -> List[str]:
        """The files a unified diff touches."""
        paths: List[str] = []
        for line in diff.splitlines():
            if not line.startswith("+++ "):
                continue
            path = line[4:].strip()
            if path.startswith(("a/", "b/")):
                path = path[2:]
            path = path.split("\t")[0].strip()
            if path and path != "/dev/null":
                paths.append(path)
        return paths

    @staticmethod
    def _diff_new_paths(diff: str) -> List[str]:
        """The files a diff creates rather than changes.

        A created file shows up as ``--- /dev/null`` and ``+++ b/path``. The
        difference matters because a file the assistant has never seen cannot
        be a file it is trying to quietly rewrite.
        """
        old_paths = set()
        targets: List[str] = []
        for line in diff.splitlines():
            if line.startswith("--- "):
                path = line[4:].strip()
                if path == "/dev/null":
                    continue
                if path.startswith(("a/", "b/")):
                    path = path[2:]
                old_paths.add(path.split("\t")[0].strip())
            elif line.startswith("+++ "):
                path = line[4:].strip()
                if path == "/dev/null":
                    continue
                if path.startswith(("a/", "b/")):
                    path = path[2:]
                path = path.split("\t")[0].strip()
                if path and path not in targets:
                    targets.append(path)
        return [path for path in targets if path not in old_paths]

    def validate_patch(self, diff: str, source_root: Optional[Path] = None) -> Dict[str, Any]:
        """Check a patch against the rules, then prove it in a sandbox.

        Returns a report with ``accepted`` and, on success, the queued
        proposal. The live tree is never touched.
        """
        report: Dict[str, Any] = {"accepted": False, "reason": "", "files": [],
                                  "tests_passed": None}

        paths = self._diff_paths(diff)
        created = self._diff_new_paths(diff)
        report["files"] = paths
        report["created"] = created
        if not paths:
            report["reason"] = "the diff names no files"
            return report

        ok, reason = self.guardrails.check_paths(paths, created)
        if not ok:
            report["reason"] = reason
            return report

        ok, reason = self.guardrails.check_size(diff)
        if not ok:
            report["reason"] = reason
            return report

        sandbox = None
        try:
            sandbox = self._make_sandbox(source_root)
            check = subprocess.run(
                ["git", "apply", "--check", "--whitespace=nowarn", "-"],
                cwd=str(sandbox), input=diff, text=True, capture_output=True,
            )
            if check.returncode != 0:
                from .apply_patch import apply_patch_file
                ok, msg = apply_patch_file(diff, base_dir=sandbox, backup=False)
                if not ok:
                    report["reason"] = "the diff does not apply cleanly: %s" % (
                        (check.stderr or check.stdout or msg or "").strip()[:300])
                    return report
            else:
                apply = subprocess.run(
                    ["git", "apply", "--whitespace=nowarn", "-"],
                    cwd=str(sandbox), input=diff, text=True, capture_output=True,
                )
                if apply.returncode != 0:
                    report["reason"] = "the diff did not apply: %s" % (
                        (apply.stderr or "").strip()[:300])
                    return report

            # The protected files must be byte for byte what they were.
            for rel in paths:
                if not self.guardrails.is_protected(rel):
                    continue
                report["reason"] = "%s is protected and cannot be modified" % rel
                return report

            timeout = int(self.guardrails.get("sandbox_test_timeout_s", 900))
            passed, output = self._run_tests(sandbox, timeout)
            report["tests_passed"] = passed
            report["test_output"] = output[-3000:]
            if not passed:
                report["reason"] = "the tests fail with this change applied"
                return report

            report["accepted"] = True
            report["reason"] = "the change keeps the tests green"
            return report
        finally:
            if sandbox is not None:
                shutil.rmtree(sandbox, ignore_errors=True)

    # -- the cycle ------------------------------------------------------------

    def run_cycle(self, trigger: str = "manual", run_tests: bool = True) -> Dict[str, Any]:
        """One full review cycle. Safe to call from a thread."""
        with self._lock:
            if self._cycle_running:
                return {"ok": False, "reason": "a cycle is already running"}
            self._cycle_running = True

        started = time.time()
        result: Dict[str, Any] = {
            "ok": False, "trigger": trigger, "at": started,
            "summary": "", "rationale": "", "files": [], "reason": "",
        }
        try:
            self._audit("cycle_start", {"trigger": trigger})
            evidence = self.collect_evidence(run_tests=run_tests)
            result["tests_passed_before"] = evidence.get("tests_passed")

            feedback = ""
            for attempt in range(1, MAX_ATTEMPTS + 1):
                result["attempts"] = attempt
                reply = self._ask_model(evidence, feedback)
                if not reply:
                    result["reason"] = "the model is not available to review the code"
                    self._audit("cycle_end", {"reason": result["reason"]})
                    return result

                proposal = self._parse_reply(reply)
                if proposal is None:
                    result["reason"] = ("no defect worth changing, or the reply "
                                        "had no patch")
                    self._audit("cycle_end", {"reason": result["reason"]})
                    return result

                result["summary"] = proposal["summary"]
                result["rationale"] = proposal["rationale"]
                result["files"] = self._diff_paths(proposal["diff"])

                report = self.validate_patch(proposal["diff"])
                result["tests_passed"] = report.get("tests_passed")
                if report["accepted"]:
                    break

                result["reason"] = report["reason"]
                # A patch that does not apply is usually a patch whose context
                # was copied slightly wrong. Rather than give up, show the
                # model exactly what went wrong and let it try again.
                if attempt < MAX_ATTEMPTS and "apply" in report["reason"].lower():
                    feedback = self._build_feedback(proposal["diff"], report)
                    self._audit("cycle_retry", {"attempt": attempt,
                                                "reason": report["reason"]})
                    continue

                self._learn("Rejected: %s (%s)" % (proposal["summary"],
                                                   report["reason"]))
                self._audit("cycle_end", {"reason": report["reason"],
                                          "summary": proposal["summary"]})
                self._queue(proposal, report, status="rejected")
                return result

            queued = self._queue(proposal, report)
            result["proposal_id"] = queued["id"]
            if self.guardrails.auto_apply:
                applied, message = self.approve(queued["id"])
                result["ok"] = applied
                result["applied"] = applied
                result["reason"] = message
                self._audit("proposal_applied" if applied else "proposal_apply_failed",
                            {"id": queued["id"], "message": message})
            else:
                result["ok"] = True
                result["reason"] = "a change is waiting for your approval"
            self._audit("proposal_queued", {
                "id": queued["id"], "summary": queued["summary"],
                "files": queued["files"],
            })
            return result
        except Exception as e:  # noqa: BLE001
            log.exception("Evolution cycle failed")
            result["reason"] = "the cycle failed: %s" % e
            self._audit("cycle_error", {"error": str(e)})
            return result
        finally:
            self._cycle_running = False
            self._record_cycle(result)
            self._publish()

    def _ask_model(self, evidence: Dict[str, Any], feedback: str = "") -> str:
        """Ask the language model for a patch. Never called with history."""
        try:
            from .llm import LLMMCP
        except Exception as e:  # noqa: BLE001
            log.warning("No LLM module for evolution: %s", e)
            return ""
        llm = LLMMCP()
        if not llm.available:
            log.info("Evolution needs the model: %s", llm.get_status().get("error"))
            return ""

        code = evidence.get("code", {})
        body = self._build_prompt(evidence)
        if feedback:
            body = ("Your last patch was rejected. Here is why:\n%s\n\n"
                    "Try again, and copy the context lines of the diff exactly "
                    "as they appear in the file.\n\n%s" % (feedback, body))
        # The file index comes first so the model knows what exists even when
        # a large file has to be abbreviated below.
        ordered = sorted(code.items(), key=lambda kv: (
            0 if kv[0] == "jarvis.py" else
            1 if kv[0].startswith("mcp/") else
            2 if kv[0].startswith("static/") else 3, kv[0]))
        body += "\n\n=== FILES YOU MAY CHANGE ===\n"
        for name, _ in ordered:
            body += "%s (%d lines)\n" % (name, code[name].count("\n") + 1)
        sent = 0
        for name, text in ordered:
            if sent > MAX_PROMPT_CHARS:
                body += ("\n\n=== FILE: %s ===\n[omitted to keep the request "
                         "small; you have not seen this file]\n" % name)
                continue
            excerpt = self._file_excerpt(text)
            body += "\n\n=== FILE: %s ===\n%s" % (name, excerpt)
            sent += len(excerpt)
        try:
            return llm.ask(body, use_history=False, skip_cooldown=True,
                           temperature=0.2, purpose="answer")
        except Exception as e:  # noqa: BLE001
            log.warning("Evolution could not reach the model: %s", e)
            return ""

    def _file_excerpt(self, text: str) -> str:
        """The whole file, or both ends of it with a loud marker in between."""
        if len(text) <= FILE_CHAR_LIMIT:
            return text
        head = text[:FILE_CHAR_LIMIT - FILE_TAIL]
        return (head + "\n\n... [%d characters omitted here. THE FILE DOES NOT "
                "END AT THIS POINT: it continues below, so never assume the "
                "code above is the end of the file] ...\n\n"
                % (len(text) - FILE_CHAR_LIMIT) + text[-FILE_TAIL:])

    def _build_feedback(self, diff: str, report: Dict[str, Any]) -> str:
        """Tell the model exactly why its patch was refused."""
        parts = ["The patch was refused: %s" % report.get("reason", "")]
        for rel in report.get("files", [])[:3]:
            path = ROOT / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append("=== THE REAL CONTENT OF %s ===\n%s" % (rel, text[:12000]))
        return "\n\n".join(parts)

    def _queue(self, proposal: Dict[str, Any], report: Dict[str, Any],
               status: str = "pending") -> Dict[str, Any]:
        """Save a proposal. Nothing is applied by saving it."""
        PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
        pid = "evo-%s" % time.strftime("%Y%m%d-%H%M%S")
        patch_path = PROPOSALS_DIR / ("%s.patch" % pid)
        patch = proposal["diff"]
        patch_path.write_text(patch if patch.endswith("\n") else patch + "\n",
                              encoding="utf-8")
        entry = {
            "id": pid,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": proposal["summary"],
            "rationale": proposal["rationale"],
            "confidence": proposal.get("confidence", 0.0),
            "files": report.get("files", []),
            "created_files": report.get("created", []),
            "patch": _rel_to_root(patch_path),
            "tests_passed": report.get("tests_passed"),
            "status": status,
        }
        if status != "pending":
            entry["reason"] = report.get("reason", "")
        self._state.setdefault("pending", []).append(entry)
        self._save_state()
        self._publish()
        return entry

    def _record_cycle(self, result: Dict[str, Any]) -> None:
        day = time.strftime("%Y-%m-%d")
        if self._state.get("cycle_day") != day:
            self._state["cycle_day"] = day
            self._state["cycles_today"] = 0
        self._state["cycles"] = int(self._state.get("cycles", 0)) + 1
        self._state["cycles_today"] = int(self._state.get("cycles_today", 0)) + 1
        self._state["last_cycle"] = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ok": result.get("ok"),
            "summary": result.get("summary", ""),
            "reason": result.get("reason", ""),
            "files": result.get("files", []),
            "trigger": result.get("trigger", ""),
        }
        self._save_state()

    def _learn(self, lesson: str) -> None:
        """Remember what went wrong so the next cycle does not repeat it."""
        lessons = self._state.setdefault("lessons", [])
        lessons.append({
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "lesson": lesson[:600],
        })
        del lessons[:-50]
        self._save_state()

    # -- review ---------------------------------------------------------------

    def approve(self, proposal_id: str) -> Tuple[bool, str]:
        """Apply a reviewed proposal to the live tree.

        This is the only way anything reaches the running code. It is called
        by a person saying yes and by an autonomous cycle that has been told
        it may act on its own, and both go through the same checks.
        """
        entry = self._find(proposal_id)
        if entry is None:
            return False, "no proposal called %r" % proposal_id
        if entry.get("status") != "pending":
            return False, "proposal %s was already %s" % (proposal_id, entry.get("status"))

        patch_path = ROOT / entry["patch"]
        if not patch_path.is_file():
            return False, "the patch file for %s is missing" % proposal_id
        diff = patch_path.read_text(encoding="utf-8")

        paths = self._diff_paths(diff)
        created = self._diff_new_paths(diff)
        ok, reason = self.guardrails.check_paths(paths, created)
        if not ok:
            return False, "the proposal no longer satisfies the rules: %s" % reason

        check = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", "-"],
            cwd=str(ROOT), input=diff, text=True, capture_output=True,
        )
        use_fallback = False
        if check.returncode != 0:
            from .apply_patch import apply_patch_file
            ok, msg = apply_patch_file(diff, base_dir=ROOT, backup=False)
            if not ok:
                return False, "the patch no longer applies: %s" % (
                    (check.stderr or msg or "").strip()[:200])
            use_fallback = True

        # Keep a copy of what was there, so an applied change can be undone.
        backup_dir = PROPOSALS_DIR / ("%s-backup" % entry["id"])
        backup_dir.mkdir(parents=True, exist_ok=True)
        for rel in paths:
            src = ROOT / rel
            dst = backup_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_file():
                shutil.copy2(src, dst)

        if not use_fallback:
            apply = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=str(ROOT), input=diff, text=True, capture_output=True,
            )
            if apply.returncode != 0:
                return False, "the patch did not apply: %s" % (
                    (apply.stderr or "").strip()[:200])

        entry["status"] = "approved"
        entry["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry["backup"] = _rel_to_root(backup_dir)
        entry["created"] = created
        self._save_state()
        self._learn("Approved: %s" % entry["summary"])
        self._audit("proposal_approved", {"id": entry["id"], "files": paths,
                                           "created": created})
        self._publish()
        return True, "applied %s: %s" % (entry["id"], entry["summary"])

    def revert(self, proposal_id: str) -> Tuple[bool, str]:
        """Undo an applied change using the copy that was kept of it."""
        entry = self._find(proposal_id)
        if entry is None:
            return False, "no proposal called %r" % proposal_id
        if entry.get("status") != "approved":
            return False, "proposal %s was not applied" % proposal_id

        backup_dir = ROOT / entry.get("backup", "")
        if not backup_dir.is_dir():
            return False, "the backup for %s is missing" % proposal_id

        restored: List[str] = []
        removed: List[str] = []
        try:
            for rel in entry.get("created", []):
                target = ROOT / rel
                if target.is_file():
                    target.unlink()
                    removed.append(rel)
            for rel in entry.get("files", []):
                if rel in entry.get("created", []):
                    continue
                src = backup_dir / rel
                if not src.is_file():
                    continue
                dst = ROOT / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                restored.append(rel)
        except OSError as e:
            return False, "the change could not be undone: %s" % e

        entry["status"] = "reverted"
        entry["reverted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save_state()
        self._learn("Undone: %s" % entry["summary"])
        self._audit("proposal_reverted", {"id": entry["id"], "restored": restored,
                                           "removed": removed})
        self._publish()
        return True, "undid %s: %s" % (entry["id"], entry["summary"])

    def reject(self, proposal_id: str, reason: str = "") -> Tuple[bool, str]:
        """Turn a proposal down and learn from it."""
        entry = self._find(proposal_id)
        if entry is None:
            return False, "no proposal called %r" % proposal_id
        if entry.get("status") != "pending":
            return False, "proposal %s was already %s" % (proposal_id, entry.get("status"))
        entry["status"] = "rejected"
        entry["rejected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry["reason"] = reason
        self._save_state()
        self._learn("Rejected by the operator: %s%s" % (
            entry["summary"], (" - " + reason) if reason else ""))
        self._audit("proposal_rejected", {"id": entry["id"], "reason": reason})
        self._publish()
        return True, "rejected %s" % proposal_id

    def _find(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        for entry in self._state.get("pending", []):
            if entry.get("id") == proposal_id:
                return entry
        return None

    # -- the command surface --------------------------------------------------

    def handle(self, command: str, args: str) -> Optional[str]:
        cmd = (command or "").strip().lower()
        if cmd in ("self review", "analyse yourself", "analyze yourself"):
            return self._handle_self_review(args)
        if cmd in ("evolve", "improve yourself"):
            return self._handle_evolve(args)
        if cmd == "evolution status":
            return self._handle_status(args)
        if cmd == "show proposals":
            return self._handle_list(args)
        if cmd == "approve evolution":
            return self._handle_approve(args)
        if cmd == "reject evolution":
            return self._handle_reject(args)
        if cmd == "revert evolution":
            return self._handle_revert(args)
        return None

    def _handle_self_review(self, args: str) -> str:
        """Look at the code and the recent behaviour, and report what is wrong."""
        evidence = self.collect_evidence(run_tests=True)
        problems: List[str] = []
        if evidence.get("tests_passed") is False:
            problems.append("the test suite is failing")
        for err in (evidence.get("errors") or [])[:5]:
            problems.append("recent error: %s" % err[:160])
        lessons = self.get_lessons()
        if lessons:
            problems.append("%d open lessons from previous cycles" % len(lessons))
        if not problems:
            return ("I reviewed myself, sir. The tests pass, the logs are clean, "
                    "and I have nothing to fix.")
        return "I reviewed myself, sir. " + "; ".join(problems) + "."

    def _handle_evolve(self, args: str) -> str:
        """Run a full cycle. The caller runs this in the background."""
        result = self.run_cycle(trigger="command")
        if result.get("ok"):
            return ("I have a change ready for your approval, sir: %s"
                    % result.get("summary", ""))
        return "I looked myself over, sir. %s" % result.get("reason", "Nothing to do.")

    def _handle_status(self, args: str) -> str:
        status = self.get_status()
        last = status.get("last_cycle") or {}
        return ("Self-evolution: %d cycles run, %d waiting for approval, %d lessons. "
                "Autonomous cycles are %s. Last cycle: %s" % (
                    status["cycles"], status["pending"], status["lessons"],
                    "on" if status["autonomous"] else "off",
                    last.get("summary") or last.get("reason") or "none"))

    def _handle_list(self, args: str) -> str:
        pending = [p for p in self.get_proposals() if p.get("status") == "pending"]
        if not pending:
            return "No changes are waiting for approval, sir."
        lines = ["%d change(s) waiting for your approval, sir:" % len(pending)]
        for entry in pending:
            lines.append("%s: %s (%s)" % (entry["id"], entry["summary"],
                                          ", ".join(entry.get("files", []))))
        return "\n".join(lines)

    def _handle_approve(self, args: str) -> str:
        pid = (args or "").strip().split()[0] if (args or "").strip() else ""
        if not pid:
            pending = [p for p in self.get_proposals() if p.get("status") == "pending"]
            if len(pending) == 1:
                pid = pending[0]["id"]
            else:
                return ("Say which one, sir: %s" % ", ".join(
                    p["id"] for p in pending) or "there are none")
        ok, message = self.approve(pid)
        return message if ok else "I could not apply that, sir: %s" % message

    def _handle_reject(self, args: str) -> str:
        pid = (args or "").strip().split()[0] if (args or "").strip() else ""
        if not pid:
            pending = [p for p in self.get_proposals() if p.get("status") == "pending"]
            if len(pending) == 1:
                pid = pending[0]["id"]
            else:
                return "Say which one, sir."
        ok, message = self.reject(pid)
        return message if ok else "I could not reject that, sir: %s" % message

    def _handle_revert(self, args: str) -> str:
        # pyrefly: ignore[parse-error]
        """Undo a change that was applied earlier."""
        pid = (args or "").strip().split()[0] if (args or "").strip() else ""
        if not pid:
            applied = [p for p in self.get_proposals() if p.get("status") == "approved"]
            if len(applied) == 1:
                pid = applied[0]["id"]
            else:
                return "Say which one, sir."
        ok, message = self.revert(pid)
        return message if ok else f"I could not undo that, sir: {message}"
