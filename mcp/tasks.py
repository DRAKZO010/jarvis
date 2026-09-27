"""Tasks MCP - Task management (add, delete, complete, list)."""

from __future__ import annotations
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.tasks")

TASKS_PATH = Path(__file__).resolve().parent.parent / "jarvis_tasks.json"


def _ui_sort_key(task: Dict) -> tuple:
    """Sort pending tasks first, then by id.

    Ids come from a hand-editable JSON file, so they may be strings, ints, or
    missing. Comparing those directly raises TypeError, which would take down
    /api/tasks and every /health call, so coerce to a sortable number.
    """
    raw = task.get("id")
    try:
        order = float(raw)
    except (TypeError, ValueError):
        order = float("inf")  # unparseable ids sort last instead of crashing
    return (bool(task.get("done", False)), order, str(raw))


class TasksMCP(MCPBase):
    """Task management MCP module."""

    name = "tasks"
    description = "Task management (add, delete, complete, list)"
    commands = [
        "add task",
        "create task",
        "new task",
        "make task",
        "complete task",
        "done task",
        "finish task",
        "mark task",
        "delete task",
        "remove task",
        "list tasks",
        "show tasks",
        "my tasks",
        "pending tasks",
        "clear tasks",
        "clear done",
        "clear completed",
        "clear all tasks",
        "delete all tasks",
        "remove all tasks",
    ]

    def __init__(self):
        super().__init__()
        self._tasks: List[Dict] = []
        self._confirmations = {
            "added": [
                "Task added, sir.",
                "Added to your task list.",
                "Consider it done, sir.",
            ],
            "completed": [
                "Task marked as complete.",
                "Done, sir. Task completed.",
                "Another one off the list.",
            ],
            "deleted": [
                "Task removed, sir.",
                "Done. Task deleted.",
                "Consider it handled.",
            ],
            "cleared": [
                "All tasks cleared, sir.",
                "Task list wiped clean.",
                "Done. Your slate is clean.",
            ],
        }

    def initialize(self) -> bool:
        """Initialize tasks module."""
        self._tasks = self._load_tasks()
        self._initialized = True
        log.info("Tasks MCP initialized with %d tasks.", len(self._tasks))
        return True

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle task commands."""
        # Add task
        if command in ("add task", "create task", "new task", "make task"):
            if args:
                return self.add_task(args)
            return "What task should I add, sir?"

        # Complete task
        if command in ("complete task", "done task", "finish task", "mark task"):
            if args and args.isdigit():
                return self.complete_task(int(args))
            return "Please specify the task number, sir."

        # Delete task
        if command in ("delete task", "remove task"):
            if args and args.isdigit():
                return self.delete_task(int(args))
            if args:
                return self.delete_task_by_description(args)
            return "Please specify the task number or description, sir."

        # List tasks
        if command in ("list tasks", "show tasks", "my tasks", "pending tasks"):
            return self.list_tasks()

        # Clear done tasks
        if command in ("clear done", "clear completed"):
            return self.clear_done_tasks()

        # Clear all tasks
        if command in ("clear tasks", "clear all tasks", "delete all tasks", "remove all tasks"):
            return self.clear_all_tasks()

        return None

    def add_task(self, description: str, due: str = "") -> str:
        """Add a new task with collision-free ID."""
        # Next available ID. Be tolerant of hand-edited files that contain
        # non-integer or missing ids instead of crashing on startup.
        numeric_ids = [
            t["id"] for t in self._tasks
            if isinstance(t.get("id"), int)
        ]
        new_id = max(numeric_ids) + 1 if numeric_ids else 1
        while any(t.get("id") == new_id for t in self._tasks):
            new_id += 1

        task = {
            "id": new_id,
            "description": description,
            "due": due,
            "done": False,
            "created": datetime.now().isoformat(),
        }
        self._tasks.append(task)
        self._save_tasks()
        return random.choice(self._confirmations["added"])

    def complete_task(self, task_id: int) -> str:
        """Mark a task as complete."""
        for t in self._tasks:
            if t["id"] == task_id:
                t["done"] = True
                self._save_tasks()
                return random.choice(self._confirmations["completed"])
        return f"Task {task_id} not found, sir."

    def delete_task(self, task_id: int) -> str:
        """Delete a task by ID."""
        for i, t in enumerate(self._tasks):
            if t["id"] == task_id:
                self._tasks.pop(i)
                self._save_tasks()
                return random.choice(self._confirmations["deleted"])
        return f"Task {task_id} not found, sir."

    def delete_task_by_description(self, description: str) -> str:
        """Delete a task by description match."""
        desc_lower = description.lower()
        for i, t in enumerate(self._tasks):
            if desc_lower in t["description"].lower():
                self._tasks.pop(i)
                self._save_tasks()
                return random.choice(self._confirmations["deleted"])
        return f"I couldn't find a task matching '{description}', sir."

    def delete_multiple_tasks(self, ids_str: str) -> str:
        """Delete multiple tasks by IDs."""
        import re
        ids = re.findall(r"\d+", ids_str)
        if not ids:
            return "Please specify the task IDs to delete, sir."
        deleted = []
        for task_id in ids:
            tid = int(task_id)
            for i, t in enumerate(self._tasks):
                if t["id"] == tid:
                    deleted.append(self._tasks.pop(i)["description"])
                    break
        if deleted:
            self._save_tasks()
            if len(deleted) == 1:
                return f"Task deleted: {deleted[0]}."
            return f"Deleted {len(deleted)} tasks."
        return "I couldn't find those tasks, sir."

    def list_tasks(self) -> str:
        """List all pending tasks."""
        pending = [t for t in self._tasks if not t["done"]]
        if not pending:
            return "No pending tasks, sir. All clear."
        lines = []
        for t in pending[:10]:
            due = f" (due {t['due']})" if t.get("due") else ""
            lines.append(f"Task {t['id']}: {t['description']}{due}")
        return "Here are your pending tasks: " + "; ".join(lines)

    def clear_done_tasks(self) -> str:
        """Clear all completed tasks."""
        done = [t for t in self._tasks if t["done"]]
        if not done:
            return "No completed tasks to clear, sir."
        self._tasks = [t for t in self._tasks if not t["done"]]
        self._save_tasks()
        return f"Cleared {len(done)} completed task{'s' if len(done) != 1 else ''}."

    def clear_all_tasks(self) -> str:
        """Clear all tasks."""
        if not self._tasks:
            return "No tasks to clear, sir."
        count = len(self._tasks)
        self._tasks = []
        self._save_tasks()
        return f"Cleared all {count} task{'s' if count != 1 else ''}, sir."

    def get_pending_count(self) -> int:
        """Get count of pending tasks."""
        return len([t for t in self._tasks if not t["done"]])

    def get_tasks_for_ui(self) -> List[Dict[str, Any]]:
        """Get all tasks in the shape the HUD expects (pending first, then done)."""
        return [
            {
                "id": t.get("id"),
                "description": t.get("description", ""),
                "due": t.get("due", ""),
                "done": bool(t.get("done", False)),
            }
            for t in sorted(self._tasks, key=_ui_sort_key)
        ]

    def get_tasks_for_prompt(self) -> str:
        """Get tasks formatted for LLM prompt."""
        pending = [t for t in self._tasks if not t["done"]]
        if not pending:
            return "  (none)"
        return "\n".join(
            f"  - Task {t['id']}: {t['description']}"
            + (f" (due {t['due']})" if t.get("due") else "")
            for t in pending[:10]
        )

    def _load_tasks(self) -> List[Dict]:
        """Load tasks from file."""
        try:
            with open(TASKS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

    def _save_tasks(self) -> None:
        """Save tasks to file."""
        try:
            with open(TASKS_PATH, "w", encoding="utf-8") as f:
                json.dump(self._tasks, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error("Failed to save tasks: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "total": len(self._tasks),
            "pending": self.get_pending_count(),
            "done": len(self._tasks) - self.get_pending_count(),
        }
