"""Reminders MCP - Set alarms, timers, and reminders."""

from __future__ import annotations
import json
import logging
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.reminders")

REMINDERS_PATH = Path(__file__).resolve().parent.parent / "jarvis_reminders.json"


class RemindersMCP(MCPBase):
    """Reminders and alarms MCP module."""

    name = "reminders"
    description = "Set alarms, timers, and reminders"
    commands = [
        "set reminder",
        "remind me",
        "set alarm",
        "set timer",
        "cancel reminder",
        "list reminders",
        "clear reminders",
    ]

    def __init__(self):
        super().__init__()
        self._reminders: List[Dict] = []
        self._active_timers: List[threading.Timer] = []
        self._callback = None  # Callback to speak reminder

    def initialize(self) -> bool:
        """Initialize reminders module."""
        self._reminders = self._load_reminders()
        self._start_pending_reminders()
        self._initialized = True
        log.info("Reminders MCP initialized with %d reminders.", len(self._reminders))
        return True

    def set_callback(self, callback) -> None:
        """Set callback function to speak reminders."""
        self._callback = callback

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle reminder commands."""
        if command in ("set reminder", "remind me"):
            if args:
                return self.set_reminder(args)
            return "What should I remind you about, sir?"

        if command in ("set alarm",):
            if args:
                return self.set_alarm(args)
            return "What time should I set the alarm, sir?"

        if command in ("set timer",):
            if args:
                return self.set_timer(args)
            return "How long should I set the timer for, sir?"

        if command in ("cancel reminder",):
            if args:
                return self.cancel_reminder(args)
            return "Which reminder should I cancel, sir?"

        if command in ("list reminders",):
            return self.list_reminders()

        if command in ("clear reminders",):
            return self.clear_all_reminders()

        return None

    def set_reminder(self, args: str) -> str:
        """Set a reminder from natural language."""
        # Parse reminder from text
        # Examples: "remind me to call mom in 30 minutes"
        #           "set reminder meeting at 3pm"
        #           "remind me to buy groceries tomorrow at 5pm"
        
        args_lower = args.lower()
        
        # Extract time offset (in X minutes/hours)
        time_offset = None
        time_unit = None
        
        if " in " in args_lower:
            parts = args_lower.split(" in ")
            if len(parts) == 2:
                time_part = parts[1].strip()
                # Parse "30 minutes", "2 hours", "1 hour", etc.
                if "minute" in time_part:
                    time_offset = self._extract_number(time_part)
                    time_unit = "minutes"
                elif "hour" in time_part:
                    time_offset = self._extract_number(time_part)
                    time_unit = "hours"
                elif "second" in time_part:
                    time_offset = self._extract_number(time_part)
                    time_unit = "seconds"
                elif "day" in time_part:
                    time_offset = self._extract_number(time_part)
                    time_unit = "days"
                
                if time_offset:
                    message = parts[0].strip()
                    return self._create_reminder_with_offset(message, time_offset, time_unit)
        
        # Try to parse specific time
        if " at " in args_lower:
            parts = args_lower.split(" at ")
            if len(parts) == 2:
                message = parts[0].strip()
                time_str = parts[1].strip()
                return self._create_reminder_at_time(message, time_str)
        
        # Default: remind in 5 minutes
        return self._create_reminder_with_offset(args, 5, "minutes")

    def set_alarm(self, args: str) -> str:
        """Set an alarm at a specific time."""
        # Try to parse time like "7:30am", "7:30 pm", "19:30"
        import re
        
        time_patterns = [
            r"(\d{1,2}):(\d{2})\s*(am|pm)",
            r"(\d{1,2})\s*(am|pm)",
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, args.lower())
            if match:
                groups = match.groups()
                hour = int(groups[0])
                minute = int(groups[1]) if len(groups) > 2 and groups[1].isdigit() else 0
                period = groups[-1] if len(groups) > 2 else None
                
                if period == "pm" and hour < 12:
                    hour += 12
                elif period == "am" and hour == 12:
                    hour = 0
                
                now = datetime.now()
                alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                if alarm_time <= now:
                    alarm_time += timedelta(days=1)
                
                delay = (alarm_time - now).total_seconds()
                message = f"Alarm for {hour:02d}:{minute:02d}"
                
                return self._schedule_reminder(message, delay)
        
        return "I couldn't understand the time format, sir. Try something like '7:30pm' or '19:30'."

    def set_timer(self, args: str) -> str:
        """Set a countdown timer."""
        import re
        
        # Parse duration like "5 minutes", "30 seconds", "1 hour"
        patterns = [
            (r"(\d+)\s*second", 1),
            (r"(\d+)\s*minute", 60),
            (r"(\d+)\s*hour", 3600),
        ]
        
        for pattern, multiplier in patterns:
            match = re.search(pattern, args.lower())
            if match:
                amount = int(match.group(1))
                delay = amount * multiplier
                return self._schedule_reminder(f"Timer: {amount} {'second' if multiplier == 1 else 'minute' if multiplier == 60 else 'hour'}{'s' if amount > 1 else ''}", delay)
        
        return "I couldn't understand the duration, sir. Try something like '5 minutes' or '30 seconds'."

    def _extract_number(self, text: str) -> Optional[int]:
        """Extract a number from text."""
        import re
        match = re.search(r"(\d+)", text)
        return int(match.group(1)) if match else None

    def _create_reminder_with_offset(self, message: str, amount: int, unit: str) -> str:
        """Create a reminder with time offset."""
        multipliers = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600,
            "days": 86400,
        }
        delay = amount * multipliers.get(unit, 60)
        return self._schedule_reminder(message, delay)

    def _create_reminder_at_time(self, message: str, time_str: str) -> str:
        """Create a reminder at a specific time."""
        from datetime import datetime as dt
        now = dt.now()
        target = None
        # Try parsing "3pm", "15:30", "3:45pm", etc.
        match = re.match(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', time_str.strip().lower())
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            ampm = match.group(3)
            if ampm == 'pm' and hour < 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                from datetime import timedelta
                target += timedelta(days=1)
        if target:
            delay = (target - now).total_seconds()
            return self._schedule_reminder(message, delay)
        # Fallback: 5 minutes
        return self._schedule_reminder(message, 300)

    def _schedule_reminder(self, message: str, delay_seconds: float) -> str:
        """Schedule a reminder with delay."""
        existing_ids = [r["id"] for r in self._reminders]
        next_id = max(existing_ids, default=0) + 1
        reminder = {
            "id": next_id,
            "message": message,
            "created": datetime.now().isoformat(),
            "trigger_at": (datetime.now() + timedelta(seconds=delay_seconds)).isoformat(),
            "triggered": False,
        }
        self._reminders.append(reminder)
        self._save_reminders()
        
        # Schedule the timer
        timer = threading.Timer(delay_seconds, self._trigger_reminder, args=[reminder["id"]])
        timer.daemon = True
        timer.start()
        self._active_timers.append(timer)
        
        # Format response
        if delay_seconds < 60:
            return f"Reminder set for {int(delay_seconds)} seconds, sir."
        elif delay_seconds < 3600:
            return f"Reminder set for {int(delay_seconds / 60)} minutes, sir."
        else:
            return f"Reminder set for {int(delay_seconds / 3600)} hours, sir."

    def _trigger_reminder(self, reminder_id: int) -> None:
        """Trigger a reminder."""
        for r in self._reminders:
            if r["id"] == reminder_id and not r["triggered"]:
                r["triggered"] = True
                self._save_reminders()
                log.info("Reminder triggered: %s", r["message"])
                
                # Speak the reminder
                if self._callback:
                    self._callback(f"Reminder: {r['message']}")
                break

    def _start_pending_reminders(self) -> None:
        """Start timers for pending reminders."""
        now = datetime.now()
        for r in self._reminders:
            if not r.get("triggered"):
                trigger_at = datetime.fromisoformat(r["trigger_at"])
                if trigger_at > now:
                    delay = (trigger_at - now).total_seconds()
                    timer = threading.Timer(delay, self._trigger_reminder, args=[r["id"]])
                    timer.daemon = True
                    timer.start()
                    self._active_timers.append(timer)

    def cancel_reminder(self, args: str) -> str:
        """Cancel a reminder by ID or message."""
        # Try to find by ID
        if args.isdigit():
            reminder_id = int(args)
            for i, r in enumerate(self._reminders):
                if r["id"] == reminder_id:
                    self._reminders.pop(i)
                    self._save_reminders()
                    return f"Reminder {reminder_id} cancelled, sir."
        
        # Try to find by message
        for i, r in enumerate(self._reminders):
            if args.lower() in r["message"].lower():
                self._reminders.pop(i)
                self._save_reminders()
                return f"Reminder '{r['message']}' cancelled, sir."
        
        return f"I couldn't find a reminder matching '{args}', sir."

    def list_reminders(self) -> str:
        """List all pending reminders."""
        pending = [r for r in self._reminders if not r.get("triggered")]
        if not pending:
            return "No pending reminders, sir."
        
        lines = []
        for r in pending[:5]:
            trigger_at = datetime.fromisoformat(r["trigger_at"])
            time_left = trigger_at - datetime.now()
            if time_left.total_seconds() > 0:
                minutes = int(time_left.total_seconds() / 60)
                lines.append(f"Reminder {r['id']}: {r['message']} (in {minutes} min)")
        
        return "Pending reminders: " + "; ".join(lines) if lines else "No pending reminders, sir."

    def clear_all_reminders(self) -> str:
        """Clear all reminders."""
        count = len(self._reminders)
        self._reminders.clear()
        self._save_reminders()
        return f"Cleared {count} reminders, sir."

    def _load_reminders(self) -> List[Dict]:
        """Load reminders from file."""
        try:
            with open(REMINDERS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

    def _save_reminders(self) -> None:
        """Save reminders to file."""
        try:
            with open(REMINDERS_PATH, "w", encoding="utf-8") as f:
                json.dump(self._reminders, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error("Failed to save reminders: %s", e)

    def shutdown(self) -> None:
        """Cancel all active timers."""
        for timer in self._active_timers:
            timer.cancel()
        self._active_timers.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        pending = len([r for r in self._reminders if not r.get("triggered")])
        return {
            "name": self.name,
            "initialized": self._initialized,
            "total_reminders": len(self._reminders),
            "pending": pending,
        }
