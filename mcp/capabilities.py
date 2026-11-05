"""Capability catalogue for LLM command routing.

The pattern table in ``jarvis.py`` is the source of truth for what the
assistant can do, but a bare list of ``snake_case`` names is not enough for a
language model to pick the right one: ``add_task`` and ``create_file`` are not
self-explanatory. This module gives each action a description so the model has
something to reason with, and it decides which actions the model is allowed to
choose at all.

Two rules shape the design:

* The regular expressions run first and are unaffected. They stay fast, offline,
  and free. Routing through the model is only reached for speech the table does
  not recognise.
* The model's choice is deny-by-default. Anything not explicitly listed in
  :data:`LLM_ROUTABLE_ACTIONS` cannot be selected, because a fuzzy, model-driven
  surface pointed at ``execute`` or ``delete_file`` is a far bigger risk than the
  narrow, reviewable patterns those actions already have.
"""

from __future__ import annotations

# Action name -> what it does, phrased so the model can match user intent.
# "chat" is the one entry with no matching row in the pattern table: it is the
# synthetic action meaning "answer conversationally", returned when the model
# decides no action fits.
ACTION_CATALOGUE: dict[str, str] = {
    # --- conversation / meta -------------------------------------------
    "chat": "Answer a question, give an opinion, or explain something. Default choice.",
    "about": "Describe yourself, your version, or what you can do.",
    "greeting": "Say hello, ask how you are, or check you are awake.",
    "thanks": "The user is thanking you.",
    "cancel": "Stop or cancel what you are currently doing.",
    "cancel_shutdown": "Cancel a scheduled shutdown or restart.",
    "recall": "Repeat or recall your last spoken reply.",
    "remember": "Store a fact the user wants kept (payload is the fact).",
    "run_skill": "Execute an installed skill procedure or pipeline. Payload is the skill name, like morning digest or deep research.",
    "list_skills": "List all installed skills.",
    "fix_error": "Diagnose a recent error in the app.",

    # --- time / information --------------------------------------------
    "time": "Tell the time or date.",
    "weather": "Weather forecast. Payload is an optional city name.",
    "news": "General current headlines. Payload is an optional topic.",
    "tech_news": "Technology headlines.",
    "sports_news": "Sports headlines, including match scores.",

    # --- system status --------------------------------------------------
    "system_info": "Overall system summary.",
    "cpu_info": "CPU usage and details.",
    "ram_info": "Memory usage and details.",
    "disk_info": "Disk usage and free space.",
    "network_info": "Network throughput and addresses.",
    "battery_info": "Battery level and charging state.",
    "list_processes": "List running processes.",
    "running_apps": "List open applications and windows.",
    "list_windows": "List all open windows.",

    # --- tasks ----------------------------------------------------------
    "list_tasks": "List outstanding tasks or the todo list.",
    "add_task": "Add a task. Payload is the task text, e.g. 'buy milk'.",
    "complete_task": "Mark a task done. Payload identifies which task.",
    "delete_task": "Delete a task by number. Payload is the number.",
    "delete_task_by_desc": "Delete a task by its description. Payload is the text.",
    "delete_multiple_tasks": "Delete several tasks. Payload lists them.",
    "clear_tasks": "Remove every completed task from the list.",
    "clear_all_tasks": "Delete the entire task list.",

    # --- reminders / timers ---------------------------------------------
    "set_reminder": "Set a reminder. Payload is what to remind about, and when if given.",
    "list_reminders": "List pending reminders.",
    "clear_reminders": "Delete all reminders.",
    "set_alarm": "Set an alarm. Payload is the time, e.g. '7am'.",
    "set_timer": "Start a countdown timer. Payload is the duration, e.g. '10 minutes'.",

    # --- launching things ------------------------------------------------
    "open_app": "Open an application. Payload is the app name, e.g. 'notepad'.",
    "open_website": "Open a website by name or address. Payload is the site.",
    "open_file": "Open a file by name or path. Payload is the file.",
    "open_folder": "Open a folder in Explorer. Payload is the folder.",
    "search": "Search the web. Payload is the query.",
    "youtube": "Open YouTube, optionally a search. Payload is optional.",
    "gmail": "Open Gmail.",
    "maps": "Open Maps. Payload is an optional place to search for.",

    # --- spotify ----------------------------------------------------------
    "spotify": "Control Spotify generally; payload says what to do.",
    "spotify_play": "Resume Spotify playback.",
    "spotify_pause": "Pause Spotify.",
    "spotify_next": "Skip to the next Spotify track.",
    "spotify_previous": "Go to the previous Spotify track.",
    "spotify_random": "Shuffle Spotify playback.",
    "spotify_repeat": "Toggle Spotify repeat.",
    "spotify_shuffle": "Toggle Spotify shuffle.",
    "spotify_mute": "Mute Spotify.",
    "spotify_volume_up": "Turn Spotify volume up. Payload is optional amount.",
    "spotify_volume_down": "Turn Spotify volume down. Payload is optional amount.",
    "spotify_volume_max": "Set Spotify volume to maximum.",
    "spotify_volume_min": "Set Spotify volume to minimum.",
    "spotify_volume_set": "Set Spotify volume. Payload is the level, e.g. '40'.",

    # --- system volume / screen -------------------------------------------
    "volume": "System volume up, down, or mute. Payload says which.",
    "volume_mute": "Mute the system volume.",
    "volume_unmute": "Unmute the system volume.",
    "brightness": "Set or adjust screen brightness. Payload is a level or direction.",

    # --- windows -----------------------------------------------------------
    "focus_jarvis": "Bring the assistant's own window to the front.",
    "focus_window": "Focus a named window. Payload is the window title.",
    "maximize_window": "Maximise the focused window.",
    "minimize_window": "Minimise the focused window.",
    "close_window": "Close the focused window.",
    "close_tab": "Close the current browser tab.",
    "close_app": "Close an application by name. Payload is the app name, e.g. 'spotify'.",

    # --- files -------------------------------------------------------------
    "read_file": "Read and show the contents of a file. Payload is the path.",
    "create_file": "Create a new empty file. Payload is the path.",
    "copy_file": "Copy a file. Payload is 'source destination'.",
    "move_file": "Move or rename a file. Payload is 'source destination'.",
    "find_file": "Search the disk for a file by name. Payload is the name.",
    "list_directory": "List the files in a folder. Payload is the folder.",
    "recent_files": "List recently opened files.",
    "get_clipboard": "Read the clipboard contents.",
    "set_clipboard": "Put text on the clipboard. Payload is the text.",

    # --- media / screenshots ------------------------------------------------
    "take_screenshot": "Capture a screenshot.",
    "delete_screenshots": "Delete saved screenshots.",
    "delete_last_screenshot": "Delete the most recent screenshot.",

    # --- voice --------------------------------------------------------------
    "voice_status": "Report whether voice input and output are working.",
    "voice_mute": "Stop speaking replies aloud.",
    "voice_unmute": "Resume speaking replies aloud.",
    "list_voices": "List available text-to-speech voices.",
    "set_voice": "Change the speaking voice. Payload is the voice name.",
    "voice_backend": "Report which speech backend is active. Payload names a "
                      "backend to check against the one that is in use.",

    # --- HUD -----------------------------------------------------------------
    "show_panels": "Show the assistant's side panels in the display window.",
    "hide_panels": "Hide the assistant's side panels and show only the core.",

    # --- self-evolution (blocked for the model; see below) -------------------
    "self_review": "Review the assistant's own code and recent behaviour, and "
                   "report anything that looks wrong.",
    "evolve": "Ask the assistant to review its own code and propose a fix. The "
              "change is sandboxed and waits for the operator to approve it.",
    "evolution_status": "Report how many self-review cycles have run, how many "
                        "changes are waiting for approval, and what was found.",
    "show_proposals": "List the changes waiting for the operator to approve.",
    "approve_evolution": "Apply a proposed change to the code. Payload is the "
                          "proposal id.",
    "reject_evolution": "Turn down a proposed change. Payload is the proposal id.",
    "revert_evolution": "Undo a change that was applied earlier. Payload is the "
                         "proposal id.",

    # --- power (blocked for the model; see below) ----------------------------
    "lock_pc": "Lock the computer.",
    "sleep_pc": "Put the computer to sleep.",
    "hibernate_pc": "Hibernate the computer.",
    "shutdown_pc": "Shut the computer down.",
    "restart_pc": "Restart the computer.",

    # --- input injection and process control (blocked for the model) ----------
    "execute": "Run a shell command. Payload is the command line.",
    "kill_process": "Kill a running process. Payload is the process name.",
    "type_text": "Type text on the keyboard. Payload is the text.",
    "press_key": "Press a single key. Payload is the key name.",
    "hotkey": "Press a keyboard shortcut. Payload is the combination.",
    "mouse_click": "Click the mouse. Payload is coordinates or a button.",
    "mouse_move": "Move the mouse. Payload is coordinates.",
    "delete_file": "Delete a file. Payload is the path.",
}

# Actions the model may choose. Deny-by-default: an action missing from this
# set cannot be selected through LLM routing, even if it exists in the table.
# Excluded on purpose: everything that runs code, touches the keyboard or mouse,
# destroys data, or changes the machine's power state. Those stay reachable
# through their own explicit patterns, which are narrow enough to audit.
LLM_ROUTABLE_ACTIONS: frozenset[str] = frozenset(ACTION_CATALOGUE) - {
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
    # Self-evolution: a cycle costs a full test run, and an approval is the
    # one decision that must never be made by the thing being supervised.
    "self_review",
    "evolve",
    "approve_evolution",
    "reject_evolution",
    "revert_evolution",
}


def blocked_actions() -> frozenset[str]:
    """Actions that exist in the table but the model may not choose."""
    return frozenset(ACTION_CATALOGUE) - LLM_ROUTABLE_ACTIONS


def is_routable(action: str) -> bool:
    """True when the LLM is allowed to select this action."""
    return action in LLM_ROUTABLE_ACTIONS


def describe_actions(actions: frozenset[str] | None = None) -> str:
    """Render the catalogue as an indented list for the routing prompt."""
    chosen = LLM_ROUTABLE_ACTIONS if actions is None else actions
    lines = []
    for name in sorted(chosen):
        description = ACTION_CATALOGUE.get(name, "").strip()
        if description:
            lines.append(f"  {name}: {description}")
    return "\n".join(lines)


def build_routing_prompt(utterance: str) -> str:
    """Build the prompt that asks the model to map speech onto an action.

    The catalogue is sent instead of bare action names so the model has enough
    information to choose well, and the blocked actions are named explicitly so
    it does not pick one and get silently dropped.
    """
    blocked = ", ".join(sorted(blocked_actions()))
    return (
        "You route voice commands for a Windows assistant. "
        "Choose the single best action for what the user said.\n\n"
        f"Allowed actions:\n{describe_actions()}\n\n"
        f"Never choose these; they are handled by exact commands only: {blocked}\n\n"
        "Rules:\n"
        '- Reply with ONLY JSON: {"action":"<name>","payload":"<value>"}\n'
        "- Use 'chat' for questions, opinions, explanations, and anything you "
        "cannot do with an action above. This is the correct default.\n"
        "- Put every extra word in payload: the app name, search terms, the "
        "task text, the time, the file path. Use an empty string when the "
        "action needs nothing.\n"
        "- Never invent an action name and never explain your choice.\n"
        f'User said: "{utterance}"'
    )
