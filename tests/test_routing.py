"""Command routing tests.

These lock in the pattern-precedence fixes. Order matters: ``_ACTION_PATTERNS``
is evaluated top to bottom and the first hit wins, so a generic rule placed
above a specific one silently shadows it.
"""

from __future__ import annotations

import re

import pytest

# (utterance, expected action, expected payload substring or None)
CASES = [
    # --- specific openers must beat the generic launcher ---
    ("open notepad", "open_app", "notepad"),
    ("launch chrome", "open_app", "chrome"),
    ("open the file report.txt", "open_file", "report.txt"),
    ("open folder documents", "open_folder", "documents"),
    ("open gmail", "gmail", None),
    ("open maps", "maps", None),
    ("open spotify", "spotify", None),
    ("start application word", "open_app", "word"),
    ("run app vscode", "open_app", "vscode"),

    # --- URLs must beat the generic launcher, not become an app name ---
    ("go to google.com", "open_website", "google.com"),
    ("browse example.org", "open_website", "example.org"),
    ("open github.com", "open_website", "github.com"),
    ("open website github.com", "open_website", "github.com"),

    # --- timers and reminders outrank the generic "start" ---
    ("start timer for 5 minutes", "set_timer", "5 minutes"),
    ("start a timer for 10 minutes", "set_timer", "10 minutes"),
    ("set a timer for 2 minutes", "set_timer", "2 minutes"),
    ("set timer 3 minutes", "set_timer", "3 minutes"),
    ("timer for 4 minutes", "set_timer", "4 minutes"),
    ("set a reminder to call mom", "set_reminder", "call mom"),
    ("set reminder drink water", "set_reminder", "drink water"),
    ("remind me to stretch", "set_reminder", "stretch"),

    # --- volume: numeric is the system mixer, relative is media ---
    ("set volume to 50", "volume", "50"),
    ("volume 30", "volume", "30"),
    ("the volume to 40", "volume", "40"),
    ("turn volume up", "spotify_volume_up", None),
    ("turn volume down", "spotify_volume_down", None),
    ("volume up", "spotify_volume_up", None),
    ("volume down", "spotify_volume_down", None),
    ("increase the volume", "spotify_volume_up", None),
    ("decrease the volume", "spotify_volume_down", None),
    # Regression: a shared "(?:up|down)" pattern sent "down" to *_up.
    ("spotify volume up", "spotify_volume_up", None),
    ("spotify volume down", "spotify_volume_down", None),
    ("spotify volume to 70", "spotify_volume_set", "70"),
    ("set spotify volume to 70", "spotify_volume_set", "70"),
    ("increase spotify volume", "spotify_volume_up", None),
    ("decrease spotify volume", "spotify_volume_down", None),
    ("spotify volume to the maximum", "spotify_volume_max", None),
    ("spotify volume to the minimum", "spotify_volume_min", None),
    # Regression: a bare "spotify <x>" rule used to swallow volume commands.
    ("spotify for the monkeys", "spotify", "the monkeys"),

    # --- mute: bare targets the voice, explicit targets the player ---
    ("mute", "voice_mute", None),
    ("unmute", "voice_unmute", None),
    ("shut up", "voice_mute", None),
    ("be quiet", "voice_mute", None),
    ("mute spotify", "spotify_mute", None),
    ("mute the music", "spotify_mute", None),
    ("mute system volume", "volume_mute", None),
    ("mute the system volume", "volume_mute", None),

    # --- file operations keep their captures ---
    ("copy report.txt to backup", "copy_file", "report.txt to backup"),
    ("move notes.md to archive", "move_file", "notes.md to archive"),
    ("type hello world", "type_text", "hello world"),
    # Regression: "write a file called x" used to be typed as literal text.
    ("write a file called readme.md", "create_file", "readme.md"),
    ("create a file at notes.txt", "create_file", "notes.txt"),
    ("read the file main.py", "read_file", "main.py"),
    ("delete the file old.txt", "delete_file", "old.txt"),

    # --- file search must not be swallowed by web search ---
    ("find file budget.xlsx", "find_file", "budget.xlsx"),
    ("search for files containing report", "find_file", "report"),
    ("search for invoice", "search", "invoice"),
    ("google cats", "search", "cats"),
    ("search the web for python tips", "search", "python tips"),

    # --- mouse and window control ---
    ("move mouse to 100 200", "mouse_move", "100, 200"),
    ("click at 50 60", "mouse_click", "50, 60"),
    ("list windows", "list_windows", None),
    ("show the windows", "list_windows", None),
    ("minimize window", "minimize_window", None),
    ("minimise the window", "minimize_window", None),
    ("maximize window", "maximize_window", None),
    ("maximise the window", "maximize_window", None),
    ("close the window", "close_window", None),
    ("close the tab", "close_tab", None),
    ("show the processes", "list_processes", None),

    # --- "running apps" needs a bare form, not just a question ---
    ("what's running", "running_apps", None),
    ("running apps", "running_apps", None),
    ("what are the running apps", "running_apps", None),
    ("active programs", "running_apps", None),

    # --- system metrics are reachable as phrases, not just via the MCP ---
    ("system info", "system_info", None),
    ("computer status", "system_info", None),
    ("cpu usage", "cpu_info", None),
    ("how much ram", "ram_info", None),
    ("battery", "battery_info", None),
    ("battery status", "battery_info", None),
    ("disk usage", "disk_info", None),
    ("network usage", "network_info", None),

    # --- core actions ---
    ("what time is it", "time", None),
    ("lock the computer", "lock_pc", None),
    ("take a screenshot", "take_screenshot", None),
    ("add task buy milk", "add_task", "buy milk"),
    ("complete task 2", "complete_task", "2"),
    ("list tasks", "list_tasks", None),
    ("my todos", "list_tasks", None),
    ("remember I parked on level 3", "remember", "level 3"),
    ("what do you know about my car", "recall", "my car"),

    # --- power commands: regression, these all used to route to *sleep* ---
    ("shut down", "shutdown_pc", None),
    ("shutdown", "shutdown_pc", None),
    ("shutdown the pc", "shutdown_pc", None),
    ("restart the computer", "restart_pc", None),
    ("reboot", "restart_pc", None),
    ("hibernate", "hibernate_pc", None),
    ("cancel shutdown", "cancel_shutdown", None),
    ("sleep", "sleep_pc", None),

    # --- voice controls ---
    ("voice status", "voice_status", None),
    ("list voices", "list_voices", None),
    ("set voice ryan", "set_voice", "ryan"),
    ("voice backend", "voice_backend", None),
    ("set voice backend fish", "voice_backend_set", "fish"),

    # --- directory listing, without swallowing "list windows"/"list voices" ---
    ("list directory", "list_directory", None),
    ("list folder", "list_directory", None),
    ("ls", "list_directory", None),

    # --- conversation ---
    ("hello", "greeting", None),
    ("hi there", "greeting", None),
    ("thanks", "thanks", None),
    ("what can you do", "about", None),
    ("who are you", "about", None),
    ("never mind", "cancel", None),
    ("fix error", "fix_error", None),
    ("heal", "fix_error", None),
    ("self heal", "fix_error", None),
]


CASE_IDS = [c[0] for c in CASES]


@pytest.mark.parametrize("text,expected_action,expected_payload", CASES, ids=CASE_IDS)
def test_routes_to_expected_action(app_module, text, expected_action, expected_payload):
    matched = app_module._match_pattern(text.lower())
    assert matched is not None, f"no pattern matched {text!r}"
    action, payload, _matched_text = matched
    assert action == expected_action, (
        f"{text!r} routed to {action!r} (payload {payload!r}), expected {expected_action!r}"
    )
    if expected_payload is not None:
        assert expected_payload in payload, (
            f"{text!r} payload {payload!r} does not contain {expected_payload!r}"
        )


def test_all_capture_groups_are_preserved(app_module):
    """Regression: a multi-group pattern used to drop everything after group 1."""
    action, payload, _ = app_module._match_pattern("click at 500 300")
    assert action == "mouse_click"
    assert payload == "500, 300", f"lost a capture group: {payload!r}"


def test_patterns_are_all_compiled(app_module):
    for pattern, action in app_module._ACTION_PATTERNS:
        try:
            re.compile(pattern)
        except re.error as e:  # pragma: no cover - only on a bad edit
            pytest.fail(f"pattern for {action} is invalid: {pattern!r} ({e})")


def test_every_pattern_action_has_a_dispatcher_branch(app_module):
    """A pattern pointing at an unhandled action is a silent no-op."""
    source = (app_module.Path(app_module.__file__)).read_text(encoding="utf-8")
    actions = {name for _, name in app_module._ACTION_PATTERNS}
    missing = sorted(a for a in actions if f'action_type == "{a}"' not in source)
    assert not missing, f"actions with no dispatcher branch: {missing}"


def test_action_map_and_names_are_consistent(app_module):
    """ACTION_MAP collapses duplicate actions, so it is keyed by action name."""
    unique_actions = {name for _, name in app_module._ACTION_PATTERNS}
    assert set(app_module.ACTION_MAP) == unique_actions
    assert set(app_module.ACTION_NAMES) == unique_actions
    # Every mapped value must be one of the table's regexes.
    table = {p for p, _ in app_module._ACTION_PATTERNS}
    assert set(app_module.ACTION_MAP.values()) <= table


def test_non_url_destinations_do_not_become_websites(app_module):
    """"go to the gym" must not be read as a web address."""
    for text in ("go to the gym", "browse the news", "open my documents"):
        matched = app_module._match_pattern(text)
        if matched is not None:
            assert matched[0] != "open_website", f"{text!r} was treated as a URL"


def test_fuzzy_match_respects_minimum_word_length(app_module):
    """Short words must not be fuzzy-corrected into nonsense."""
    pairs = app_module._fuzzy_correct_words("open teh notepad")
    assert ("teh", True) not in pairs
    assert ("teh", False) in pairs


def test_fuzzy_match_corrects_real_typos(app_module):
    corrected = {w for w, changed in app_module._fuzzy_correct_words("opne notepd") if changed}
    assert "opne" not in corrected, "should have repaired 'opne' -> 'open'"


def test_fuzzy_match_leaves_correct_words_alone(app_module):
    assert all(not changed for _, changed in app_module._fuzzy_correct_words("open notepad chrome"))


def test_detect_action_falls_back_to_chat_for_unknown_utterance(app_module):
    """Unknown input must not raise; it routes to the conversational path."""
    action, _ = app_module._detect_action("zzz qqq wwww")
    assert action in ("chat", "llm_parse"), f"unexpected action {action!r}"


def test_short_input_is_ignored(app_module):
    for text in ("", "   ", "a"):
        app_module.handle_command(text, {})
