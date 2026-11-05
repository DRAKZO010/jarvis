"""Unit tests for robust patch applicator."""

import pytest
from mcp.apply_patch import apply_hunks_to_content, apply_patch_file, parse_patch


def test_parse_patch():
    diff = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def hello():\n"
        "-    print('hi')\n"
        "+    print('hello world')\n"
        "     return True\n"
    )
    path, hunks = parse_patch(diff)
    assert path == "foo.py"
    assert len(hunks) == 1
    assert hunks[0].old_start == 1


def test_apply_hunks_exact():
    original = "def hello():\n    print('hi')\n    return True\n"
    diff = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def hello():\n"
        "-    print('hi')\n"
        "+    print('hello world')\n"
        "     return True\n"
    )
    _, hunks = parse_patch(diff)
    result = apply_hunks_to_content(original, hunks)
    assert "print('hello world')" in result
    assert "print('hi')" not in result


def test_apply_hunks_with_line_drift():
    # Insert 5 lines at top so line numbers drift
    original = (
        "# extra 1\n# extra 2\n# extra 3\n# extra 4\n# extra 5\n"
        "def hello():\n    print('hi')\n    return True\n"
    )
    # The diff says old_start=1, but the target code is actually at line 6
    diff = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def hello():\n"
        "-    print('hi')\n"
        "+    print('hello world')\n"
        "     return True\n"
    )
    _, hunks = parse_patch(diff)
    result = apply_hunks_to_content(original, hunks)
    assert "print('hello world')" in result
    assert "# extra 1" in result

