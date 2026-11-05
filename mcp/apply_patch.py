"""Robust Patch Applicator — adapted from OpenJarvis.

Parses unified diff hunks, tolerates CRLF / LF line endings and minor line
offset drifts, creates atomic backups, and cleanly applies or rejects patches.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger("jarvis.mcp.apply_patch")

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    """A single hunk from a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str] = field(default_factory=list)


def parse_patch(patch_text: str) -> Tuple[Optional[str], List[Hunk]]:
    """Parse unified diff text into target path and hunks."""
    lines = patch_text.splitlines()
    target_path: Optional[str] = None
    hunks: List[Hunk] = []
    current_hunk: Optional[Hunk] = None

    for line in lines:
        if line.startswith("+++ "):
            path_part = line[4:].strip()
            if path_part.startswith("b/"):
                path_part = path_part[2:]
            if path_part != "/dev/null":
                target_path = path_part
            continue

        if line.startswith("--- "):
            continue

        m = _HUNK_HEADER_RE.match(line)
        if m:
            current_hunk = Hunk(
                old_start=int(m.group(1)),
                old_count=int(m.group(2)) if m.group(2) is not None else 1,
                new_start=int(m.group(3)),
                new_count=int(m.group(4)) if m.group(4) is not None else 1,
            )
            hunks.append(current_hunk)
            continue

        if current_hunk is not None:
            if line.startswith((" ", "+", "-")):
                current_hunk.lines.append(line)
            elif line == "\\ No newline at end of file":
                continue
            elif line == "":
                current_hunk.lines.append(" ")

    if not hunks:
        raise ValueError("No valid hunks found in patch")

    return target_path, hunks


def _find_best_match(
    orig_lines: List[str],
    expected_old_lines: List[str],
    target_pos: int,
    max_drift: int = 50,
) -> Optional[int]:
    """Find the best matching starting line within +/- max_drift of target_pos."""
    if not expected_old_lines:
        return target_pos

    n = len(expected_old_lines)
    best_pos = None
    min_dist = float("inf")

    # Check target position first
    candidates = [target_pos]
    for d in range(1, max_drift + 1):
        candidates.append(target_pos + d)
        candidates.append(target_pos - d)

    for cand in candidates:
        if 0 <= cand <= len(orig_lines) - n:
            match = True
            for i, exp in enumerate(expected_old_lines):
                if orig_lines[cand + i].rstrip("\r\n") != exp.rstrip("\r\n"):
                    match = False
                    break
            if match:
                dist = abs(cand - target_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_pos = cand
                    if dist == 0:
                        break

    return best_pos


def apply_hunks_to_content(content: str, hunks: List[Hunk]) -> str:
    """Apply parsed hunks to a string containing the original content."""
    orig_lines = content.splitlines(keepends=True)
    offset = 0

    for hunk_idx, hunk in enumerate(hunks):
        expected_old_lines: List[str] = []
        for l in hunk.lines:
            if l.startswith((" ", "-")):
                expected_old_lines.append(l[1:])

        suggested_pos = max(0, hunk.old_start - 1 + offset)
        actual_pos = _find_best_match(orig_lines, expected_old_lines, suggested_pos, max_drift=60)

        if actual_pos is None:
            # Fallback to direct comparison to produce descriptive error
            raise ValueError(
                f"Hunk {hunk_idx + 1} failed: context lines do not match target around line {suggested_pos + 1}"
            )

        new_replacement_lines: List[str] = []
        check_pos = actual_pos

        for diff_line in hunk.lines:
            tag = diff_line[0]
            line_body = diff_line[1:]

            if tag == " ":
                new_replacement_lines.append(orig_lines[check_pos])
                check_pos += 1
            elif tag == "-":
                check_pos += 1
            elif tag == "+":
                newline = "\r\n" if (orig_lines and orig_lines[0].endswith("\r\n")) else "\n"
                new_replacement_lines.append(line_body.rstrip("\r\n") + newline)

        consumed = check_pos - actual_pos
        orig_lines[actual_pos : actual_pos + consumed] = new_replacement_lines
        offset += len(new_replacement_lines) - consumed

    return "".join(orig_lines)


def apply_patch_file(
    patch_text: str,
    target_path: Optional[str] = None,
    backup: bool = True,
    base_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Apply a unified diff patch to a file.

    Returns (success, message).
    """
    try:
        detected_path, hunks = parse_patch(patch_text)
    except Exception as e:
        return False, f"Failed to parse patch: {e}"

    rel_path = target_path or detected_path
    if not rel_path:
        return False, "Target file path could not be detected from patch."

    file_path = (base_dir / rel_path) if base_dir else Path(rel_path)
    if not file_path.exists():
        return False, f"Target file does not exist: {file_path}"

    try:
        content = file_path.read_text(encoding="utf-8")
        patched = apply_hunks_to_content(content, hunks)
    except Exception as e:
        return False, f"Error applying hunks: {e}"

    # Create backup
    bak_path = file_path.with_suffix(file_path.suffix + ".bak")
    if backup:
        try:
            shutil.copyfile(file_path, bak_path)
        except Exception as e:
            log.warning("Could not create backup %s: %s", bak_path, e)

    try:
        file_path.write_text(patched, encoding="utf-8")
        return True, f"Successfully patched {file_path}"
    except Exception as e:
        if backup and bak_path.exists():
            shutil.copyfile(bak_path, file_path)
        return False, f"Failed to write patched file: {e}"

