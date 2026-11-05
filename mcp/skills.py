"""Skill System — adapted from OpenJarvis.

Follows the agentskills.io standard (SKILL.md with YAML frontmatter + procedural
instructions / executable tool pipeline steps).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

log = logging.getLogger("jarvis.mcp.skills")

DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
_PARAM_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")


@dataclass(slots=True)
class SkillStep:
    """A single deterministic step in a skill pipeline."""

    tool_name: str = ""
    arguments_template: str = "{}"
    output_key: str = ""


@dataclass(slots=True)
class SkillManifest:
    """Manifest describing a reusable skill loaded from SKILL.md."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = "Jarvis"
    steps: List[SkillStep] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    markdown_content: str = ""
    path: Optional[Path] = None


class SkillParser:
    """Parse SKILL.md files into SkillManifest objects."""

    @staticmethod
    def parse_text(text: str, file_path: Optional[Path] = None) -> SkillManifest:
        frontmatter = {}
        markdown_body = text

        # Parse YAML frontmatter if present
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    markdown_body = parts[2].strip()
                except Exception as e:
                    log.warning("Failed to parse YAML frontmatter in %s: %s", file_path, e)

        name = frontmatter.get("name") or (file_path.parent.name if file_path else "unnamed_skill")
        description = frontmatter.get("description") or ""
        version = str(frontmatter.get("version", "0.1.0"))
        author = str(frontmatter.get("author", "Jarvis"))
        tags = frontmatter.get("tags") or []

        # Parse steps if defined in frontmatter
        steps_data = frontmatter.get("steps") or []
        steps: List[SkillStep] = []
        for s in steps_data:
            if isinstance(s, dict):
                steps.append(
                    SkillStep(
                        tool_name=s.get("tool_name", s.get("tool", "")),
                        arguments_template=str(s.get("arguments_template", s.get("args", "{}"))),
                        output_key=str(s.get("output_key", "")),
                    )
                )

        # Detect parameters from step templates or frontmatter
        params: set[str] = set(frontmatter.get("parameters") or [])
        for step in steps:
            for match in _PARAM_RE.finditer(step.arguments_template):
                params.add(match.group(1))

        return SkillManifest(
            name=name,
            version=version,
            description=description,
            author=author,
            steps=steps,
            tags=tags,
            parameters=sorted(params),
            markdown_content=markdown_body,
            path=file_path,
        )

    @classmethod
    def parse_file(cls, path: Path) -> SkillManifest:
        text = path.read_text(encoding="utf-8")
        return cls.parse_text(text, file_path=path)


class SkillManager:
    """Manages skill discovery, cataloging, and execution."""

    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        tool_dispatcher: Optional[Callable[[str, Any], Any]] = None,
    ):
        self.skills_dir = skills_dir or DEFAULT_SKILLS_DIR
        self.tool_dispatcher = tool_dispatcher
        self._skills: Dict[str, SkillManifest] = {}
        self.reload()

    def reload(self) -> int:
        """Scan skills directory and load all valid SKILL.md manifests."""
        self._skills.clear()
        if not self.skills_dir.exists():
            try:
                self.skills_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                log.warning("Could not create skills dir %s: %s", self.skills_dir, e)
                return 0

        # Scan for SKILL.md in immediate subfolders or directly
        for skill_file in self.skills_dir.glob("**/SKILL.md"):
            try:
                manifest = SkillParser.parse_file(skill_file)
                self._skills[manifest.name] = manifest
                log.info("Loaded skill: %s v%s", manifest.name, manifest.version)
            except Exception as e:
                log.error("Failed to load skill at %s: %s", skill_file, e)

        return len(self._skills)

    def register_manifest(self, manifest: SkillManifest) -> None:
        """Register an in-memory skill manifest."""
        self._skills[manifest.name] = manifest

    def get_skill(self, name: str) -> Optional[SkillManifest]:
        return self._skills.get(name)

    def list_skills(self) -> List[Dict[str, Any]]:
        """Return catalog list of all loaded skills."""
        return [
            {
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "tags": m.tags,
                "parameters": m.parameters,
                "is_pipeline": len(m.steps) > 0,
            }
            for m in self._skills.values()
        ]

    def execute_skill(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute a skill by running its steps or returning instructions."""
        skill = self.get_skill(name)
        if not skill:
            return {"ok": False, "error": f"Skill '{name}' not found"}

        # If it's an instruction / procedural skill (no deterministic steps)
        if not skill.steps:
            return {
                "ok": True,
                "skill": name,
                "type": "procedural",
                "instructions": skill.markdown_content,
                "description": skill.description,
            }

        # Deterministic pipeline execution
        if not self.tool_dispatcher:
            return {
                "ok": False,
                "error": "No tool dispatcher configured to run pipeline steps",
            }

        context = dict(kwargs)
        step_outputs: List[Dict[str, Any]] = []

        for i, step in enumerate(skill.steps):
            # Render arguments template
            args_str = step.arguments_template
            for k, v in context.items():
                args_str = args_str.replace(f"{{{k}}}", str(v))

            try:
                result = self.tool_dispatcher(step.tool_name, args_str)
                if step.output_key:
                    context[step.output_key] = result
                step_outputs.append(
                    {
                        "step": i + 1,
                        "tool": step.tool_name,
                        "args": args_str,
                        "result": result,
                    }
                )
            except Exception as e:
                log.exception("Step %d (%s) failed in skill %s", i + 1, step.tool_name, name)
                return {
                    "ok": False,
                    "skill": name,
                    "error": f"Step {i + 1} ({step.tool_name}) failed: {e}",
                    "partial_results": step_outputs,
                }

        return {
            "ok": True,
            "skill": name,
            "type": "pipeline",
            "results": step_outputs,
            "context": context,
        }

