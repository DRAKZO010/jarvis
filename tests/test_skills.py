"""Unit tests for the Skill System."""

from pathlib import Path
from mcp.skills import SkillManager, SkillManifest, SkillParser, SkillStep


def test_parse_skill_markdown():
    markdown = """---
name: sample_skill
version: 1.0.0
description: A sample test skill
author: Antigravity
tags:
  - testing
  - automation
steps:
  - tool: get_weather
    args: "{city}"
    output_key: weather_data
---

# Sample Skill Guide
This skill demonstrates procedural instructions.
1. Check the weather
2. Report the summary
"""
    manifest = SkillParser.parse_text(markdown)
    assert manifest.name == "sample_skill"
    assert manifest.version == "1.0.0"
    assert manifest.description == "A sample test skill"
    assert "testing" in manifest.tags
    assert len(manifest.steps) == 1
    assert manifest.steps[0].tool_name == "get_weather"
    assert "city" in manifest.parameters
    assert "Sample Skill Guide" in manifest.markdown_content


def test_skill_manager_execution(tmp_path):
    skills_dir = tmp_path / "skills"
    test_skill_dir = skills_dir / "calc_weather"
    test_skill_dir.mkdir(parents=True)

    (test_skill_dir / "SKILL.md").write_text(
        """---
name: calc_weather
description: Look up weather and format
steps:
  - tool: weather
    args: "{city}"
    output_key: weather
---
# Instructions
""",
        encoding="utf-8",
    )

    def fake_dispatcher(tool, args):
        if tool == "weather":
            return f"Sunny in {args}"
        return "unknown"

    manager = SkillManager(skills_dir=skills_dir, tool_dispatcher=fake_dispatcher)
    assert len(manager.list_skills()) == 1

    res = manager.execute_skill("calc_weather", city="Tokyo")
    assert res["ok"]
    assert res["context"]["weather"] == "Sunny in Tokyo"

