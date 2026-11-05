"""Unit tests for automatic memory fact extractor."""

import json
from mcp.memory_extractor import FactExtractor


class DummyMemory:
    def __init__(self):
        self.facts = []

    def remember(self, fact):
        self.facts.append(fact)


def test_fact_extractor_sync():
    def fake_llm(prompt, sys_prompt):
        return json.dumps(["User works as a software engineer", "User prefers dark mode"])

    mem = DummyMemory()
    extractor = FactExtractor(llm_caller=fake_llm, memory_store=mem)

    facts = extractor.extract_from_turn(
        user_text="I am a software engineer and I always use dark mode in all my apps.",
        assistant_text="I will keep your preferences in mind, sir.",
    )

    assert len(facts) == 2
    assert "software engineer" in facts[0]
    assert "dark mode" in facts[1]


def test_fact_extractor_handles_empty():
    def fake_llm(prompt, sys_prompt):
        return "[]"

    mem = DummyMemory()
    extractor = FactExtractor(llm_caller=fake_llm, memory_store=mem)

    facts = extractor.extract_from_turn(
        user_text="What time is it in Tokyo?",
        assistant_text="It is currently 6:45 PM in Tokyo.",
    )

    assert facts == []

