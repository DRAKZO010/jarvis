"""Confirm the quota is per-model, and that a second model has its own budget."""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
for line in (root / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

from openai import OpenAI

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
    timeout=30.0,
)

ROUTING = (
    'Reply with ONLY JSON: {"action":"add_task","payload":"buy milk"}\n'
    'Allowed: chat, add_task, time, weather\n'
    'User said: "remind me to get milk"'
)

for model in ("gemini-flash-lite-latest", "gemini-3.8-flash-lite",
              "gemini-flash-lite", "gemini-flash-lite-preview"):
    print("\n=== %s ===" % model)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": ROUTING}],
            temperature=0.0,
            max_tokens=800,
        )
        c = r.choices[0].message.content
        print("OK   content=%r" % (c,))
        print("     usage: prompt=%s completion=%s total=%s" % (
            r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens))
        print("     finish=%r" % r.choices[0].finish_reason)
    except Exception as e:
        print("FAIL %s" % str(e)[:300])
