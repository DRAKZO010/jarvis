"""Which aliases actually work, and how many independent daily budgets exist?

The response echoes the resolved concrete model, which is the unit the free-tier
daily quota is counted against.
"""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
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

CANDIDATES = [
    "gemini-flash-lite-latest",
    "gemini-3.8-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
]

for model in CANDIDATES:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK."}],
            temperature=0.0,
            max_tokens=32,
        )
        print("OK   %-28s -> resolved %-28s content=%r" % (
            model, getattr(r, "model", "?"), r.choices[0].message.content))
    except Exception as e:
        msg = str(e)
        if "429" in msg:
            resolved = "?"
            if "model: " in msg:
                resolved = msg.split("model: ", 1)[1].split("\n")[0].split(" ")[0]
            quota = "?"
            if "quotaValue" in msg:
                quota = msg.split("quotaValue', '")[1].split("'")[0] if "quotaValue', '" in msg else "?"
            print("429  %-28s -> resolved %-28s quota=%s" % (model, resolved, quota))
        else:
            first = msg.split("message': '")[-1][:90]
            print("FAIL %-28s -> %s" % (model, first))
