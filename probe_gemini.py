"""Probe the real Gemini endpoint and print the exact failure, if any."""
import os
import sys
import traceback
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

# Minimal .env loader so the probe uses the same configuration as the app.
for line in (root / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"'))

print("model:", os.environ.get("LLM_MODEL"))
print("max_tokens:", os.environ.get("LLM_MAX_TOKENS"))
print("key present:", bool(os.environ.get("GOOGLE_API_KEY")))

from openai import OpenAI

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
    timeout=30.0,
)

for model in (os.environ.get("LLM_MODEL"), "gemini-flash-latest",
              "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-lite-latest"):
    print("\n=== %s ===" % model)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK."}],
            temperature=0.0,
            max_tokens=64,
        )
        c = r.choices[0].message.content
        print("OK  content=%r" % (c,))
        print("    usage=%s" % (r.usage,))
    except Exception as e:
        print("FAIL %s: %s" % (type(e).__name__, e))
        body = getattr(getattr(e, "response", None), "text", None)
        if body:
            print("    body=%s" % body[:800])
