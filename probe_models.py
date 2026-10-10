"""List the models this key can actually use, via the OpenAI-compat endpoint."""
import json
import os
import sys
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
for line in (root / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

key = os.environ["GOOGLE_API_KEY"]
url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"
req = urllib.request.Request(url, headers={"x-goog-api-key": key})
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.load(r)

print("total models:", len(data.get("models", [])))
print()
for m in data.get("models", []):
    name = m.get("name", "")
    methods = ",".join(m.get("supportedGenerationMethods", []))
    if "generateContent" not in methods:
        continue
    print("%-46s in=%s out=%s" % (name, m.get("inputTokenLimit"), m.get("outputTokenLimit")))
