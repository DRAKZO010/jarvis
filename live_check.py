"""End-to-end check against a real server process.

The earlier audit imported the app and drove it through TestClient, which does
not exercise the process that actually runs: the uvicorn server, the real
WebSocket, and the live LLM path with a real key. This starts uvicorn on a
separate port, talks to it over TCP, and reads what comes back.

Only read-only and reversible commands are sent. Nothing here shuts down,
deletes, kills, or types.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8791
BASE = f"http://127.0.0.1:{PORT}"

os.environ["JARVIS_DISABLE_MIC"] = "1"
os.environ["JARVIS_NO_BROWSER"] = "1"

results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}   {name:<34} {detail}"[:200], flush=True)


def free_port_in_use():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def http_get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return r.status, r.read().decode("utf-8", "replace")


def wait_for_server(deadline=90):
    end = time.time() + deadline
    while time.time() < end:
        if free_port_in_use():
            try:
                http_get("/health")
                return True
            except Exception:
                pass
        time.sleep(0.5)
    return False


LAUNCH = (
    "import uvicorn, os; "
    "os.environ['JARVIS_DISABLE_MIC']='1'; "
    "import jarvis; "
    f"uvicorn.run(jarvis.app, host='127.0.0.1', port={PORT}, log_level='warning')"
)

# Safe, read-only utterances. No shutdown, no delete, no kill, no typing.
COMMANDS = [
    ("pattern: system status", "what is your system status"),
    ("pattern: list tasks", "list my tasks"),
    ("pattern: time", "what time is it"),
    ("llm: free-form question", "in one sentence, why is the sky blue?"),
    ("llm: unfamiliar phrasing", "could you jot down that I need to water the plants"),
]


async def ws_session(commands):
    import websockets

    seen = []
    uri = f"ws://127.0.0.1:{PORT}/ws"
    async with websockets.connect(uri, max_size=4 * 1024 * 1024) as ws:
        # Drain the broadcast the server pushes on connect.
        try:
            await asyncio.wait_for(ws.recv(), timeout=3)
        except asyncio.TimeoutError:
            pass

        for label, text in commands:
            before = len(seen)
            await ws.send(json.dumps({"type": "command", "text": text}))
            # Give the assistant time to answer; a live LLM turn is slow.
            end = time.time() + 45
            while time.time() < end and len(seen) == before:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=45)
                except asyncio.TimeoutError:
                    break
                seen.append(json.loads(raw))
            record(
                f"ws: {label}",
                len(seen) > before,
                _summarise(seen[before:]) if len(seen) > before else "no reply",
            )
    return seen


def _summarise(messages):
    parts = []
    for m in messages:
        if m.get("type") == "transcript":
            parts.append(str(m.get("text", ""))[:90])
        elif m.get("type") == "response":
            parts.append(str(m.get("text", ""))[:90])
        elif m.get("type") == "error":
            parts.append("ERROR: " + str(m.get("message", ""))[:80])
    return " | ".join(parts) if parts else "(no text)"


def main():
    if free_port_in_use():
        record("server port free", False, f"port {PORT} already in use")
        return 1

    print("starting uvicorn in a real process...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-c", LAUNCH],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if not wait_for_server():
            record("server starts", False, "no answer from /health")
            proc.terminate()
            return 1
        record("server starts", True, f"pid {proc.pid} on {BASE}")

        for path, check in (
            ("/", lambda b: "<html" in b.lower()),
            ("/health", lambda b: json.loads(b).get("status") == "online"),
            ("/api/metrics", lambda b: "cpu_pct" in b),
            ("/api/tasks", lambda b: "task" in b.lower()),
            ("/style.css", lambda b: len(b) > 0),
            ("/app.js", lambda b: len(b) > 0),
        ):
            try:
                status, body = http_get(path)
                ok = status == 200 and check(body)
                record(f"http: {path}", ok, f"{status}, {len(body)} bytes")
            except Exception as e:
                record(f"http: {path}", False, repr(e))

        try:
            status, body = http_get("/health")
            health = json.loads(body)
            llm = health.get("modules", {}).get("llm", health.get("llm", {}))
            record(
                "health reports llm state",
                bool(llm),
                json.dumps({
                    k: llm.get(k)
                    for k in ("model", "active_model", "available",
                              "exhausted_models", "daily_requests_used")
                    if k in llm
                }),
            )
        except Exception as e:
            record("health reports llm state", False, repr(e))

        try:
            status, body = http_get("/does-not-exist")
            record("404 for unknown path", status == 404, f"status {status}")
        except urllib.error.HTTPError as e:
            record("404 for unknown path", e.code == 404, f"status {e.code}")
        except Exception as e:
            record("404 for unknown path", False, repr(e))

        print("running websocket commands...", flush=True)
        asyncio.run(ws_session(COMMANDS))

        record("server still alive", proc.poll() is None, f"poll={proc.poll()}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    failed = [n for n, ok, _ in results if not ok]
    print()
    print("=" * 70)
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
