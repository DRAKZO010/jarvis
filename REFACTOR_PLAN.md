# Jarvis — Test Report, Log Analysis, and Refactor / Upgrade Proposal

Status: proposal for review. Nothing here has been applied to the code.
Sources: `tests/` suite run, `logs/*.log`, root-level `*.log`, and the current
working tree (`git status` shows uncommitted edits to `mcp/llm.py`,
`mcp/voice.py`, `README.md`, `.env.example`, `tests/`).

---

## 1. Test suite result

```
platform win32 -- Python 3.13.14, pytest-9.1.1
collected 449 items
449 passed in 468.03s (0:07:48)
```

All green. Coverage is broad (routing, LLM, STT, voice, app API, system/tasks).
The suite is offline, mutes the mic, and disables network — good. It does **not**
exercise: real TTS backend failure/fallback, real provider 402/429/503 paths at
the integration level, WebSocket disconnect handling, or startup with a missing
`model/` + missing key simultaneously. Those gaps line up with every failure
below, which is why the bugs survived 449 passing tests.

---

## 2. Log analysis — recurring faults and loopholes

Counts from `logs/jarvis.log` (3,361 lines): 100 `ERROR`, 110 `WARNING`.

### E1. LLM daily-quota exhaustion storms (highest impact)
- ~70 lines of `Error code: 402 / 429 Resource_Exhausted` across 2026-07-10 →
  2026-09-28. The free Gemini tier is capped at **20 requests/day**.
- The client rotates models (`gemini-flash-latest` → lite models), but the log
  shows one long burst (lines 2297–2375) where **every retry returns 429 within
  ~3s of the previous one**, ignoring the provider's own `retryDelay` (13s, 9s,
  58s…). That is a retry storm against a *daily* cap being mistaken for a
  transient burst.
- `mcp/llm.py` already classifies `daily` vs `burst` and gates routing
  (`_routing_budget_gate`). **Loophole:** `ask_stream()` (the path used by the
  chat handler in `jarvis.py:1281`) does **not** apply the routing reserve gate
  and does **not** honor `_penalty_until` between models — `_check_cooldown(False)`
  is called once up front, then the loop immediately re-fires on the next model.
  A daily cap shared per-project (not per-model) is therefore hammered.

### E2. TTS backend churn and permanent silence
- History shows three generations: `kittentts` (`espeak not installed`,
  `'Jasper is not a file in the archive'`), `edge-tts` (`No audio was received`),
  then Fish Audio (`HTTP 402: Insufficient API credit`, disabled permanently).
- The current tree (working copy) has **removed the fallback chain by design** —
  `mcp/voice.py:460 _build_tts_chain()` now puts only `fish-audio` in the order.
  This matches `README.md` but **contradicts the historical log behavior** and
  the still-present `_disable_backend` / `_active_backend` machinery and
  `_TTS_BACKENDS` dict. Dead code: the "fallback" language in docstrings and the
  `disabled backends` status fields no longer have more than one backend to fall
  back to.
- **Loophole:** with Fish-only, a single 402 makes Jarvis silent for the whole
  session with no HUD-visible persistent banner. `/health` reports it, but the
  user gets no spoken/visual "voice is off" at the moment it breaks.

### E3. `No module named 'openai'` at runtime (`stdout.log:50`, 61)
- `mcp/llm.py` guards with `_import_ok()`/`_get_client()`, and correctly returns
  `_offline_message()`. But `jarvis.py:356 _try_self_heal` and the routing path
  log it as `ERROR - LLM parse error: No module named 'openai'` from
  `_llm_parse_command` — i.e. it reaches the LLM path even though the client is
  known-missing. The guard exists; the caller still spends a call attempt.

### E4. WebSocket close codes logged as errors
- 54× `WebSocket error: (1006, '')`, 26× `(1005, None)`, several `(1001, '')`.
- All are normal browser-tab closes / reloads. They are caught as generic
  exceptions at `jarvis.py:1765` and logged at WARNING. **Loophole:** real
  failures are indistinguishable from tab reloads; the log is noisy and buries
  genuine faults.

### E5. Microphone selection instability
- `The default microphone [1] Microphone Array (Realtek(R) Au is silent or
  unavailable (rms=0.00002)` on every startup, then `No microphone carried any
  signal; falling back to the system default.` — i.e. the probe rejects every
  device and lands back on the default it just rejected.
- One-off: `Microphone error: initializer for ctype 'char *' must be a cdata
  pointer, not _cffi_backend.buffer` — a sounddevice/ctypes buffer mismatch in
  the capture callback.

### E6. `Action error: nope` (`jarvis.log:3179`)
- A routing result dispatched a callable that raised the bare string `nope`.
  `_record_error`/`_try_self_heal` was not in that path — this came from a
  `_speak_during_act`/`_task_action` wrapper (they log `Action error:`). There is
  no traceback, so the origin is lost. **Loophole:** bare `str` exceptions from
  actions cannot be diagnosed from the log.

### E7. Configuration drift in root-level logs
- `stderr.log`: `ModuleNotFoundError: No module named 'speech_recognition'` from
  an older `jarvis.py` line 35 — a stale artifact/old entry point still in the
  tree (`jarvis_hud.py` is untracked). Confirms two app entry points coexist.
- `debug.log`/`stdout.log` show the model was `qwen/qwen-2.5-7b-instruct`
  (OpenRouter) before switching to Gemini; env/`.env` history is inconsistent
  with `requirements.txt`, which only pins `openai` for Gemini.

---

## 3. Root-cause themes

1. **Quota is treated as retryable.** A per-day cap is hit with per-second
   retries. The fix is classification + a hard gate on *every* LLM entry point,
   not just `ask(purpose="route")`.
2. **Single-provider hard dependency (TTS).** Deliberate, but with no graceful
   degradation the failure mode is total silence and a confusing UX.
3. **Known-bad states still attempted.** Missing `openai`, dead mic, dead TTS —
   each is detected, then partially re-attempted.
4. **Error taxonomy is weak.** WS closes (normal) and bare-string exceptions
   (real) are logged the same way.
5. **Duplicated entry points / stale code.** `jarvis.py`, `jarvis_hud.py`, old
   `speech_recognition` import, multi-backend TTS remnants.

---

## 4. Proposed updates

### P0 — Correctness / robustness (do first)

**P0.1 Gate the streaming LLM path like `ask()`.**
In `mcp/llm.py::ask_stream`, apply `_routing_budget_gate` semantics and honor
`_penalty_until` *inside* the per-model loop (re-check before each model), and
skip exhausted models on the daily (not just burst) signal. Add a test that a
`daily` classification stops the loop immediately instead of trying every model.

**P0.2 Distinguish WebSocket close codes.**
In `jarvis.py:websocket_endpoint`, special-case `WebSocketDisconnect` and close
codes 1000/1001/1005/1006 as `INFO "client went away"`; keep WARNING only for
unexpected codes. Removes ~80 noise lines per session.

**P0.3 Stop attempting the LLM when the client is known-missing.**
In `_llm_parse_command` and `_try_self_heal`, check `llm_mcp.available` first and
short-circuit to conversation/offline message. Removes E3.

**P0.4 Preserve tracebacks for action errors.**
Change `_speak_during_act` / `_task_action` / `_speak_and_act` to log
`log.exception("Action error in %s", context)` so E6-style bare-string faults
carry a traceback. Wrap all dispatched handlers so a non-`Exception` raise
(e.g. `raise "nope"`) is caught and reported as a real error, not a silent ack.

**P0.5 HUD voice-down banner.**
When `VoiceMCP` has zero live TTS backends, broadcast a one-shot
`{"type":"warning","level":"voice_down"}` so the HUD shows it immediately, in
addition to `/health`.

### P1 — Refactor (structure / dead code)

**P1.1 Collapse the TTS backend machinery to one backend, explicitly.**
Given the Fish-only decision, remove `_backend_order`/`_disabled_backends`/
`_TTS_BACKENDS`/`_disable_backend` and the "fallback" naming, or (preferred)
make the design honest: a single `_tts_speak()` with a typed
`VoiceUnavailable` state. Update the tests that still assert fallback ordering.
This deletes a whole class of confusing dead paths.

**P1.2 One entry point.**
Retire `jarvis_hud.py` (untracked) and remove the stale
`speech_recognition` import path. Document `jarvis.py` as the sole entry point.

**P1.3 Extract routing out of `jarvis.py`.**
`jarvis.py` is 1,935 lines and holds the 207-pattern table, fuzzy repair, LLM
routing, the FastAPI app, and the command dispatch switch (a ~470-line
`if/elif`). Move to `mcp/router.py` (patterns + `_detect_action`) and
`mcp/dispatch.py` (the action switch). `jarvis.py` keeps app wiring only. This is
the single biggest maintainability win.

**P1.4 Replace the `if/elif` dispatch with a table.**
Map `action -> (callable, acknowledgement template, args-adapter)`. Today the
switch mixes arg parsing (`int(payload)`), confirmation text, and side effects;
a table makes each row testable and removes the repeated
`_speak_during_act(func, "Doing X...", payload)` boilerplate.

**P1.5 Structured logging + correlation id.**
Emit JSON logs with `name`, `level`, `event`, and a per-command `cmd_id` so a
single utterance's routing → dispatch → TTS chain can be grepped as one unit.
This directly fixes the "bare string, no traceback" class.

### P2 — Upgrades (capability / UX)

**P2.1 LLM provider abstraction with budget accounting.**
Introduce a `Provider` interface (Gemini today, pluggable tomorrow). Centralize
daily/burst accounting per provider *and* per model, persist the spend to disk so
restart doesn't reset the local guard, and expose budget in `/health`. Keep the
`openai` SDK for the OpenAI-compatible endpoint.

**P2.2 Durable quota state.**
`_daily_spend`/`_exhausted_until` are in-memory and reset on restart, so a
restarted app re-spends the free tier. Persist to `jarvis_llm_budget.json`.

**P2.3 Mic selection: don't reject then fall back to the rejected default.**
Fix the probe so "all devices silent" keeps the *loudest measured* device rather
than the OS default it already measured as dead, and surface the cffi buffer
error path with a clear message.

**P2.4 Configurable wake/PTT and audio device hot-apply.**
Device choice is re-checked every 10s (`OUTPUT_RECHECK_S`) but input is chosen
once. Add input re-probe on the same cadence + a HUD device picker.

**P2.5 Security hardening of `execute`.**
`automation.py:252` and `system.py:210` run `subprocess.run(cmd, shell=True)`
with an arbitrary payload from (a) an explicit voice pattern and (b) LLM routing
is blocked — good. But the pattern table is the only gate. Add a denylist and a
confirmation step for destructive shell metacharacters (`&&`, `|`, `del /f`,
`format`, `shutdown`), and log every execution with the full command at INFO.

**P2.6 HUD + backend version handshake.**
`VERSION = "1.1.0"` is hard-coded; old `debug.log` shows `v1.0.0`. Serve the
version to the HUD and warn on mismatch so stale cached `app.js` is visible.

### P3 — Test-infrastructure upgrades

- Add integration tests (mocked HTTP) for: 402 Fish mid-session, 429 daily vs
  burst distinction driving `ask_stream`, WebSocket 1006 clean-close, action
  raising a non-`Exception`.
- Add a test that fails if an action name in the pattern table is missing from
  `ACTION_CATALOGUE` (README claims this exists; verify it and extend to the
  reverse direction, catalogue ⊄ table).
- Add a "no single provider hard-fails the app" smoke test.

---

## 5. Suggested sequencing

| Order | Items | Rationale |
| --- | --- | --- |
| 1 | P0.1–P0.5 | Stop the bleeding; each is small and test-backed |
| 2 | P1.2, P1.1 | Delete stale entry point + dead TTS code |
| 3 | P1.3, P1.4 | Biggest maintainability win; do behind tests |
| 4 | P1.5, P2.1, P2.2 | Observability + durable quota |
| 5 | P2.3–P2.6, P3 | UX, security, test depth |

Estimated effort: P0 ~1 day, P1 ~2–3 days, P2/P3 ~3–4 days.

---

## 6. Verification plan

- Re-run `.\venv\Scripts\python.exe -m pytest` after each P0/P1 change; keep 449
  green and add the new tests.
- Diff `logs/jarvis.log` line counts for `ERROR`/`WARNING` before vs after; the
  WS-close and quota-storm categories should drop to near zero.
- Point at live providers with a real key and confirm: a forced 402 disables
  voice with a HUD banner; a forced daily-429 stops retries within one attempt.
