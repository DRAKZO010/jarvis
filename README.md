# Jarvis

A local voice assistant for Windows. Speaks, listens for the wake word, and
carries out commands across your system, your apps, and the web.

The UI is a browser-based HUD served by a FastAPI app; the assistant core is a
set of MCP modules that can be used on their own.

---

## Requirements

- Windows 10 or 11 (window and clipboard control are Windows-specific)
- Python 3.11+
- A microphone
- A Vosk speech model in `model/` (see [Speech recognition](#speech-recognition))

## Install

```powershell
git clone <your-fork-url> jarvis
cd jarvis

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
copy .env.example .env      # then edit .env and add your keys
```

## Run

```powershell
.\venv\Scripts\python.exe jarvis.py
```

This runs one process, which is what you want. Windows lets several processes
open the same capture device at once, so if you start the app yourself with
`uvicorn jarvis:app --workers N`, every worker opens the microphone
successfully and there is no warning: all of them hear the same speech, and
each runs your command once. The two workers also keep separate copies of the
WebSocket clients and the task list, so they will disagree with each other.

Keep it to a single worker. If you need several, set `JARVIS_DISABLE_MIC=1`
everywhere except the one worker that should hear you.

Then open <http://127.0.0.1:8765>. To change the port, edit `PORT` at the top
of `jarvis.py`.

Jarvis is useful without any API keys: system, file, window, and task commands
all work offline. Keys unlock the rest.

| Key | Unlocks |
| --- | --- |
| `GOOGLE_API_KEY` | Free-form conversation, weather, news, and natural-language command parsing (Gemini) |
| `FISH_API_KEY` | Speech. Required — there is no fallback voice |

Without `GOOGLE_API_KEY` Jarvis still works: every pattern-matched command runs
offline, and unmatched speech falls back to a brief spoken prompt instead of
routing through a model. The Gemini free tier is limited to 20 requests per
day, which is comfortable for occasional phrased commands but tight for
continuous conversation; a paid key removes the limit.

## Configuration

All settings live in `.env` — see `.env.example` for the full list with
comments. The two that matter most:

```ini
GOOGLE_API_KEY=...
FISH_API_KEY=...
```

## How it works

```
mic -> Vosk -> wake word -> intent match -> MCP module -> HUD + TTS
                                            |
                                            +-> typo repair
                                            |
                                            +-> unmatched -> Gemini picks a capability
                                                          |
                                                          +-> allowlisted -> dispatch
                                                          +-> not allowed -> conversation
```

### How a command is matched

1. **Patterns.** 207 regular expressions in `jarvis.py` are tried in order and
   the first hit wins. This path is offline, instant, and free, so it handles
   every command it can recognise.
2. **Typo repair.** Obvious speech-recognition errors are corrected, but only
   when the result is also a confident pattern match.
3. **Gemini routing.** Anything still unmatched is offered to Gemini, which
   chooses one action from the catalogue in `mcp/capabilities.py` and returns
   the arguments as a payload, e.g. `{"action": "add_task", "payload": "buy
   milk"}`. Each action carries a description, because a bare list of names
   such as `add_task` and `create_file` is not enough to choose between.
4. **Allowlist.** The model's choice is checked against
   `LLM_ROUTABLE_ACTIONS`, which is deny-by-default. Anything that runs code,
   synthesises keyboard or mouse input, destroys data, or changes the machine's
   power state is not offered to the model at all; those actions stay reachable
   only through their own explicit patterns. A blocked pick is dropped and the
   utterance becomes conversation, so the user still gets a reply.
5. **Conversation.** If nothing is routable, Gemini answers the question using
   the persona in `SOUL.md`.

Routing runs at `temperature=0` so the same phrasing always resolves to the
same action, and decisions are cached in memory: the Gemini free tier allows
**20 requests per day**, not per minute, so a repeated phrase does not spend
another one. Adding an action to the command table without describing it in the
catalogue fails the test suite, which keeps the two in step.

Because that budget is per project and per model rather than per conversation,
a separate chat window would not raise it. Two things do cut the cost instead.
Utterances that plainly read as questions skip the routing request and go
straight to the answer, halving what each question costs, and repeat phrasings
are served from the cache.

| Utterance | Requests |
| --- | --- |
| Matches the pattern table, e.g. "open notepad" | 0 |
| Matches a question, e.g. "how do I reverse a linked list" | 1 (the answer) |
| Needs routing, e.g. "put the kettle on for 4 minutes" | 1 (the decision) |

`jarvis.py` owns the FastAPI app, the WebSocket that pushes updates to the
HUD, and the command table. Everything else lives in `mcp/`:

| Module | Responsibility |
| --- | --- |
| `capabilities.py` | Action descriptions, LLM routing prompt, safety allowlist |
| `voice.py` | Speech recognition, TTS, barge-in |
| `llm.py` | Google Gemini client, rate-limit and quota handling |
| `system.py` | CPU, RAM, disk, network, battery, uptime |
| `tasks.py` | Task list, persisted to `jarvis_tasks.json` |
| `memory.py` | Long-term facts you ask it to remember |
| `context.py` | Recent conversation, for follow-up questions |
| `reminders.py` | Scheduled reminders |
| `automation.py` | Launchers, window control, mouse, volume, power |
| `files.py` | Read, create, move, and search files |
| `browser.py` | Open URLs, web search, YouTube, and Spotify control |
| `weather.py` | Forecast lookup |
| `news.py` | News headlines |

### Voice output

**Fish Audio is the only speech backend.** If it is unavailable or out of API
credit, Jarvis is silent rather than quietly speaking in a voice you did not
choose. That keeps the spoken voice predictable, at the cost of no audio at all
when Fish is down — a deliberate trade.

Fish bills API credit separately from a normal subscription, so a paid plan can
still return HTTP 402; see <https://fish.audio/app/developers>.

A barge-in never counts as a backend failure.

Because there is only one backend, `"set voice backend edge"` is answered rather
than performed: Jarvis says which voice it is actually using instead of
accepting the change and then speaking in the old one anyway. `"set voice X"`
sets `FISH_VOICE_ID` for the rest of the session.

### Voice input

Speech is transcribed locally by Vosk, so nothing leaves the machine. There are
two ways to give it a command:

| Mode | How | Gate |
| --- | --- | --- |
| Wake word | Say `"Jarvis, what time is it"` | The wake word must be spoken |
| Push-to-talk | Hold `T` in the HUD, speak, release | None |

Push-to-talk transcribes the moment you let go, so there is no pause to wait
for. Speech that is not a command is discarded, and so is anything Jarvis says
itself, so it cannot answer the room.

The mic button in the HUD uses the browser's own `SpeechRecognition` API, which
is a separate path. Only one of the two runs at a time.

Set `JARVIS_DISABLE_MIC=1` to keep the microphone closed — on a headless box, or
where another application owns the capture device. Everything else keeps
working; only the microphone is disabled.

### Choosing a microphone

`JARVIS_INPUT_DEVICE` takes either a device index or any part of a device name:

```powershell
JARVIS_INPUT_DEVICE=Realtek
JARVIS_INPUT_DEVICE=7
```

Leave it unset and JARVIS picks for you: it probes the Windows default and, if
that input turns out to be silent, falls back to the loudest one it can find.
That matters on a machine with several headsets and virtual audio devices,
where the default is often a dead one. An override that matches nothing is an
error — JARVIS reports it rather than quietly listening on a different
microphone than the one you asked for.

To see what is available:

```powershell
.\venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices())"
```

### Choosing where replies play

`JARVIS_OUTPUT_DEVICE` takes either a device index or any part of a device name:

```powershell
JARVIS_OUTPUT_DEVICE=Stone 358
JARVIS_OUTPUT_DEVICE=19
```

Leave it unset and JARVIS prefers a headset or pair of earbuds that is actually
connected, and otherwise uses the Windows default, which is normally the built-in
speakers. The connected test matters more than it sounds: Windows keeps every
Bluetooth headset you have ever paired in the device list, so choosing by name
alone would send speech to earbuds nobody is wearing, which is indistinguishable
from a broken speaker. A device is only used if it also opens successfully.

The choice is re-checked every few seconds, so plugging earbuds in partway
through a conversation is picked up without a restart. An override that matches
nothing is reported and then ignored, so a typo does not silence the assistant.

`/health` reports the device currently in use, so "why can I not hear it" is one
request away rather than a log dive:

```powershell
curl http://127.0.0.1:8765/health
```

The two causes that actually happen are a muted or near-silent output volume and
a device the user is not wearing; both are visible in that field.

### When the microphone is quiet or the room is loud

At startup JARVIS measures the room and places the speech gate between the
silence and the voice, because a fixed threshold is wrong on every machine but
one. It reads a low percentile of the block levels rather than the peak or the
median: a peak measures a cough or a keystroke, and the median measures the user,
since the probe window is often sampled while they are still speaking.

The gate is never allowed to sit below the measured room noise, because a gate
under the noise reads every breath as speech and JARVIS then talks over itself
and never finishes an utterance. In a room too loud for that to be possible, the
gate rises above its normal ceiling and `/health` returns a `vad_warning`
explaining it, rather than failing quietly.

Detection and recognition want different levels, and JARVIS applies them
separately. The gate runs on the raw signal. The audio handed to Vosk is lifted
by a smoothed automatic gain control, because a gate can be low enough to admit
a voice that is still far too quiet to recognise — a laptop array at arm's length
peaks near 0.004 RMS, roughly −48 dBFS, and Vosk returns nothing for that.

`/health` also reports `noise_floor`, `vad_threshold` and `speaking_threshold`, so
the three numbers behind any recognition problem are visible.

`JARVIS_VAD_THRESHOLD` overrides the measured value with a hand-set one, by index
or name like the device settings above. It is a last resort: the calibrator
normally gets this right, and a hand-set value does not follow the room.

If recognition is unreliable, the cause is nearly always the input level rather
than the code. Check that the Windows input level is unmuted, and prefer a
headset microphone over the built-in laptop array. Do not simply turn the input
level to 100%: a weak preamp amplifies its own hiss along with the voice, and on
the array here the room tone measured 0.00001 at 0% gain, 0.0019 at 50% and
0.027 at 100%, while the voice barely improved. The gain that maximises the
voice over the hiss is well below maximum.

### Barge-in

Say the wake word, or press the interrupt button in the HUD, and playback stops
immediately. The queue is drained so JARVIS does not talk over you, and the
next reply starts clean.

## Tests

```powershell
pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pytest
```

The suite is offline and repeatable: no network calls, no audio playback, no
microphone, and no writes to your real task or memory files. The voice and LLM
tests use fakes in place of the providers.

To point it at the live providers instead, use your own key in `.env` and run
the app.

## Speech recognition

JARVIS uses [Vosk](https://alphacephei.com/vosk/models). Download a model and
extract it so that you end up with `model/` in the project root:

```powershell
.\venv\Scripts\python.exe -c "from vosk import Model; Model('model')"
```

**Use the full desktop model, not a mobile one.** `vosk-model-en-us-0.22` is
around 1.9 GB; the mobile `vosk-model-small-en-us-0.15` is 40 MB. The small model
is built for phones and is markedly worse on exactly the input a laptop
microphone produces: quiet, noisy, and at a distance. On a built-in array it
transcribes a clearly spoken command as fragments.

If `model/` is missing, JARVIS still starts and every typed command still works —
only the microphone is disabled.

### Diagnosing recognition

"it did not hear me" has three quite different causes, and `/health` separates
them. Look at `input_levels` first:

- `above_gate` not rising — nothing is reaching the gate. The gate is too high
  or the microphone is dead; compare `median` against `vad_threshold`.
- `above_gate` rising but `utterances` staying flat — the gate opens but the
  utterance never closes, so no pause was found. The room is too continuous.
- `utterances` rising with `last_transcript` empty — detection works and
  recognition is the problem. That is the model, not the level.
- `gain` pinned at its maximum — the audio needed everything the AGC had, which
  means the microphone is very quiet. A headset microphone beats turning the
  input level up, because a high gain amplifies the preamp's own hiss along
  with the voice.

Captured audio is analysed in 30 ms frames, which is what speech detectors are
built around. A quarter-second frame averages a syllable together with the
pauses around it, and the peaks that detection depends on are averaged away.

## Project layout

```
jarvis.py          FastAPI app, HUD routes, command table
mcp/capabilities.py Action descriptions, routing prompt, safety allowlist
mcp/               assistant modules
static/            HUD (index.html, style.css, app.js)
tests/             offline test suite
model/             Vosk model (not in git)
logs/              runtime logs (not in git)
```

Runtime state such as `jarvis_tasks.json`, `jarvis_reminders.json`, and
`jarvis_memory.json` is personal and is not tracked by git.
