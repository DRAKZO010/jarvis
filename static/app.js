/* =============================================================================
   J.A.R.V.I.S. IRON MAN HUD - APP
   ============================================================================= */

(function () {
  "use strict";

  // Same-origin WebSocket: works on any port the server is bound to.
  const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  let ws = null;
  let reconnectTimer = null;
  let lastNet = null;
  let lastNetAt = 0;
  const RECONNECT_DELAY = 3000;

  // DOM refs
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const clockEl = $("#clock");
  const dateEl = $("#date");
  const statusText = $("#status-text");
  const conversation = $("#conversation");
  const taskList = $("#task-list");
  const cmdInput = $("#cmd-input");
  const cmdSend = $("#cmd-send");
  const systemStatus = $("#system-status");

  // Clock
  function updateClock() {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString("en-US", { hour12: false });
    dateEl.textContent = now.toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
    }).toUpperCase();
  }
  setInterval(updateClock, 1000);
  updateClock();

  // Metrics
  function setBar(id, pct, val) {
    const bar = $(`#${id}-bar`);
    const valEl = $(`#${id}-val`);
    if (!bar || !valEl) return;
    const clamped = Math.max(0, Math.min(100, Number(pct) || 0));
    bar.style.width = clamped + "%";
    valEl.textContent = val;

    bar.classList.remove("warning", "danger");
    if (clamped > 90) bar.classList.add("danger");
    else if (clamped > 75) bar.classList.add("warning");
  }

  function setCircle(id, pct, val) {
    const el = $(`#${id}`);
    if (!el) return;
    const clamped = Math.max(0, Math.min(100, Number(pct) || 0));
    const fill = el.querySelector(".fill");
    const valEl = el.querySelector(".cm-value");
    if (fill) fill.style.strokeDashoffset = 100 - clamped;
    if (valEl) valEl.textContent = val;
  }

  function updateMetrics(data) {
    // Server sends *_pct names; keep the aliases so both shapes work.
    const cpu = data.cpu_pct ?? data.cpu ?? 0;
    const ram = data.ram_pct ?? data.ram ?? 0;
    const disk = data.disk_pct ?? data.disk ?? 0;
    const hasBattery = data.battery !== null && data.battery !== undefined;
    const bat = hasBattery ? data.battery : 0;

    // The server sends cumulative network counters, so derive a rate here.
    const now = Date.now();
    const total = (Number(data.net_recv) || 0) + (Number(data.net_sent) || 0);
    let netKBps = 0;
    if (lastNet !== null && now > lastNetAt) {
      const deltaBytes = total >= lastNet.total ? total - lastNet.total : 0;
      netKBps = deltaBytes / 1024 / ((now - lastNetAt) / 1000);
    }
    lastNet = { total: total, at: now };
    lastNetAt = now;
    const netKB = Math.max(0, Math.round(netKBps));
    const netPct = Math.min(100, netKB / 10);

    setBar("cpu", cpu, Math.round(cpu) + "%");
    setBar("ram", ram, Math.round(ram) + "%");
    setBar("disk", disk, Math.round(disk) + "%");
    setBar("net", netPct, netKB + " KB/s");
    setBar("bat", bat, hasBattery ? Math.round(bat) + "%" : "N/A");

    setCircle("cm-cpu", cpu, Math.round(cpu) + "%");
    setCircle("cm-ram", ram, Math.round(ram) + "%");
    setCircle("cm-disk", disk, Math.round(disk) + "%");
    setCircle("cm-net", netPct, netKB + " KB/s");
    setCircle("cm-bat", bat, hasBattery ? Math.round(bat) + "%" : "N/A");

    if (data.uptime) {
      const u = $("#uptime");
      if (u) u.textContent = data.uptime;
    }
    if (data.process_count) {
      const p = $("#process-count");
      if (p) p.textContent = data.process_count;
    }
  }

  // Conversation
  function addMessage(speaker, text, append) {
    const empty = conversation.querySelector(".empty-text");
    if (empty) empty.remove();

    // For streaming append: update the last JARVIS message instead of creating new
    if (append && speaker === "jarvis") {
      const lastMsg = conversation.querySelector(".conv-msg.jarvis:last-child");
      if (lastMsg) {
        const textEl = lastMsg.querySelector(".text");
        if (textEl) {
          textEl.textContent += " " + text;
          conversation.scrollTop = conversation.scrollHeight;
          return;
        }
      }
    }

    const div = document.createElement("div");
    div.className = `conv-msg ${speaker}`;
    div.innerHTML = `<div class="speaker ${speaker}">${speaker === "user" ? "YOU" : "J.A.R.V.I.S."}</div><div class="text">${escapeHtml(text)}</div>`;
    conversation.appendChild(div);
    conversation.scrollTop = conversation.scrollHeight;

    // Limit messages
    while (conversation.children.length > 50) {
      conversation.removeChild(conversation.firstChild);
    }
  }

  // Tasks
  function updateTasks(tasks) {
    if (!tasks || tasks.length === 0) {
      taskList.innerHTML = '<div class="empty-text">No pending tasks</div>';
      return;
    }
    taskList.innerHTML = "";
    tasks.forEach((t) => {
      const div = document.createElement("div");
      div.className = "task-item" + (t.done ? " done" : "");
      // Server sends `description`; accept `text` for compatibility.
      const label = t.description || t.text || "";
      const due = t.due ? ` <span class="task-due">(${escapeHtml(t.due)})</span>` : "";
      div.innerHTML =
        `<span class="task-id">#${escapeHtml(String(t.id ?? "?"))}</span>` +
        `<span class="task-text">${escapeHtml(label)}${due}</span>`;
      taskList.appendChild(div);
    });
  }

  // WebSocket
  function connect() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log("[JARVIS] WS connected");
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        handleEvent(msg);
      } catch (e) {
        console.error("[JARVIS] Parse error:", e);
      }
    };

    ws.onclose = () => {
      console.log("[JARVIS] WS disconnected, reconnecting...");
      reconnectTimer = setTimeout(connect, RECONNECT_DELAY);
    };

    ws.onerror = (err) => {
      console.error("[JARVIS] WS error:", err);
      ws.close();
    };
  }

  function handleEvent(msg) {
    switch (msg.type) {
      case "transcript":
        addMessage(
          msg.speaker === "user" ? "user" : "jarvis",
          msg.text,
          msg.append || false
        );
        break;
      case "status":
        setStatus(msg.state || msg.status);
        break;
      // Authoritative speaking state, needed for barge-in.
      case "speaking":
        isSpeaking = !!msg.speaking;
        if (isSpeaking) setStatus("speaking");
        else if (micStatus && micEnabled) setStatus("listening");
        break;
      case "metrics":
        updateMetrics(msg);
        break;
      case "tasks":
        updateTasks(msg.tasks);
        break;
      // Server-side Vosk heard an utterance.
      case "heard":
        console.log("[JARVIS] Vosk heard:", msg.text);
        break;
      // The server's view of push-to-talk, so the indicator stays in step even
      // if a key release was missed.
      case "ptt":
        applyPttState(!!msg.active, true);
        break;
    }
  }

  function sendCommand(text) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      // Offline: show it anyway so the user is not left guessing.
      addMessage("user", text);
      addMessage("jarvis", "I am not connected to the core, sir.");
      return;
    }
    ws.send(JSON.stringify({ type: "command", text: text }));
  }

  function sendInterrupt() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "interrupt" }));
    }
  }

  // Input
  function handleSend() {
    const text = cmdInput.value.trim();
    if (!text) return;
    sendCommand(text);
    cmdInput.value = "";
  }

  cmdInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleSend();
  });
  cmdSend.addEventListener("click", handleSend);

  // Nav buttons
  $$(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".nav-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const cmd = btn.dataset.cmd;
      if (cmd) {
        if (cmd === "search for ") {
          const q = prompt("Search for:");
          if (q) sendCommand("search for " + q);
        } else {
          sendCommand(cmd);
        }
      }
    });
  });

  // Helpers
  function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }

  // ===== SPEECH RECOGNITION (STT) =====
  let recognition = null;
  let isListening = false;
  let isSpeaking = false;
  let micEnabled = false;

  const micBtn = $("#mic-btn");
  const micStatus = $("#mic-status");

  function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.log("[JARVIS] Speech Recognition not supported");
      if (micStatus) micStatus.textContent = "NOT SUPPORTED";
      return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      isListening = true;
      console.log("[JARVIS] Listening...");
      if (micStatus) micStatus.textContent = "LISTENING";
    };

    recognition.onresult = (event) => {
      let interimTranscript = "";
      let finalTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      // Barge-in: user started talking while JARVIS is speaking.
      if (isSpeaking && (interimTranscript.length > 3 || finalTranscript.trim().length > 3)) {
        console.log("[JARVIS] Interrupt detected!");
        sendInterrupt();
        isSpeaking = false;
      }

      // Process final result. The server echoes the transcript back, so we do
      // not render it here as well - that would duplicate every user message.
      if (finalTranscript.trim()) {
        const text = finalTranscript.trim();
        console.log("[JARVIS] User said:", text);
        sendCommand(text);
      }
    };

    recognition.onerror = (event) => {
      console.log("[JARVIS] Speech error:", event.error);
      if (event.error === "not-allowed") {
        if (micStatus) micStatus.textContent = "BLOCKED";
        micEnabled = false;
        micBtn.classList.remove("active");
      }
    };

    recognition.onend = () => {
      isListening = false;
      // Auto-restart if mic is still enabled
      if (micEnabled) {
        setTimeout(() => {
          if (recognition && micEnabled) {
            try {
              recognition.start();
            } catch (e) {
              console.log("[JARVIS] Restart error:", e);
            }
          }
        }, 300);
      }
    };
  }

  function toggleMic() {
    if (!recognition) {
      initSpeechRecognition();
      if (!recognition) return;
    }

    if (micEnabled) {
      // Turn off
      micEnabled = false;
      recognition.stop();
      micBtn.classList.remove("active");
      if (micStatus) micStatus.textContent = "MIC OFF";
    } else {
      // Turn on
      micEnabled = true;
      try {
        recognition.start();
        micBtn.classList.add("active");
        if (micStatus) micStatus.textContent = "STARTING...";
      } catch (e) {
        console.log("[JARVIS] Start error:", e);
        if (micStatus) micStatus.textContent = "ERROR";
        micEnabled = false;
      }
    }
  }

  // Mic button click
  if (micBtn) {
    micBtn.addEventListener("click", toggleMic);
  }

  // --- Hold-to-talk (server-side Vosk) -------------------------------------
  // The mic button above uses the browser's SpeechRecognition API. This is the
  // other path: Vosk running in JARVIS itself, which needs no browser support
  // and works from any client. Holding the key opens the gate; releasing it
  // transcribes immediately instead of waiting for a pause.
  const PTT_KEY = "t";
  let pttActive = false;

  function sendPtt(active) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ptt", active: active }));
    }
  }

  // `fromServer` updates arrive from the server's own echo, so they must not be
  // sent back or the two would trade messages forever.
  function applyPttState(active, fromServer) {
    if (pttActive === active) return;
    pttActive = active;
    if (!fromServer) sendPtt(active);
    document.body.classList.toggle("ptt", active);
    if (statusText) statusText.textContent = active ? "TRANSMITTING" : "STANDBY";
    if (systemStatus) {
      systemStatus.textContent = active ? "TRANSMITTING" : "ONLINE";
    }
    // Two recognisers listening at once would dispatch every phrase twice.
    if (active && micEnabled && recognition) {
      recognition.stop();
    }
  }

  function setPtt(active) {
    applyPttState(active, false);
  }

  document.addEventListener("keydown", (e) => {
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (e.key && e.key.toLowerCase() === PTT_KEY) {
      e.preventDefault();
      setPtt(true);
    }
  });

  document.addEventListener("keyup", (e) => {
    if (e.key && e.key.toLowerCase() === PTT_KEY) setPtt(false);
  });

  // Losing focus mid-press would otherwise leave the gate stuck open.
  window.addEventListener("blur", () => setPtt(false));

  // Report whether the server-side recogniser is actually available, so the
  // key does not appear to do nothing on a machine with no microphone.
  async function probeListening() {
    try {
      const res = await fetch("/api/listening");
      if (!res.ok) return;
      const info = await res.json();
      if (micStatus && !micEnabled) {
        micStatus.textContent = info.listening ? "SR READY" : "SR OFF";
      }
      if (info.listen_error) {
        console.log("[JARVIS] Server STT unavailable:", info.listen_error);
      }
    } catch (e) {
      /* offline: the WebSocket path already reports the problem */
    }
  }

  // Track speaking status
  function setStatus(status) {
    status = status || "standby";
    document.body.classList.remove("speaking", "listening", "processing");
    const map = {
      speaking: "speaking",
      listening: "listening",
      processing: "processing",
    };
    if (map[status]) document.body.classList.add(map[status]);

    const labels = {
      standby: "STANDBY",
      online: "ONLINE",
      listening: "LISTENING",
      processing: "PROCESSING",
      speaking: "SPEAKING",
    };
    const label = labels[status] || String(status).toUpperCase();
    if (statusText) statusText.textContent = label;
    if (systemStatus) systemStatus.textContent = labels[status] || "ONLINE";

    // Track speaking state for interruption
    if (status === "speaking") {
      isSpeaking = true;
    } else if (status === "listening") {
      isSpeaking = false;
    }
  }

  // Boot
  connect();
  setStatus("standby");
  initSpeechRecognition();
  probeListening();
})();
