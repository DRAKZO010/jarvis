/* =============================================================================
   Jarvis HUD - APP
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
  const micLink = $("#mic-link");
  const panelsBtn = $("#panels-btn");
  const replyCaption = $("#reply-caption");
  const evolutionList = $("#evolution-list");
  const evolveBtn = $("#evolve-btn");

  // The side panels are hidden until the user asks for them. The server owns
  // the state so a reload comes back the way they left it.
  let panelsVisible = false;

  function setPanels(visible) {
    panelsVisible = !!visible;
    document.body.classList.toggle("panels-hidden", !panelsVisible);
    if (panelsBtn) {
      panelsBtn.classList.toggle("on", panelsVisible);
      const label = panelsBtn.querySelector("span");
      if (label) label.textContent = panelsVisible ? "HIDE" : "PANELS";
    }
    // The core moved, so the galaxy has to be measured against it again.
    requestAnimationFrame(resizeGalaxy);
  }

  function setMicLink(text) {
    if (micLink) micLink.textContent = text;
  }

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

    if (data.uptime) {
      const u = $("#uptime");
      if (u) u.textContent = data.uptime;
    }
    if (data.process_count) {
      const p = $("#process-count");
      if (p) p.textContent = data.process_count;
    }
    // The always-on microphone reports itself, so the HUD can say so.
    if (typeof data.listening === "boolean") {
      setMicLink(data.listening ? "LISTENING" : "OFF");
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
    div.innerHTML = `<div class="speaker ${speaker}">${speaker === "user" ? "YOU" : "Jarvis"}</div><div class="text">${escapeHtml(text)}</div>`;
    conversation.appendChild(div);
    conversation.scrollTop = conversation.scrollHeight;

    // Limit messages
    while (conversation.children.length > 50) {
      conversation.removeChild(conversation.firstChild);
    }
  }

  // The last line said, shown under the core while the panels are hidden.
  function setCaption(speaker, text, append) {
    if (!replyCaption) return;
    if (append) {
      const current = replyCaption.querySelector(".caption-text");
      if (current) {
        current.textContent += " " + text;
        return;
      }
    }
    const who = speaker === "user" ? "YOU" : "JARVIS";
    replyCaption.innerHTML =
      `<span class="caption-who${speaker === "user" ? " user" : ""}">${who}</span>` +
      `<span class="caption-text"></span>`;
    replyCaption.querySelector(".caption-text").textContent = text;
  }

  // Tasks
  function updateTasks(tasks) {
    if (!tasks || tasks.length === 0) {
      taskList.innerHTML = '<div class="empty-text">No pending missions</div>';
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

  // Self-evolution. The panel lists what the assistant has proposed for
  // itself and what it has already applied; nothing here is applied without
  // the sandbox saying the tests are still green.
  function updateEvolution(state) {
    if (!evolutionList) return;
    const pending = (state && state.proposals) || [];
    const open = pending.filter((p) => p.status === "pending");
    const applied = pending.filter((p) => p.status === "approved");
    if (open.length === 0 && applied.length === 0) {
      evolutionList.innerHTML = '<div class="empty-text">No changes pending</div>';
      return;
    }
    evolutionList.innerHTML = "";
    open.forEach((p) => {
      const div = document.createElement("div");
      div.className = "evolution-item";
      const files = (p.files || []).join(", ");
      div.innerHTML =
        `<div class="evolution-summary">${escapeHtml(p.summary || p.id)}</div>` +
        `<div class="evolution-files">${escapeHtml(files)}</div>` +
        `<div class="evolution-actions">` +
        `<button class="evo-btn approve" data-id="${escapeHtml(p.id)}">APPLY</button>` +
        `<button class="evo-btn reject" data-id="${escapeHtml(p.id)}">DISCARD</button>` +
        `</div>`;
      evolutionList.appendChild(div);
    });
    applied.forEach((p) => {
      const div = document.createElement("div");
      div.className = "evolution-item applied";
      const files = (p.files || []).join(", ");
      div.innerHTML =
        `<div class="evolution-summary">${escapeHtml(p.summary || p.id)}</div>` +
        `<div class="evolution-files">applied to ${escapeHtml(files)}</div>` +
        `<div class="evolution-actions">` +
        `<button class="evo-btn revert" data-id="${escapeHtml(p.id)}">UNDO</button>` +
        `</div>`;
      evolutionList.appendChild(div);
    });
    evolutionList.querySelectorAll(".evo-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.id;
        let action = "reject";
        if (btn.classList.contains("approve")) action = "approve";
        else if (btn.classList.contains("revert")) action = "revert";
        btn.disabled = true;
        fetch(`/api/evolution/${encodeURIComponent(id)}?action=${action}`, {
          method: "POST",
        })
          .then((r) => r.json())
          .then((data) => {
            addMessage(
              "jarvis",
              data.ok ? data.message : `I could not do that, sir: ${data.message}`
            );
            loadEvolution();
          })
          .catch(() => {
            btn.disabled = false;
            addMessage("jarvis", "I could not reach the core, sir.");
          });
      });
    });
  }

  function loadEvolution() {
    fetch("/api/evolution")
      .then((r) => r.json())
      .then((data) => {
        updateEvolution({
          proposals: data.proposals,
          ...(data.status || {}),
        });
      })
      .catch(() => {
        /* The panel simply keeps what it has. */
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
      loadCosmos();
      loadEvolution();
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
        setCaption(
          msg.speaker === "user" ? "user" : "jarvis",
          msg.text,
          msg.append || false
        );
        // A reply may have added or removed a memory; refresh the galaxy.
        if (msg.speaker !== "user" && !msg.append) scheduleCosmosRefresh();
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
      // The panels are shown or hidden only when they are asked for.
      case "panels":
        setPanels(msg.visible);
        break;
      // The evolution module pushes its state when a cycle ends or a proposal
      // is queued, applied or turned down.
      case "evolution":
        loadEvolution();
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
      if (btn.id === "panels-btn") {
        // The server holds the state, so ask it to flip the panels.
        sendCommand(panelsVisible ? "hide panels" : "show panels");
        return;
      }
      if (btn.id === "evolve-btn") {
        // A cycle runs the test suite, so it is slow and it happens on the
        // server. The panel fills in when there is something to show.
        sendCommand("evolve");
        setPanels(true);
        return;
      }
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

  function setText(selector, value) {
    const el = $(selector);
    if (el) el.textContent = value;
  }

  // ===========================================================================
  // NEURAL GALAXY
  // Every particle is a memory, skill, ability, or mission orbiting the core.
  // ===========================================================================

  const canvas = $("#galaxy-canvas");
  const ctx = canvas ? canvas.getContext("2d") : null;
  const tooltip = $("#particle-tooltip");
  const TAU = Math.PI * 2;

  const PALETTE = {
    memory: { r: 255, g: 198, b: 92 },
    skill: { r: 77, g: 226, b: 255 },
    ability: { r: 176, g: 123, b: 255 },
    mission: { r: 92, g: 255, b: 176 },
  };

  const KIND_LABEL = {
    memory: "MEMORY",
    skill: "SKILL",
    ability: "ABILITY",
    mission: "MISSION",
  };

  // Orbit radius as a fraction of the measured galaxy radius.
  const RING = { memory: 0.38, skill: 0.60, ability: 0.80, mission: 1.0 };
  const CAP = { memory: 36, skill: 40, ability: 18, mission: 12 };

  const galaxy = {
    w: 0,
    h: 0,
    dpr: 1,
    cx: 0,
    cy: 0,
    vx: 0,
    vy: 0,
    radius: 300,
    stars: [],
    particles: [],
    time: 0,
    lastTs: 0,
    energy: 0.22,
    energyTarget: 0.22,
    parX: 0,
    parY: 0,
    parTargetX: 0,
    parTargetY: 0,
    mouse: { x: 0, y: 0, inside: false },
    hovered: null,
    reducedMotion: false,
  };

  let cosmosTimer = null;

  function initGalaxy() {
    if (!canvas || !ctx) return;
    galaxy.reducedMotion = !!(window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    resizeGalaxy();
    window.addEventListener("resize", () => {
      // Wait a frame so the grid has settled before measuring the core.
      requestAnimationFrame(resizeGalaxy);
    });
    canvas.addEventListener("pointermove", onGalaxyPointerMove);
    canvas.addEventListener("pointerleave", onGalaxyPointerLeave);
    requestAnimationFrame(renderGalaxy);
  }

  function resizeGalaxy() {
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    galaxy.dpr = dpr;
    galaxy.w = window.innerWidth;
    galaxy.h = window.innerHeight;
    canvas.width = Math.round(galaxy.w * dpr);
    canvas.height = Math.round(galaxy.h * dpr);
    canvas.style.width = galaxy.w + "px";
    canvas.style.height = galaxy.h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    measureGalaxyCenter();
    buildStars();
    loadCosmos();
  }

  function measureGalaxyCenter() {
    const el = $("#brain-core") || $("#center");
    if (!el) {
      galaxy.cx = galaxy.w / 2;
      galaxy.cy = galaxy.h / 2;
      galaxy.radius = Math.min(galaxy.w, galaxy.h) * 0.32;
      return;
    }
    const rect = el.getBoundingClientRect();
    galaxy.cx = rect.left + rect.width / 2;
    galaxy.cy = rect.top + rect.height / 2;
    const availW = Math.min(galaxy.cx, galaxy.w - galaxy.cx) - 24;
    const availH = Math.min(galaxy.cy, galaxy.h - galaxy.cy) - 24;
    galaxy.radius = Math.max(150, Math.min(380, Math.min(availW, availH)));
  }

  function buildStars() {
    const count = Math.min(280, Math.round((galaxy.w * galaxy.h) / 8200));
    const stars = [];
    for (let i = 0; i < count; i++) {
      const tint = Math.random();
      stars.push({
        x: Math.random() * galaxy.w,
        y: Math.random() * galaxy.h,
        r: 0.25 + Math.random() * 1.15,
        a: 0.15 + Math.random() * 0.55,
        tw: Math.random() * TAU,
        twSpeed: 0.3 + Math.random() * 1.4,
        depth: 0.2 + Math.random() * 0.8,
        hue: tint < 0.82 ? "207, 230, 255" : tint < 0.92 ? "180, 160, 255" : "255, 214, 170",
      });
    }
    galaxy.stars = stars;

    // Stars are drawn in three tint groups so the fill colour is set three
    // times a frame instead of once per star. Per-star brightness is carried
    // by globalAlpha, which is far cheaper than building a colour string.
    const groups = new Map();
    for (const s of stars) {
      let group = groups.get(s.hue);
      if (!group) {
        group = { color: "rgb(" + s.hue + ")", stars: [] };
        groups.set(s.hue, group);
      }
      group.stars.push(s);
    }
    galaxy.starGroups = Array.from(groups.values());
  }

  // --- Cosmos data ---------------------------------------------------------

  async function loadCosmos() {
    try {
      const res = await fetch("/api/cosmos");
      if (!res.ok) return;
      applyCosmos(await res.json());
    } catch (e) {
      /* offline: stars and core still render, particles return on reconnect */
    }
  }

  function scheduleCosmosRefresh() {
    if (cosmosTimer) return;
    cosmosTimer = setTimeout(() => {
      cosmosTimer = null;
      loadCosmos();
    }, 1500);
  }

  function sampleEvenly(list, cap) {
    if (list.length <= cap) return list;
    const out = [];
    const step = list.length / cap;
    for (let i = 0; i < cap; i++) out.push(list[Math.floor(i * step)]);
    return out;
  }

  function prettyName(name) {
    return String(name || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function formatStored(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d.getTime())) return "";
    return "stored " +
      d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  }

  function applyCosmos(data) {
    data = data || {};
    const memories = Array.isArray(data.memories) ? data.memories : [];
    const skills = Array.isArray(data.skills) ? data.skills : [];
    const abilities = Array.isArray(data.abilities) ? data.abilities : [];
    const tasks = (Array.isArray(data.tasks) ? data.tasks : []).filter((t) => !t.done);
    const stats = data.stats || {};

    const particles = [];

    function add(options) {
      const kind = options.kind;
      const total = Math.max(1, options.total);
      const angle =
        (options.index / total) * TAU +
        (kind.length * 0.63) +
        Math.random() * 0.4;
      particles.push({
        kind: kind,
        label: options.label,
        detail: options.detail || "",
        meta: options.meta || "",
        angle: angle,
        radius: RING[kind] + (Math.random() - 0.5) * 0.05,
        speed: (0.000045 + Math.random() * 0.00006) *
          (kind === "mission" ? 1.5 : kind === "ability" ? 0.75 : 1),
        size: (kind === "memory" ? 2.0 : 1.5) +
          Math.random() * (kind === "memory" ? 1.4 : 1.9),
        phase: Math.random() * TAU,
        depth: 0.55 + Math.random() * 0.45,
        pulseOffset: Math.random(),
      });
    }

    const shownMemories = sampleEvenly(memories, CAP.memory);
    shownMemories.forEach((m, i) => {
      add({
        kind: "memory",
        label: m.fact,
        detail: "Committed to long-term memory.",
        meta: formatStored(m.timestamp),
        index: i,
        total: shownMemories.length,
      });
    });

    const shownSkills = sampleEvenly(skills, CAP.skill);
    shownSkills.forEach((s, i) => {
      add({
        kind: "skill",
        label: prettyName(s.name),
        detail: s.description || "",
        meta: "routable skill",
        index: i,
        total: shownSkills.length,
      });
    });

    const shownAbilities = sampleEvenly(abilities, CAP.ability);
    shownAbilities.forEach((a, i) => {
      add({
        kind: "ability",
        label: prettyName(a.name),
        detail: a.description || "",
        meta: "privileged ability",
        index: i,
        total: shownAbilities.length,
      });
    });

    const shownTasks = sampleEvenly(tasks, CAP.mission);
    shownTasks.forEach((t, i) => {
      add({
        kind: "mission",
        label: t.description || t.text || "Untitled mission",
        detail: "Mission #" + (t.id ?? "?"),
        meta: t.due ? "due " + t.due : "pending",
        index: i,
        total: shownTasks.length,
      });
    });

    // Back-to-front so nearer particles glow on top.
    particles.sort((a, b) => a.depth - b.depth);
    galaxy.particles = particles;
    galaxy.hovered = null;
    if (tooltip) tooltip.classList.remove("visible");

    setText("#cosmos-operator", (data.identity && data.identity.name) || "UNKNOWN");
    setText("#cosmos-memories", stats.memories ?? memories.length);
    setText("#cosmos-skills", stats.skills ?? skills.length);
    setText("#cosmos-abilities", stats.abilities ?? abilities.length);
    setText("#cosmos-missions", stats.missions ?? tasks.length);
  }

  // --- Pointer / tooltip ---------------------------------------------------

  function onGalaxyPointerMove(e) {
    galaxy.mouse.x = e.clientX;
    galaxy.mouse.y = e.clientY;
    galaxy.mouse.inside = true;
    galaxy.parTargetX = (e.clientX / galaxy.w - 0.5) * -26;
    galaxy.parTargetY = (e.clientY / galaxy.h - 0.5) * -18;
    if (galaxy.hovered) positionTooltip();
  }

  function onGalaxyPointerLeave() {
    galaxy.mouse.inside = false;
    galaxy.parTargetX = 0;
    galaxy.parTargetY = 0;
    galaxy.hovered = null;
    if (tooltip) tooltip.classList.remove("visible");
    if (canvas) canvas.style.cursor = "default";
  }

  function updateHover() {
    if (!tooltip) return;
    if (!galaxy.mouse.inside) {
      if (galaxy.hovered) {
        galaxy.hovered = null;
        tooltip.classList.remove("visible");
      }
      return;
    }

    let best = null;
    let bestD = Infinity;
    for (let i = 0; i < galaxy.particles.length; i++) {
      const p = galaxy.particles[i];
      const dx = p.x - galaxy.mouse.x;
      const dy = p.y - galaxy.mouse.y;
      const d = dx * dx + dy * dy;
      const hit = Math.max(13, p.size * 4);
      if (d <= hit * hit && d < bestD) {
        bestD = d;
        best = p;
      }
    }

    if (best !== galaxy.hovered) {
      galaxy.hovered = best;
      if (best) showTooltip(best);
      else tooltip.classList.remove("visible");
      if (canvas) canvas.style.cursor = best ? "pointer" : "default";
    }
    if (best) positionTooltip();
  }

  function showTooltip(p) {
    tooltip.className = "particle-tooltip visible " + p.kind;
    const kindEl = tooltip.querySelector(".tt-kind");
    const labelEl = tooltip.querySelector(".tt-label");
    const detailEl = tooltip.querySelector(".tt-detail");
    const metaEl = tooltip.querySelector(".tt-meta");
    if (kindEl) kindEl.textContent = KIND_LABEL[p.kind] || "PARTICLE";
    if (labelEl) labelEl.textContent = p.label || "";
    if (detailEl) {
      detailEl.textContent = p.detail || "";
      detailEl.style.display = p.detail ? "" : "none";
    }
    if (metaEl) metaEl.textContent = p.meta || "";
    positionTooltip();
  }

  function positionTooltip() {
    if (!tooltip) return;
    const pad = 12;
    const w = tooltip.offsetWidth;
    const h = tooltip.offsetHeight;
    let x = galaxy.mouse.x + 18;
    let y = galaxy.mouse.y + 18;
    if (x + w > galaxy.w - pad) x = galaxy.mouse.x - w - 18;
    if (y + h > galaxy.h - pad) y = galaxy.mouse.y - h - 18;
    if (x < pad) x = pad;
    if (y < pad) y = pad;
    tooltip.style.transform = `translate(${x}px, ${y}px)`;
  }

  // --- Render loop ---------------------------------------------------------

  // Sixty frames a second of full-screen canvas work on top of a dozen
  // blurred layers is what makes the browser tab feel heavy, and the eye
  // cannot tell thirty frames from sixty on a slow drift. The clock keeps
  // running at full speed; only the drawing is throttled.
  const FRAME_INTERVAL_MS = 1000 / 30;

  function renderGalaxy(ts) {
    if (!ctx) return;
    if (!galaxy.lastTs) galaxy.lastTs = ts;
    const dt = Math.min(64, Math.max(0, ts - galaxy.lastTs));
    galaxy.lastTs = ts;
    galaxy.time += dt;

    requestAnimationFrame(renderGalaxy);
    if (dt < FRAME_INTERVAL_MS) return;

    const t = galaxy.time;

    let target = galaxy.energyTarget;
    if (document.body.classList.contains("ptt")) target = Math.max(target, 0.95);
    galaxy.energy += (target - galaxy.energy) * Math.min(1, dt / 400);

    galaxy.parX += (galaxy.parTargetX - galaxy.parX) * Math.min(1, dt / 300);
    galaxy.parY += (galaxy.parTargetY - galaxy.parY) * Math.min(1, dt / 300);
    galaxy.vx = galaxy.cx + galaxy.parX;
    galaxy.vy = galaxy.cy + galaxy.parY;

    ctx.clearRect(0, 0, galaxy.w, galaxy.h);

    drawStars(t);
    drawOrbits(t);
    updateParticles(t, dt);
    drawNebulaLinks(t);
    drawParticles(t);
    drawCoreAtmosphere(t);

    updateHover();
  }

  function drawStars(t) {
    const groups = galaxy.starGroups;
    if (!groups) return;
    for (let g = 0; g < groups.length; g++) {
      const group = groups[g];
      ctx.fillStyle = group.color;
      const list = group.stars;
      for (let i = 0; i < list.length; i++) {
        const s = list[i];
        const twinkle = 0.55 + 0.45 * Math.sin(t * 0.001 * s.twSpeed + s.tw);
        ctx.globalAlpha = s.a * twinkle;
        ctx.beginPath();
        ctx.arc(
          s.x + galaxy.parX * s.depth * 0.4,
          s.y + galaxy.parY * s.depth * 0.4,
          s.r,
          0,
          TAU
        );
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  // Built once: this used to be rebuilt on every frame.
  const ORBIT_RINGS = [
    ["memory", RING.memory, [2, 9]],
    ["skill", RING.skill, [3, 8]],
    ["ability", RING.ability, [1, 7]],
    ["mission", RING.mission, [2, 12]],
  ];

  function drawOrbits(t) {
    const rings = ORBIT_RINGS;
    ctx.save();
    ctx.lineWidth = 1;
    for (let i = 0; i < rings.length; i++) {
      const kind = rings[i][0];
      const factor = rings[i][1];
      const col = PALETTE[kind];
      ctx.strokeStyle = `rgba(${col.r},${col.g},${col.b},${0.07 + galaxy.energy * 0.09})`;
      ctx.setLineDash(rings[i][2]);
      ctx.lineDashOffset = -t * 0.012;
      ctx.beginPath();
      ctx.ellipse(
        galaxy.vx,
        galaxy.vy,
        galaxy.radius * factor,
        galaxy.radius * factor * 0.72,
        0,
        0,
        TAU
      );
      ctx.stroke();
    }
    ctx.restore();
  }

  function updateParticles(t, dt) {
    const speedScale = galaxy.reducedMotion ? 0.15 : 1;
    for (let i = 0; i < galaxy.particles.length; i++) {
      const p = galaxy.particles[i];
      p.angle += p.speed * dt * (0.35 + galaxy.energy * 1.6) * speedScale;

      const base = galaxy.radius * p.radius;
      const wobble = Math.sin(t * 0.00035 + p.phase) *
        (p.kind === "memory" ? 3 : 6);
      const rx = base + wobble;
      const ry = base * 0.72 + wobble * 0.5;

      p.x = galaxy.vx + Math.cos(p.angle) * rx;
      p.y = galaxy.vy + Math.sin(p.angle) * ry;
      p.alpha = (0.45 + 0.55 * (0.5 + 0.5 * Math.sin(t * 0.0016 + p.phase))) * p.depth;
    }
  }

  // The tether colours only change with the assistant's energy, so they are
  // cached instead of rebuilt as a colour string for every particle, frame
  // after frame.
  const linkStyles = { key: null, byKind: {} };

  function tetherStyles(lineAlpha) {
    const key = Math.round(lineAlpha * 500);
    if (linkStyles.key !== key) {
      linkStyles.key = key;
      for (const kind in PALETTE) {
        const col = PALETTE[kind];
        linkStyles.byKind[kind] = `rgba(${col.r},${col.g},${col.b},${lineAlpha})`;
      }
    }
    return linkStyles.byKind;
  }

  function drawNebulaLinks(t) {
    const lineAlpha = 0.03 + galaxy.energy * 0.055;
    const styles = tetherStyles(lineAlpha);
    for (let i = 0; i < galaxy.particles.length; i++) {
      const p = galaxy.particles[i];

      // Every particle is tethered to the brain.
      ctx.strokeStyle = styles[p.kind];
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(galaxy.vx, galaxy.vy);
      ctx.stroke();

      // Signals travel along the tether: memories and skills into the core,
      // abilities and missions dispatched outward.
      if (i % 2 === 0) {
        const inward = p.kind === "memory" || p.kind === "skill";
        const prog = (t * 0.00009 + p.pulseOffset) % 1;
        const fromX = inward ? p.x : galaxy.vx;
        const fromY = inward ? p.y : galaxy.vy;
        const toX = inward ? galaxy.vx : p.x;
        const toY = inward ? galaxy.vy : p.y;
        const px = fromX + (toX - fromX) * prog;
        const py = fromY + (toY - fromY) * prog;
        const fade = 1 - Math.abs(prog - 0.5) * 2;
        ctx.globalAlpha = 0.20 + fade * 0.55;
        ctx.fillStyle = "rgb(255,255,255)";
        ctx.beginPath();
        ctx.arc(px, py, 1.3, 0, TAU);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  // A particle glow used to be a fresh radial gradient for every particle on
  // every frame: over a hundred gradients a frame, sixty times a second, all
  // of them re-rasterised by the compositor. The glow is baked once per kind
  // into a small sprite and stamped from then on.
  const glowSprites = {};
  const GLOW_SPRITE_PX = 64;

  function glowSprite(kind) {
    if (glowSprites[kind]) return glowSprites[kind];
    const col = PALETTE[kind];
    const c = document.createElement("canvas");
    c.width = GLOW_SPRITE_PX;
    c.height = GLOW_SPRITE_PX;
    const g = c.getContext("2d");
    const half = GLOW_SPRITE_PX / 2;
    const grad = g.createRadialGradient(half, half, 0, half, half, half);
    grad.addColorStop(0, `rgba(${col.r},${col.g},${col.b},0.92)`);
    grad.addColorStop(0.45, `rgba(${col.r},${col.g},${col.b},0.28)`);
    grad.addColorStop(1, `rgba(${col.r},${col.g},${col.b},0)`);
    g.fillStyle = grad;
    g.fillRect(0, 0, GLOW_SPRITE_PX, GLOW_SPRITE_PX);
    glowSprites[kind] = c;
    return c;
  }

  function drawParticles(t) {
    for (let i = 0; i < galaxy.particles.length; i++) {
      const p = galaxy.particles[i];
      const col = PALETTE[p.kind];
      const pulse = 0.7 + 0.3 * Math.sin(t * 0.002 + p.phase);
      const glowR = p.size * (3.6 + galaxy.energy * 3.2) * p.depth;

      const sprite = glowSprite(p.kind);
      ctx.globalAlpha = pulse;
      ctx.drawImage(sprite, p.x - glowR, p.y - glowR, glowR * 2, glowR * 2);

      // Hot centre, tinted by category.
      ctx.globalAlpha = 0.55 + 0.35 * pulse;
      ctx.fillStyle = "rgb(255,255,255)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(0.8, p.size * 0.55), 0, TAU);
      ctx.fill();

      if (p === galaxy.hovered) {
        ctx.globalAlpha = 0.85;
        ctx.strokeStyle = `rgb(${col.r},${col.g},${col.b})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size + 7, 0, TAU);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  function drawCoreAtmosphere(t) {
    const energy = galaxy.energy;

    // Bloom behind the DOM core.
    const bloomR = 130 + energy * 80;
    const bloom = ctx.createRadialGradient(
      galaxy.vx, galaxy.vy, 0,
      galaxy.vx, galaxy.vy, bloomR
    );
    bloom.addColorStop(0, `rgba(150,205,255,${0.10 + energy * 0.14})`);
    bloom.addColorStop(0.5, `rgba(90,150,255,${0.04 + energy * 0.06})`);
    bloom.addColorStop(1, "rgba(70,110,255,0)");
    ctx.fillStyle = bloom;
    ctx.beginPath();
    ctx.arc(galaxy.vx, galaxy.vy, bloomR, 0, TAU);
    ctx.fill();

    // Ripples shed by the brain, faster when JARVIS is active.
    const maxR = galaxy.radius * 0.30;
    const speed = 0.00010 + energy * 0.00022;
    for (let i = 0; i < 3; i++) {
      const prog = (t * speed + i / 3) % 1;
      const r = 36 + prog * maxR;
      ctx.strokeStyle = `rgba(140,200,255,${(1 - prog) * (0.10 + energy * 0.22)})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(
        galaxy.vx,
        galaxy.vy,
        r,
        r * 0.72,
        0,
        0,
        TAU
      );
      ctx.stroke();
    }
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
        setMicLink("ERROR");
      } else {
        setMicLink(info.listening ? "LISTENING" : "OFF");
      }
    } catch (e) {
      /* offline: the WebSocket path already reports the problem */
    }
  }

  // Track speaking status
  const ENERGY_BY_STATE = {
    standby: 0.22,
    online: 0.30,
    listening: 0.80,
    processing: 0.62,
    speaking: 1.0,
  };

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

    // The galaxy breathes harder while JARVIS is working.
    galaxy.energyTarget = ENERGY_BY_STATE[status] ?? 0.3;

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
  setPanels(false);
  initSpeechRecognition();
  probeListening();
  initGalaxy();
  loadCosmos();
  setInterval(loadCosmos, 30000);
})();
