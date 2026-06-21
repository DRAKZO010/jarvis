/* J.A.R.V.I.S. HUD — Frontend Logic */

(() => {
    "use strict";

    const WS_URL = `ws://${location.hostname || "localhost"}:8765`;
    const RECONNECT_MS = 2000;
    const PARTICLE_COUNT = 50;
    const WAVE_BARS = 48;

    const $time = document.getElementById("time");
    const $date = document.getElementById("date");
    const $connDot = document.getElementById("connDot");
    const $sysLabel = document.getElementById("sysLabel");
    const $cpuBar = document.getElementById("cpuBar");
    const $ramBar = document.getElementById("ramBar");
    const $cpuVal = document.getElementById("cpuVal");
    const $ramVal = document.getElementById("ramVal");
    const $ramDetail = document.getElementById("ramDetail");
    const $batteryRow = document.getElementById("batteryRow");
    const $battBar = document.getElementById("battBar");
    const $battVal = document.getElementById("battVal");
    const $status = document.getElementById("statusText");
    const $core = document.getElementById("coreOrb");
    const $log = document.getElementById("conversationLog");
    const $input = document.getElementById("cmdInput");
    const $send = document.getElementById("sendBtn");
    const $waveCanvas = document.getElementById("waveCanvas");
    const $partCanvas = document.getElementById("particleCanvas");

    let ws = null;
    let voiceLevel = 0;
    let targetLevel = 0;
    let currentState = "standby";
    const waveBars = new Float32Array(WAVE_BARS);

    // Clock
    function updateClock() {
        const now = new Date();
        $time.textContent = now.toLocaleTimeString("en-US", {
            hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false
        });
        $date.textContent = now.toLocaleDateString("en-US", {
            weekday: "short", year: "numeric", month: "short", day: "numeric"
        }).toUpperCase();
    }
    updateClock();
    setInterval(updateClock, 1000);

    // WebSocket
    function connect() {
        ws = new WebSocket(WS_URL);
        ws.onopen = () => {
            $connDot.classList.add("online");
            $sysLabel.textContent = "ALL SYSTEMS NOMINAL";
            addSystemMsg("Connection established.");
        };
        ws.onclose = () => {
            $connDot.classList.remove("online");
            $sysLabel.textContent = "CONNECTION LOST";
            setTimeout(connect, RECONNECT_MS);
        };
        ws.onerror = () => ws.close();
        ws.onmessage = (e) => {
            try { handleEvent(JSON.parse(e.data)); } catch (_) {}
        };
    }

    function handleEvent(msg) {
        switch (msg.type) {
            case "status": setUIState(msg.state); break;
            case "transcript": addMessage(msg.text, msg.speaker); break;
            case "metrics": updateMetrics(msg); break;
            case "voice_level": targetLevel = msg.level; break;
        }
    }

    function setUIState(state) {
        currentState = state;
        const labels = { standby: "STANDBY", listening: "LISTENING", processing: "PROCESSING", speaking: "SPEAKING" };
        $status.textContent = labels[state] || state.toUpperCase();
        $status.className = "reactor-status " + state;
        $core.className = "core-orb " + state;
    }

    function updateMetrics(m) {
        $cpuBar.style.width = m.cpu + "%";
        $cpuVal.textContent = Math.round(m.cpu) + "%";

        $ramBar.style.width = m.ram_pct + "%";
        $ramVal.textContent = Math.round(m.ram_pct) + "%";
        $ramDetail.textContent = m.ram_used + " / " + m.ram_total + " GB";

        if (m.battery !== undefined) {
            $batteryRow.style.display = "grid";
            $battBar.style.width = m.battery + "%";
            $battVal.textContent = Math.round(m.battery) + "%";
            if (m.charging) {
                $battBar.style.background = "var(--green)";
            } else if (m.battery < 20) {
                $battBar.style.background = "var(--danger)";
            } else if (m.battery < 50) {
                $battBar.style.background = "var(--orange)";
            } else {
                $battBar.style.background = "var(--green)";
            }
        }
    }

    function addMessage(text, speaker) {
        const div = document.createElement("div");
        div.className = "log-msg log-" + speaker;
        div.textContent = text;
        $log.appendChild(div);
        $log.scrollTop = $log.scrollHeight;
        while ($log.children.length > 50) $log.removeChild($log.firstChild);
    }

    function addSystemMsg(text) {
        const div = document.createElement("div");
        div.className = "log-msg log-system";
        div.textContent = text;
        $log.appendChild(div);
        $log.scrollTop = $log.scrollHeight;
    }

    function sendCommand() {
        const text = $input.value.trim();
        if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({ type: "command", text }));
        $input.value = "";
    }

    $send.addEventListener("click", sendCommand);
    $input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendCommand(); });

    // Waveform
    const wCtx = $waveCanvas.getContext("2d");

    function resizeWaveCanvas() {
        const size = Math.min($waveCanvas.parentElement.clientWidth, $waveCanvas.parentElement.clientHeight) || 400;
        $waveCanvas.width = size;
        $waveCanvas.height = size;
    }
    resizeWaveCanvas();
    window.addEventListener("resize", resizeWaveCanvas);

    function renderWaveform() {
        const W = $waveCanvas.width;
        const H = $waveCanvas.height;
        const cx = W / 2;
        const cy = H / 2;
        const baseR = W * 0.12;
        const maxBarH = W * 0.14;

        wCtx.clearRect(0, 0, W, H);
        voiceLevel += (targetLevel - voiceLevel) * 0.15;

        for (let i = WAVE_BARS - 1; i > 0; i--) {
            waveBars[i] = waveBars[i - 1] * 0.92;
        }
        waveBars[0] = Math.min(voiceLevel * 40, 1.0);

        for (let i = 0; i < WAVE_BARS; i++) {
            const angle = (i / WAVE_BARS) * Math.PI * 2 - Math.PI / 2;
            const barH = waveBars[i] * maxBarH + 1;
            const x1 = cx + Math.cos(angle) * baseR;
            const y1 = cy + Math.sin(angle) * baseR;
            const x2 = cx + Math.cos(angle) * (baseR + barH);
            const y2 = cy + Math.sin(angle) * (baseR + barH);

            let color;
            const alpha = 0.2 + waveBars[i] * 0.6;
            switch (currentState) {
                case "listening": color = `rgba(0, 255, 136, ${alpha})`; break;
                case "processing": color = `rgba(255, 140, 0, ${alpha})`; break;
                case "speaking": color = `rgba(0, 212, 255, ${alpha * 1.2})`; break;
                default: color = `rgba(0, 180, 255, ${alpha * 0.3})`;
            }

            wCtx.beginPath();
            wCtx.moveTo(x1, y1);
            wCtx.lineTo(x2, y2);
            wCtx.strokeStyle = color;
            wCtx.lineWidth = 2;
            wCtx.lineCap = "round";
            wCtx.stroke();
        }

        requestAnimationFrame(renderWaveform);
    }
    renderWaveform();

    // Particles
    const pCtx = $partCanvas.getContext("2d");
    let particles = [];

    function resizeParticleCanvas() {
        $partCanvas.width = window.innerWidth;
        $partCanvas.height = window.innerHeight;
    }
    resizeParticleCanvas();
    window.addEventListener("resize", resizeParticleCanvas);

    function initParticles() {
        particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            particles.push({
                x: Math.random() * $partCanvas.width,
                y: Math.random() * $partCanvas.height,
                vx: (Math.random() - 0.5) * 0.1,
                vy: -Math.random() * 0.2 - 0.03,
                r: Math.random() * 1.2 + 0.2,
                a: Math.random() * 0.15 + 0.02,
            });
        }
    }
    initParticles();

    function renderParticles() {
        pCtx.clearRect(0, 0, $partCanvas.width, $partCanvas.height);
        for (const p of particles) {
            p.x += p.vx;
            p.y += p.vy;
            if (p.y < -10) p.y = $partCanvas.height + 10;
            if (p.x < -10) p.x = $partCanvas.width + 10;
            if (p.x > $partCanvas.width + 10) p.x = -10;

            pCtx.beginPath();
            pCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            pCtx.fillStyle = `rgba(0, 180, 255, ${p.a})`;
            pCtx.fill();
        }
        requestAnimationFrame(renderParticles);
    }
    renderParticles();

    // Tick marks for arc reactor
    (function generateTicks() {
        const g = document.getElementById("tickMarks");
        if (!g) return;
        for (let i = 0; i < 60; i++) {
            const angle = (i / 60) * 360;
            const len = i % 5 === 0 ? 8 : 3;
            const r1 = 235;
            const r2 = r1 + len;
            const rad = (angle * Math.PI) / 180;
            const x1 = 250 + r1 * Math.cos(rad);
            const y1 = 250 + r1 * Math.sin(rad);
            const x2 = 250 + r2 * Math.cos(rad);
            const y2 = 250 + r2 * Math.sin(rad);
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", x1);
            line.setAttribute("y1", y1);
            line.setAttribute("x2", x2);
            line.setAttribute("y2", y2);
            line.setAttribute("stroke", "rgba(0,180,255,0.1)");
            line.setAttribute("stroke-width", i % 5 === 0 ? "1" : "0.4");
            g.appendChild(line);
        }
    })();

    // Window controls (frameless mode)
    const $btnMinimize = document.getElementById("btnMinimize");
    const $btnClose = document.getElementById("btnClose");
    if ($btnMinimize) {
        $btnMinimize.addEventListener("click", () => {
            if (window.pywebview) window.pywebview.api.minimize();
        });
    }
    if ($btnClose) {
        $btnClose.addEventListener("click", () => {
            if (window.pywebview) window.pywebview.api.close();
            else window.close();
        });
    }

    connect();
    setUIState("standby");
})();
