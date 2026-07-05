const fmtPct = (value) => (Number(value) * 100).toFixed(2) + "%";
const fmtNum = (value, digits = 2) => Number(value).toFixed(digits);
let replayTimer = null;
let replayWindow = 1;

let liveSocket = null;
let liveReconnectDelayMs = 1000;
const LIVE_RECONNECT_MAX_MS = 15000;
let liveFeedMessageCount = 0;
let liveFeedAlertCount = 0;
const liveFeedEvents = [];
const LIVE_FEED_MAX_EVENTS = 15;

let unacknowledgedAlerts = 0;
let audioContext = null;
let liveWaveformFetchInFlight = false;

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function shortName(path) {
  return String(path || "").split("/").pop().replace("_report.json", "");
}

function renderSummary(summary) {
  const best = summary.best_model;
  const kafka = summary.kafka;
  setText("modelName", best.name);
  setText("testRecall", fmtPct(best.test_recall));
  setText("testPrecision", fmtPct(best.test_precision));
  setText("testFpr", fmtPct(best.test_fpr));
  setText("testFn", best.test_fn);
  setText("kafkaMessages", kafka.messages_processed);
  setText("kafkaAlerts", kafka.alerts);
  setText("kafkaLatency", `${fmtNum(kafka.avg_latency_ms)} ms`);
  setText("firstAlert", kafka.first_alert ? `#${kafka.first_alert.window_index}` : "None");

  const health = document.getElementById("healthBadge");
  health.textContent = kafka.alerts > 0 ? "Alert detected" : "Normal";
  health.className = kafka.alerts > 0 ? "badge danger" : "badge ok";
}

function renderModelRows(rows) {
  const body = document.getElementById("modelRows");
  body.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.model}<br><span class="alert-meta">${shortName(row.report)}</span></td>
          <td>${fmtPct(row.test_recall)}</td>
          <td>${fmtPct(row.test_fpr)}</td>
          <td>${fmtPct(row.test_precision)}</td>
          <td>${row.test_fn}</td>
          <td>${row.test_fp}</td>
        </tr>
      `
    )
    .join("");
}

function renderRareRows(rows) {
  const body = document.getElementById("rareRows");
  body.innerHTML = rows
    .slice(0, 8)
    .map(
      (row) => `
        <tr>
          <td>${row.data}</td>
          <td>${row.model}</td>
          <td>${fmtPct(row.fault_ratio)}</td>
          <td>${fmtPct(row.recall)}</td>
          <td>${fmtPct(row.precision)}</td>
          <td>${fmtNum(row.alerts_per_10000_windows)}</td>
        </tr>
      `
    )
    .join("");
}

function renderAlerts(alerts) {
  const list = document.getElementById("alertList");
  if (!alerts.length) {
    list.innerHTML = '<p class="alert-meta">No alert incidents found.</p>';
    return;
  }
  list.innerHTML = alerts
    .slice()
    .reverse()
    .map((incident) => {
      return `
        <article class="alert-item">
          <strong>Windows #${incident.start_window} - #${incident.end_window} | ${fmtPct(incident.max_probability)} max probability</strong>
          <div class="alert-meta">
            Sample ${incident.sample_id}, ${incident.start_time}s to ${incident.end_time}s,
            grouped windows: ${incident.count}, true label: ${incident.true_window_label || "unknown"}
          </div>
          <div class="alert-meta">Top signal: ${incident.top_feature || "No feature detail"}</div>
        </article>
      `;
    })
    .join("");
}

function normalize(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return values.map((value) => (value - min) / span);
}

function drawWaveform(waveform, canvasId = "waveformCanvas") {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, width, height);

  const channels = Object.entries(waveform.channels).slice(0, 8);
  const bandHeight = height / channels.length;
  const colors = ["#5eead4", "#93c5fd", "#fca5a5", "#fcd34d", "#c4b5fd", "#86efac", "#fdba74", "#67e8f9"];

  ctx.font = "14px sans-serif";
  channels.forEach(([name, values], channelIndex) => {
    const yOffset = channelIndex * bandHeight;
    const scaled = normalize(values);
    ctx.strokeStyle = colors[channelIndex % colors.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    scaled.forEach((value, index) => {
      const x = (index / (scaled.length - 1)) * width;
      const y = yOffset + bandHeight - value * (bandHeight - 18) - 8;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.72)";
    ctx.fillText(name, 10, yOffset + 18);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath();
    ctx.moveTo(0, yOffset + bandHeight);
    ctx.lineTo(width, yOffset + bandHeight);
    ctx.stroke();
  });
}

function updateLiveLabels(frame) {
  setText("liveWindow", `#${frame.window_index} / ${frame.max_windows}`);
  setText("liveTime", `${fmtNum(frame.window_start_time, 3)}s - ${fmtNum(frame.window_end_time, 3)}s`);
  setText("livePrediction", frame.prediction.toUpperCase());
  setText("liveProbability", fmtPct(frame.probability));
  setText("liveTruth", frame.true_window_label);

  const health = document.getElementById("healthBadge");
  if (frame.alert) {
    health.textContent = "Live alert";
    health.className = "badge danger";
  } else {
    health.textContent = "Live normal";
    health.className = "badge ok";
  }
}

async function stepReplay() {
  try {
    const frame = await getJson(`/demo/replay/window?sample_id=0&window_index=${replayWindow}`);
    drawWaveform(frame.waveform);
    updateLiveLabels(frame);
    replayWindow += 1;
    if (replayWindow > frame.max_windows) {
      replayWindow = 1;
    }
  } catch (error) {
    console.error(error);
    stopReplay();
  }
}

function startReplay() {
  if (replayTimer) return;
  stepReplay();
  replayTimer = setInterval(stepReplay, 600);
}

function stopReplay() {
  if (!replayTimer) return;
  clearInterval(replayTimer);
  replayTimer = null;
}

function resetReplay() {
  stopReplay();
  replayWindow = 1;
  stepReplay();
}

function setLiveConnectionBadge(state) {
  const badge = document.getElementById("liveConnectionBadge");
  if (state === "live") {
    badge.textContent = "Live";
    badge.className = "badge ok";
  } else if (state === "connecting") {
    badge.textContent = "Connecting";
    badge.className = "badge subtle";
  } else {
    badge.textContent = "Offline (no Kafka connection)";
    badge.className = "badge danger";
  }
}

function renderLiveFeedList() {
  const list = document.getElementById("liveFeedList");
  if (!liveFeedEvents.length) {
    list.innerHTML = '<p class="alert-meta">Waiting for live Kafka messages...</p>';
    return;
  }
  list.innerHTML = liveFeedEvents
    .map((event) => {
      const alertClass = event.alert ? "alert-item live-fault" : "alert-item live-normal";
      const topMessage = (event.top_features || [])[0]?.message;
      return `
        <article class="${alertClass}">
          <strong>Window #${event.window_index} | ${event.prediction.toUpperCase()} (${fmtPct(event.probability)})</strong>
          <div class="alert-meta">
            Sample ${event.sample_id}, ${fmtNum(event.window_start_time, 3)}s - ${fmtNum(event.window_end_time, 3)}s,
            latency ${fmtNum(event.latency_ms)} ms, true label: ${event.true_window_label || "unknown"}
          </div>
          ${topMessage ? `<div class="alert-meta">${topMessage}</div>` : ""}
        </article>
      `;
    })
    .join("");
}

function renderUnacknowledgedBadge() {
  const badge = document.getElementById("unacknowledgedBadge");
  if (unacknowledgedAlerts > 0) {
    badge.textContent = `${unacknowledgedAlerts} new alert${unacknowledgedAlerts === 1 ? "" : "s"}`;
    badge.className = "badge danger pulse";
    badge.classList.remove("hidden");
  } else {
    badge.classList.add("hidden");
  }
}

function playAlertSound() {
  try {
    if (!audioContext) return;
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.type = "sine";
    oscillator.frequency.value = 880;
    gain.gain.setValueAtTime(0.15, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.35);
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.35);
  } catch (error) {
    console.error("could not play alert sound", error);
  }
}

async function drawLiveWaveform(event) {
  if (liveWaveformFetchInFlight) return;
  liveWaveformFetchInFlight = true;
  try {
    const frame = await getJson(
      `/demo/replay/window?sample_id=${event.sample_id}&window_index=${event.window_index}`
    );
    drawWaveform(frame.waveform, "liveWaveformCanvas");
  } catch (error) {
    console.error("could not draw live waveform", error);
  } finally {
    liveWaveformFetchInFlight = false;
  }
}

function handleLiveEvent(event) {
  liveFeedMessageCount += 1;
  if (event.alert) {
    liveFeedAlertCount += 1;
    unacknowledgedAlerts += 1;
    renderUnacknowledgedBadge();
    playAlertSound();
  }

  liveFeedEvents.unshift(event);
  if (liveFeedEvents.length > LIVE_FEED_MAX_EVENTS) liveFeedEvents.length = LIVE_FEED_MAX_EVENTS;

  setText("liveFeedMessages", liveFeedMessageCount);
  setText("liveFeedAlerts", liveFeedAlertCount);
  setText("liveFeedLatency", `${fmtNum(event.latency_ms)} ms`);
  setText("liveFeedWindow", `#${event.window_index} (sample ${event.sample_id})`);
  renderLiveFeedList();
  drawLiveWaveform(event);
}

function connectLiveSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${protocol}//${window.location.host}/ws/alerts`;
  setLiveConnectionBadge("connecting");

  liveSocket = new WebSocket(url);

  liveSocket.addEventListener("open", () => {
    liveReconnectDelayMs = 1000;
    setLiveConnectionBadge("live");
  });

  liveSocket.addEventListener("message", (message) => {
    try {
      handleLiveEvent(JSON.parse(message.data));
    } catch (error) {
      console.error("bad live event payload", error);
    }
  });

  liveSocket.addEventListener("close", () => {
    setLiveConnectionBadge("offline");
    setTimeout(connectLiveSocket, liveReconnectDelayMs);
    liveReconnectDelayMs = Math.min(liveReconnectDelayMs * 2, LIVE_RECONNECT_MAX_MS);
  });

  liveSocket.addEventListener("error", () => {
    liveSocket.close();
  });
}

async function loadEpisodeOptions() {
  try {
    const episodes = await getJson("/demo/episodes?limit=12");
    const select = document.getElementById("episodeSelect");
    select.innerHTML = episodes
      .map((ep) => `<option value="${ep.sample_id}">#${ep.sample_id} - type ${ep.sc_type} (${ep.fault_target})</option>`)
      .join("");
  } catch (error) {
    console.error("could not load episode list", error);
  }
}

function ensureAudioContext() {
  if (!audioContext) {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    if (Ctor) audioContext = new Ctor();
  } else if (audioContext.state === "suspended") {
    audioContext.resume();
  }
}

async function triggerReplay() {
  ensureAudioContext();
  const select = document.getElementById("episodeSelect");
  const button = document.getElementById("replayButton");
  const status = document.getElementById("replayStatus");

  if (!select.value) {
    status.textContent = "No episode selected - try reloading the page.";
    return;
  }
  const sampleId = Number(select.value);

  button.disabled = true;
  status.textContent = `Starting replay for episode #${sampleId}...`;
  try {
    const response = await fetch("/demo/kafka-replay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: sampleId, max_windows: 46, sleep_seconds: 0.05 }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `request failed with ${response.status}`);
    }
    status.textContent = `Replaying episode #${sampleId} - watch the live feed below.`;
  } catch (error) {
    console.error(error);
    status.textContent = `Replay failed: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

async function refreshDashboard() {
  const [summary, models, rare, alerts] = await Promise.all([
    getJson("/reports/summary"),
    getJson("/reports/model-comparison"),
    getJson("/reports/realistic-evaluation"),
    getJson("/alerts/incidents?limit=20"),
  ]);
  renderSummary(summary);
  renderModelRows(models);
  renderRareRows(rare);
  renderAlerts(alerts);
}

document.getElementById("refreshButton").addEventListener("click", refreshDashboard);
document.getElementById("playButton").addEventListener("click", startReplay);
document.getElementById("pauseButton").addEventListener("click", stopReplay);
document.getElementById("resetButton").addEventListener("click", resetReplay);
document.getElementById("replayButton").addEventListener("click", triggerReplay);
document.getElementById("acknowledgeButton").addEventListener("click", () => {
  unacknowledgedAlerts = 0;
  renderUnacknowledgedBadge();
});
refreshDashboard().catch((error) => {
  console.error(error);
  document.getElementById("healthBadge").textContent = "Load error";
  document.getElementById("healthBadge").className = "badge danger";
});
stepReplay();
connectLiveSocket();
loadEpisodeOptions();
renderUnacknowledgedBadge();
