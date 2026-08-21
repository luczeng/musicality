/**
 * Recording + tap-tempo capture UI. Save writes straight to the IndexedDB
 * queue (queue.js) — this module never makes a network call to persist a
 * capture, only (best-effort) to populate the dataset picker's suggestions.
 */

import {
  addPendingCapture,
  listPending,
  markSynced,
  deletePending,
  flushQueue,
} from "./queue.js";

// Mirrors tools/annotator/tap_tempo_widget.py's stats exactly (same
// warmup/recent-window constants, same population std) so a phone capture
// and a desktop capture behave identically.
const RECENT_N = 8;
const WARMUP = 4;

// Mirrors tools/annotator/data.py's DEFAULT_N_BEATS — tapping always starts
// on beat 1 of an 8-count phrase, so playback clicks cycle 1..8 the same way.
const BEATS_PER_CYCLE = 8;

const datasetInput = document.getElementById("dataset-input");
const datasetOptions = document.getElementById("dataset-options");
const trackNameInput = document.getElementById("track-name-input");
const deviceInput = document.getElementById("device-input");
const structureSwingBtn = document.getElementById("structure-swing-btn");
const structureBluesBtn = document.getElementById("structure-blues-btn");
const structureHelpBtn = document.getElementById("structure-help-btn");
const structureHelpText = document.getElementById("structure-help-text");
const sectionStartBtn = document.getElementById("section-start-btn");
const sectionMidBtn = document.getElementById("section-mid-btn");
const sectionHelpBtn = document.getElementById("section-help-btn");
const sectionHelpText = document.getElementById("section-help-text");
const recordBtn = document.getElementById("record-btn");
const tapBtn = document.getElementById("tap-btn");
const listenBtn = document.getElementById("listen-btn");
const saveBtn = document.getElementById("save-btn");
const statusEl = document.getElementById("status");
const pendingCountEl = document.getElementById("pending-count");
const syncBtn = document.getElementById("sync-btn");
const flushBtn = document.getElementById("flush-btn");
const syncStatusEl = document.getElementById("sync-status");
const tapCountEl = document.getElementById("tap-count");
const tapLastEl = document.getElementById("tap-last");
const tapRecentEl = document.getElementById("tap-recent");
const tapAllEl = document.getElementById("tap-all");
const beatDotEls = document.querySelectorAll(".beat-dot");
const recordsToggleBtn = document.getElementById("records-toggle-btn");
const recordsView = document.getElementById("records-view");
const recordsListEl = document.getElementById("records-list");
const recordsRefreshBtn = document.getElementById("records-refresh-btn");
const recordsPlayBtn = document.getElementById("records-play-btn");
const recordsStatusEl = document.getElementById("records-status");

let audioCtx = null;
let workletModulePromise = null;
let mediaStream = null;
let sourceNode = null;
let workletNode = null;
let silentGainNode = null;
let pcmChunks = [];
let recordSampleRate = null;
let recordedSamples = null;
let recordedBlob = null;
let recordDurationS = null;
let recording = false;
let recordStartCtxTime = null;
let tapTimesS = [];
let flushResolve = null;
let listenSourceNodes = [];
let listening = false;

// Browsers don't expose a real device name (privacy) the way a desktop OS
// does — this guesses a rough label from the user agent as a starting
// point, but the field stays a plain editable input, and whatever the user
// types is remembered in localStorage so it only needs typing once.
const DEVICE_STORAGE_KEY = "musicality-device-name";

function guessDeviceName() {
  const ua = navigator.userAgent;
  if (/iPhone/.test(ua)) return "iPhone";
  if (/iPad/.test(ua)) return "iPad";
  if (/Android/.test(ua)) return "Android phone";
  return "";
}

deviceInput.value = localStorage.getItem(DEVICE_STORAGE_KEY) || guessDeviceName();

let structure = "swing";

function setStructure(value) {
  structure = value;
  structureSwingBtn.classList.toggle("active", value === "swing");
  structureBluesBtn.classList.toggle("active", value === "blues");
}

structureSwingBtn.addEventListener("click", () => setStructure("swing"));
structureBluesBtn.addEventListener("click", () => setStructure("blues"));

structureHelpBtn.addEventListener("click", () => {
  const expanded = structureHelpText.classList.toggle("hidden") === false;
  structureHelpBtn.setAttribute("aria-expanded", String(expanded));
});

// Tapping always starts on count position 1 — that's guaranteed, not
// something to record. This instead tracks whether that first tap also
// happens to be the true start of a section, vs. landing mid-section.
// Defaults to true (the common case); the toggle only needs pressing when
// tapping actually started partway through a section.
let sectionAligned = true;

function setSectionAligned(value) {
  sectionAligned = value;
  sectionStartBtn.classList.toggle("active", value === true);
  sectionMidBtn.classList.toggle("active", value === false);
}

sectionStartBtn.addEventListener("click", () => setSectionAligned(true));
sectionMidBtn.addEventListener("click", () => setSectionAligned(false));

sectionHelpBtn.addEventListener("click", () => {
  const expanded = sectionHelpText.classList.toggle("hidden") === false;
  sectionHelpBtn.setAttribute("aria-expanded", String(expanded));
});

async function loadDatasetOptions() {
  try {
    const response = await fetch("/datasets");
    const datasets = await response.json();
    datasetOptions.innerHTML = "";
    for (const d of datasets) {
      const option = document.createElement("option");
      option.value = d.name;
      datasetOptions.appendChild(option);
    }
  } catch {
    // Offline or server unreachable — fine, dataset-input still works as free text.
  }
}

function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

function std(values, avg) {
  return Math.sqrt(mean(values.map((v) => (v - avg) ** 2)));
}

// Shared with the save handler below, so the mean/median/std persisted as
// track metadata are exactly the "All" figures the user saw on screen.
function tapBpmStats(timesS) {
  const tempos = [];
  for (let i = 1; i < timesS.length; i++) {
    tempos.push(60 / (timesS[i] - timesS[i - 1]));
  }
  const valid = tempos.slice(WARMUP);
  if (valid.length === 0) return null;

  const avg = mean(valid);
  return { valid, mean: avg, median: median(valid), std: std(valid, avg) };
}

function renderTapStats() {
  tapCountEl.textContent = `N: ${tapTimesS.length}`;

  const stats = tapBpmStats(tapTimesS);
  if (!stats) {
    tapLastEl.textContent = "Last: —";
    tapRecentEl.textContent = `Recent (${RECENT_N}): —`;
    tapAllEl.textContent = "All — Mean: —   Median: —   Std: —";
    return;
  }

  tapLastEl.textContent = `Last: ${stats.valid[stats.valid.length - 1].toFixed(1)} BPM`;

  const recent = stats.valid.slice(-RECENT_N);
  tapRecentEl.textContent = `Recent (${RECENT_N}): ${mean(recent).toFixed(1)} BPM`;

  tapAllEl.textContent =
    `All — Mean: ${stats.mean.toFixed(1)}   ` +
    `Median: ${stats.median.toFixed(1)}   ` +
    `Std: ${stats.std.toFixed(2)}`;
}

function resetTapState() {
  tapTimesS = [];
  renderTapStats();
}

async function refreshPendingCount() {
  const pending = await listPending();
  pendingCountEl.textContent = `${pending.length} capture(s) queued locally`;
}

async function startRecording() {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor || !("audioWorklet" in AudioContextCtor.prototype)) {
    throw new Error(
      "This browser doesn't support AudioWorklet — recording needs iOS Safari 14.5+ or a recent Chrome/Android."
    );
  }

  const ctx = getAudioContext();
  await ctx.resume();
  await ensureWorkletLoaded(ctx);

  // AGC/EC/NS off: this is a tempo-annotation source, not a call signal.
  // Browsers' WebRTC audio-processing pipeline (used for AGC/EC/NS) adds
  // its own extra buffering/latency on top of raw passthrough — disabling
  // it also reduces the very kind of variable latency this pipeline exists
  // to eliminate, not just matches desktop's unprocessed capture.
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });

  sourceNode = ctx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(ctx, "pcm-capture-processor", {
    numberOfInputs: 1,
    numberOfOutputs: 1, // + silent gain below — an output-less worklet risks
    outputChannelCount: [1], // not being reliably pulled in some browsers
    channelCount: 1,
    channelCountMode: "explicit",
  });
  silentGainNode = ctx.createGain();
  silentGainNode.gain.value = 0;

  pcmChunks = [];
  workletNode.port.onmessage = (event) => {
    if (event.data.type === "chunk") {
      pcmChunks.push(event.data.samples);
    } else if (event.data.type === "flushed") {
      flushResolve?.();
      flushResolve = null;
    }
  };

  sourceNode.connect(workletNode);
  workletNode.connect(silentGainNode);
  silentGainNode.connect(ctx.destination);

  if (listening) stopListening();
  resetTapState();
  recordedBlob = null;
  recordedSamples = null;
  saveBtn.disabled = true;
  listenBtn.disabled = true;

  recordSampleRate = ctx.sampleRate;
  recordStartCtxTime = ctx.currentTime;
  recordBtn.textContent = "■ Stop";
  tapBtn.disabled = false;
  statusEl.textContent = "Recording…";
}

async function stopRecording() {
  await new Promise((resolve) => {
    flushResolve = resolve;
    workletNode.port.postMessage({ type: "flush" });
  });

  sourceNode.disconnect();
  workletNode.disconnect();
  silentGainNode.disconnect();
  mediaStream.getTracks().forEach((track) => track.stop());

  const totalSamples = pcmChunks.reduce((n, c) => n + c.length, 0);
  const samples = new Float32Array(totalSamples);
  let offset = 0;
  for (const chunk of pcmChunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  pcmChunks = [];

  recordedSamples = samples;
  recordDurationS = totalSamples / recordSampleRate;
  recordedBlob = encodeWavPCM16(samples, recordSampleRate);

  recordBtn.textContent = "● Record";
  tapBtn.disabled = true;
  saveBtn.disabled = false;
  listenBtn.disabled = tapTimesS.length === 0;
  statusEl.textContent = "Recording stopped. Review taps, then Save.";
}

recordBtn.addEventListener("click", async () => {
  if (recording) {
    recording = false;
    await stopRecording();
    return;
  }
  try {
    await startRecording();
    recording = true;
  } catch (err) {
    statusEl.textContent = `Could not access microphone: ${err.message}`;
  }
});

tapBtn.addEventListener("click", () => {
  tapTimesS.push(getAudioContext().currentTime - recordStartCtxTime);
  renderTapStats();
});

function getAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioCtx;
}

// addModule() only needs to run once per context — cached here so repeated
// record/stop cycles within one page session don't re-register it.
function ensureWorkletLoaded(ctx) {
  if (!workletModulePromise) {
    workletModulePromise = ctx.audioWorklet.addModule("/static/pcm-capture-processor.js");
  }
  return workletModulePromise;
}

// Encodes mono Float32 PCM ([-1, 1]) as a 16-bit PCM WAV Blob. Hand-rolled
// since this is a vanilla-JS PWA with no bundler/dependency tooling.
function encodeWavPCM16(samples, sampleRate) {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

function peakAmplitude(audioBuffer) {
  let peak = 0;
  for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < data.length; i++) {
      const abs = Math.abs(data[i]);
      if (abs > peak) peak = abs;
    }
  }
  return peak;
}

// Recordings are captured with AGC off (see startRecording — deliberate,
// to avoid the WebRTC audio-processing pipeline's extra latency), so a
// far-mic or quiet-venue take stays quiet all the way through upload and
// storage. Boost it here at playback time only — this never touches the
// stored WAV bytes, which double as training data for this repo's tempo
// models, so playback loudness and archived amplitude stay independent.
const PLAYBACK_TARGET_PEAK = 0.95;
const MAX_PLAYBACK_GAIN = 12; // cap so a near-silent take isn't blown up into raw noise-floor hiss

function playbackGainFor(audioBuffer) {
  const peak = peakAmplitude(audioBuffer);
  if (peak <= 0) return 1;
  return clamp(PLAYBACK_TARGET_PEAK / peak, 1, MAX_PLAYBACK_GAIN);
}

const LISTEN_IDLE_LABEL = "▶ Listen";

// Whichever button most recently started playback — stopListening() resets
// this one, so it works correctly regardless of which caller (the listen-btn
// handler, the records-view handler, or a natural end-of-track) triggers it.
let activeListenButton = listenBtn;
let activeListenIdleLabel = LISTEN_IDLE_LABEL;

function stopListening() {
  for (const node of listenSourceNodes) {
    try {
      node.onended = null;
      node.stop();
    } catch {
      // Already stopped/ended — fine.
    }
  }
  listenSourceNodes = [];
  listening = false;
  activeListenButton.textContent = activeListenIdleLabel;
  stopBeatDot();
}

// Fires a visual "bip" on one of the 4 beat dots at each tap time, scheduled
// against the Web Audio clock (ctx.currentTime) rather than setTimeout, so
// the dots stay locked to what's actually playing even under main-thread
// jank. The 8-count phrase maps onto 4 dots by pairing counts a beat apart
// by 4 on the same dot (dot 0 = counts 1 & 5, dot 1 = counts 2 & 6, ...) —
// only the true count 1 (not 5) bips green, the rest bip yellow. Each bip
// auto-reverts to idle gray shortly after, so it reads as a blink rather
// than a color that just sits there until the next beat on that dot.
const BIP_DURATION_MS = 130;
const N_BEAT_DOTS = 4;

let beatDotRafId = null;
const beatDotTimeoutIds = new Array(N_BEAT_DOTS).fill(null);

function stopBeatDot() {
  if (beatDotRafId !== null) {
    cancelAnimationFrame(beatDotRafId);
    beatDotRafId = null;
  }
  for (let dotIndex = 0; dotIndex < N_BEAT_DOTS; dotIndex++) {
    if (beatDotTimeoutIds[dotIndex] !== null) {
      clearTimeout(beatDotTimeoutIds[dotIndex]);
      beatDotTimeoutIds[dotIndex] = null;
    }
    beatDotEls[dotIndex].classList.remove("bip-one", "bip-beat");
  }
}

function scheduleBeatDot(ctx, startAt, tapTimes) {
  let nextIndex = 0;

  // ctx.currentTime marks when a sample is scheduled/processed, not when
  // it's actually audible — outputLatency (falling back to the older
  // baseLatency on browsers that don't report it yet) is the browser's own
  // measured estimate of that gap. It reflects the real output path (built-in
  // speaker vs. Bluetooth, which alone can add 100ms+) instead of a guessed
  // constant, so this adapts per device/output rather than needing tuning.
  const outputLatency = ctx.outputLatency ?? ctx.baseLatency ?? 0;

  function tick() {
    const elapsed = ctx.currentTime - startAt + outputLatency;
    while (nextIndex < tapTimes.length && tapTimes[nextIndex] <= elapsed) {
      bip(nextIndex % N_BEAT_DOTS, nextIndex % BEATS_PER_CYCLE === 0);
      nextIndex++;
    }
    if (listening) beatDotRafId = requestAnimationFrame(tick);
  }
  beatDotRafId = requestAnimationFrame(tick);
}

function bip(dotIndex, isOne) {
  const dotEl = beatDotEls[dotIndex];
  if (beatDotTimeoutIds[dotIndex] !== null) clearTimeout(beatDotTimeoutIds[dotIndex]);
  dotEl.classList.remove("bip-one", "bip-beat");
  void dotEl.offsetWidth; // restart the transition even for back-to-back same-type bips
  dotEl.classList.add(isOne ? "bip-one" : "bip-beat");
  beatDotTimeoutIds[dotIndex] = setTimeout(() => {
    dotEl.classList.remove("bip-one", "bip-beat");
    beatDotTimeoutIds[dotIndex] = null;
  }, BIP_DURATION_MS);
}

// Plays a take back and drives the beat dot (see scheduleBeatDot below) in
// sync with each tap — no click sound, watch the dot instead of listening
// for an overlaid click, so the actual recording plays back unaltered. Takes
// its source explicitly (rather than reading recordedSamples/tapTimesS
// directly) so the same pipeline can replay either the just-recorded take or
// a fetched past record. `button`/`idleLabel` identify whichever UI control
// triggered playback, so stopListening() — called both on manual stop and on
// natural end-of-track — resets the right one.
async function startListening(
  samples,
  sampleRate,
  tapTimes,
  button = listenBtn,
  idleLabel = LISTEN_IDLE_LABEL
) {
  if (!samples || !sampleRate) return;
  activeListenButton = button;
  activeListenIdleLabel = idleLabel;

  const ctx = getAudioContext();
  await ctx.resume();

  const audioBuffer = ctx.createBuffer(1, samples.length, sampleRate);
  audioBuffer.copyToChannel(samples, 0);

  const startAt = ctx.currentTime + 0.15;
  listenSourceNodes = [];

  // Gain node boosts a quiet take up toward full scale. playbackGainFor()
  // scales by exactly (target peak / actual peak), so the loudest sample is
  // mathematically guaranteed to land at PLAYBACK_TARGET_PEAK (< 1.0) — no
  // limiter needed, and skipping one avoids its lookahead latency, which
  // would otherwise delay audio reaching the speaker relative to the beat
  // dots below (which schedule off the audio clock directly, not the graph).
  const gainNode = ctx.createGain();
  gainNode.gain.value = playbackGainFor(audioBuffer);
  gainNode.connect(ctx.destination);

  const trackSource = ctx.createBufferSource();
  trackSource.buffer = audioBuffer;
  trackSource.connect(gainNode);
  trackSource.start(startAt);
  trackSource.onended = () => {
    if (listening) stopListening();
  };
  listenSourceNodes.push(trackSource);

  scheduleBeatDot(ctx, startAt, tapTimes.filter((t) => t >= 0));

  listening = true;
  button.textContent = "■ Stop";
}

listenBtn.addEventListener("click", async () => {
  if (listening) {
    stopListening();
    return;
  }
  try {
    statusEl.textContent = "Loading playback…";
    await startListening(recordedSamples, recordSampleRate, tapTimesS);
    statusEl.textContent = "Listening…";
  } catch (err) {
    statusEl.textContent = `Could not play back: ${err.message}`;
  }
});

recordsToggleBtn.addEventListener("click", async () => {
  const opening = recordsView.classList.toggle("hidden") === false;
  if (opening) await loadRecordsList();
});

recordsRefreshBtn.addEventListener("click", loadRecordsList);

recordsListEl.addEventListener("change", () => {
  recordsPlayBtn.disabled = !recordsListEl.value;
});

// Mirrors loadDatasetOptions() above — same fetch-and-degrade-quietly
// pattern, since this view is only reachable manually (not on page load),
// an unreachable server here should read as "nothing to show", not an error.
async function loadRecordsList() {
  const dataset = datasetInput.value.trim();
  if (!dataset) {
    recordsStatusEl.textContent = "Enter a dataset (Author) first.";
    return;
  }
  recordsStatusEl.textContent = "Loading…";
  recordsPlayBtn.disabled = true;
  try {
    const response = await fetch(`/datasets/${encodeURIComponent(dataset)}/tracks`);
    const tracks = await response.json();
    recordsListEl.innerHTML = "";
    for (const t of tracks) {
      const option = document.createElement("option");
      option.value = t.track_id;
      option.textContent = t.has_annotation
        ? `${t.track_id}  (${t.meter || "•"})`
        : `${t.track_id}  (no taps)`;
      option.disabled = !t.has_annotation;
      recordsListEl.appendChild(option);
    }
    recordsStatusEl.textContent = tracks.length ? "" : "No records in this dataset yet.";
  } catch (err) {
    recordsStatusEl.textContent = `Could not load records: ${err.message}`;
  }
}

const RECORDS_PLAY_IDLE_LABEL = "▶ Play";

recordsPlayBtn.addEventListener("click", async () => {
  if (listening) {
    stopListening();
    return;
  }
  const dataset = datasetInput.value.trim();
  const trackId = recordsListEl.value;
  if (!dataset || !trackId) return;

  try {
    recordsStatusEl.textContent = "Loading…";
    const [audioResponse, detail] = await Promise.all([
      fetch(`/datasets/${encodeURIComponent(dataset)}/tracks/${encodeURIComponent(trackId)}/audio`),
      fetch(
        `/datasets/${encodeURIComponent(dataset)}/tracks/${encodeURIComponent(trackId)}/annotations`
      ).then((r) => r.json()),
    ]);
    if (!audioResponse.ok) throw new Error(`audio fetch failed (${audioResponse.status})`);

    const decoded = await getAudioContext().decodeAudioData(await audioResponse.arrayBuffer());
    await startListening(
      decoded.getChannelData(0),
      decoded.sampleRate,
      detail.tap_times,
      recordsPlayBtn,
      RECORDS_PLAY_IDLE_LABEL
    );
    recordsStatusEl.textContent = "Listening…";
  } catch (err) {
    recordsStatusEl.textContent = `Could not play back: ${err.message}`;
  }
});

saveBtn.addEventListener("click", async () => {
  if (!recordedBlob) return;
  const dataset = datasetInput.value.trim();
  if (!dataset) {
    statusEl.textContent = "Enter a dataset name before saving.";
    return;
  }
  const trackName = trackNameInput.value.trim();
  const device = deviceInput.value.trim();
  const bpmStats = tapBpmStats(tapTimesS);

  if (device) localStorage.setItem(DEVICE_STORAGE_KEY, device);

  await addPendingCapture({
    blob: recordedBlob,
    tapTimes: tapTimesS,
    dataset,
    trackName: trackName || null,
    structure,
    sectionAligned,
    device: device || null,
    durationS: recordDurationS,
    bpmStats,
  });

  if (listening) stopListening();
  recordedBlob = null;
  recordedSamples = null;
  recordDurationS = null;
  saveBtn.disabled = true;
  listenBtn.disabled = true;
  trackNameInput.value = "";
  setStructure("swing");
  setSectionAligned(true);
  resetTapState();
  statusEl.textContent = "Saved locally.";
  await refreshPendingCount();
});

async function syncOneCapture(capture) {
  const uploadForm = new FormData();
  uploadForm.append("file", capture.blob, "capture.audio");
  if (capture.trackName) uploadForm.append("name", capture.trackName);

  const uploadResponse = await fetch(
    `/datasets/${encodeURIComponent(capture.dataset)}/tracks`,
    { method: "POST", body: uploadForm }
  );
  if (!uploadResponse.ok) {
    const detail = await uploadResponse.text().catch(() => "");
    throw new Error(`upload failed (${uploadResponse.status}): ${detail}`);
  }
  const { track_id } = await uploadResponse.json();

  const annotationResponse = await fetch(
    `/datasets/${encodeURIComponent(capture.dataset)}/tracks/${encodeURIComponent(track_id)}/annotations`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tap_times: capture.tapTimes,
        structure: capture.structure,
        device: capture.device,
        duration_s: capture.durationS,
        bpm_mean: capture.bpmMean,
        bpm_median: capture.bpmMedian,
        bpm_std: capture.bpmStd,
        section_aligned: capture.sectionAligned,
      }),
    }
  );
  if (!annotationResponse.ok) {
    const detail = await annotationResponse.text().catch(() => "");
    throw new Error(`annotation failed (${annotationResponse.status}): ${detail}`);
  }
}

// Drains the queue.js queue into the steps 5–6 backend endpoints. Failures
// (server unreachable, decode error, ...) leave the capture queued for a
// later retry rather than losing it.
let syncInFlight = false;

// Returns the number of captures successfully synced, so callers (the
// manual Sync button, below) can decide whether to react to it — the
// opportunistic on-load sync just fires this and ignores the result.
async function syncPendingCaptures() {
  // Checked and set synchronously, before any `await`, so the opportunistic
  // on-load sync and a manual Sync click can't both start: the server's
  // auto-generated track id only has microsecond resolution, so two
  // concurrent syncs of the same capture racing each other could otherwise
  // overwrite one another (observed in practice before this guard existed).
  if (syncInFlight) return 0;
  syncInFlight = true;
  syncBtn.disabled = true;

  try {
    const pending = await listPending();
    if (pending.length === 0) {
      syncStatusEl.textContent = "Nothing to sync.";
      return 0;
    }

    syncStatusEl.textContent = `Syncing ${pending.length} capture(s)…`;

    let succeeded = 0;
    let failed = 0;
    for (const capture of pending) {
      try {
        await syncOneCapture(capture);
        await markSynced(capture.id);
        await deletePending(capture.id);
        succeeded++;
      } catch (err) {
        failed++;
        console.warn("[sync] capture failed:", capture.id, err);
      }
    }

    syncStatusEl.textContent =
      `Synced ${succeeded}` + (failed ? `, ${failed} failed (still queued).` : ".");
    await refreshPendingCount();
    return succeeded;
  } finally {
    syncInFlight = false;
    syncBtn.disabled = false;
  }
}

syncBtn.addEventListener("click", async () => {
  const succeeded = await syncPendingCaptures();
  if (succeeded > 0) {
    alert("Thank you for contributing to the swing database!");
  }
});

flushBtn.addEventListener("click", async () => {
  const pending = await listPending();
  if (pending.length === 0) {
    syncStatusEl.textContent = "Queue already empty.";
    return;
  }
  const ok = confirm(
    `Discard ${pending.length} queued capture(s)? This cannot be undone.`
  );
  if (!ok) return;

  await flushQueue();
  syncStatusEl.textContent = `Discarded ${pending.length} capture(s).`;
  await refreshPendingCount();
});

loadDatasetOptions();
refreshPendingCount();
renderTapStats();

// Best-effort preload so the worklet module is already loaded by the time
// the user hits Record — not required for correctness, startRecording()
// awaits ensureWorkletLoaded() itself regardless.
try {
  ensureWorkletLoaded(getAudioContext());
} catch {
  // Unsupported browser — startRecording() will surface the real error.
}

// Opportunistic sync on load — iOS Safari's Background Sync API is
// unreliable, so a manual Sync button (above) has to remain the primary
// trigger; this is just a convenience for the common case of opening the
// app already back on WiFi with captures still queued from the venue.
if (navigator.onLine) {
  syncPendingCaptures();
}
