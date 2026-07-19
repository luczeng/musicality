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

const datasetInput = document.getElementById("dataset-input");
const datasetOptions = document.getElementById("dataset-options");
const trackNameInput = document.getElementById("track-name-input");
const deviceInput = document.getElementById("device-input");
const structureSwingBtn = document.getElementById("structure-swing-btn");
const structureBluesBtn = document.getElementById("structure-blues-btn");
const structureHelpBtn = document.getElementById("structure-help-btn");
const structureHelpText = document.getElementById("structure-help-text");
const recordBtn = document.getElementById("record-btn");
const tapBtn = document.getElementById("tap-btn");
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

let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
let recordedBlob = null;
let recordStartTime = null;
let recordDurationS = null;
let tapTimestampsMs = [];

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
function tapBpmStats(timestampsMs) {
  const tempos = [];
  for (let i = 1; i < timestampsMs.length; i++) {
    const intervalS = (timestampsMs[i] - timestampsMs[i - 1]) / 1000;
    tempos.push(60 / intervalS);
  }
  const valid = tempos.slice(WARMUP);
  if (valid.length === 0) return null;

  const avg = mean(valid);
  return { valid, mean: avg, median: median(valid), std: std(valid, avg) };
}

function renderTapStats() {
  tapCountEl.textContent = `N: ${tapTimestampsMs.length}`;

  const stats = tapBpmStats(tapTimestampsMs);
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
  tapTimestampsMs = [];
  renderTapStats();
}

async function refreshPendingCount() {
  const pending = await listPending();
  pendingCountEl.textContent = `${pending.length} capture(s) queued locally`;
}

async function startRecording() {
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(mediaStream);
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
  };
  mediaRecorder.onstop = () => {
    recordedBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
    saveBtn.disabled = false;
    mediaStream.getTracks().forEach((track) => track.stop());
  };

  recordStartTime = performance.now();
  resetTapState();
  recordedBlob = null;
  saveBtn.disabled = true;

  mediaRecorder.start();
  recordBtn.textContent = "■ Stop";
  tapBtn.disabled = false;
  statusEl.textContent = "Recording…";
}

function stopRecording() {
  mediaRecorder.stop();
  recordDurationS = (performance.now() - recordStartTime) / 1000;
  recordBtn.textContent = "● Record";
  tapBtn.disabled = true;
  statusEl.textContent = "Recording stopped. Review taps, then Save.";
}

recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecording();
    return;
  }
  try {
    await startRecording();
  } catch (err) {
    statusEl.textContent = `Could not access microphone: ${err.message}`;
  }
});

tapBtn.addEventListener("click", () => {
  tapTimestampsMs.push(performance.now());
  renderTapStats();
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
  const tapTimesS = tapTimestampsMs.map((t) => (t - recordStartTime) / 1000);

  if (device) localStorage.setItem(DEVICE_STORAGE_KEY, device);

  await addPendingCapture(
    recordedBlob,
    tapTimesS,
    dataset,
    trackName || null,
    structure,
    device || null
  );

  recordedBlob = null;
  saveBtn.disabled = true;
  trackNameInput.value = "";
  setStructure("swing");
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
    throw new Error(`upload failed (${uploadResponse.status})`);
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
      }),
    }
  );
  if (!annotationResponse.ok) {
    throw new Error(`annotation failed (${annotationResponse.status})`);
  }
}

// Drains the queue.js queue into the steps 5–6 backend endpoints. Failures
// (server unreachable, decode error, ...) leave the capture queued for a
// later retry rather than losing it.
let syncInFlight = false;

async function syncPendingCaptures() {
  // Checked and set synchronously, before any `await`, so the opportunistic
  // on-load sync and a manual Sync click can't both start: the server's
  // auto-generated track id only has microsecond resolution, so two
  // concurrent syncs of the same capture racing each other could otherwise
  // overwrite one another (observed in practice before this guard existed).
  if (syncInFlight) return;
  syncInFlight = true;
  syncBtn.disabled = true;

  try {
    const pending = await listPending();
    if (pending.length === 0) {
      syncStatusEl.textContent = "Nothing to sync.";
      return;
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
      } catch {
        failed++;
      }
    }

    syncStatusEl.textContent =
      `Synced ${succeeded}` + (failed ? `, ${failed} failed (still queued).` : ".");
    await refreshPendingCount();
  } finally {
    syncInFlight = false;
    syncBtn.disabled = false;
  }
}

syncBtn.addEventListener("click", syncPendingCaptures);

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

// Opportunistic sync on load — iOS Safari's Background Sync API is
// unreliable, so a manual Sync button (above) has to remain the primary
// trigger; this is just a convenience for the common case of opening the
// app already back on WiFi with captures still queued from the venue.
if (navigator.onLine) {
  syncPendingCaptures();
}
