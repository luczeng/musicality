/**
 * Recording + tap-tempo capture UI. Save writes straight to the IndexedDB
 * queue (queue.js) — this module never makes a network call to persist a
 * capture, only (best-effort) to populate the dataset picker's suggestions.
 */

import { addPendingCapture, listPending, markSynced, deletePending } from "./queue.js";

// Mirrors tools/annotator/tap_tempo_widget.py's stats exactly (same
// warmup/recent-window constants, same population std) so a phone capture
// and a desktop capture behave identically.
const RECENT_N = 8;
const WARMUP = 4;

const datasetInput = document.getElementById("dataset-input");
const datasetOptions = document.getElementById("dataset-options");
const trackNameInput = document.getElementById("track-name-input");
const recordBtn = document.getElementById("record-btn");
const tapBtn = document.getElementById("tap-btn");
const saveBtn = document.getElementById("save-btn");
const statusEl = document.getElementById("status");
const pendingCountEl = document.getElementById("pending-count");
const tapCountEl = document.getElementById("tap-count");
const tapLastEl = document.getElementById("tap-last");
const tapRecentEl = document.getElementById("tap-recent");
const tapAllEl = document.getElementById("tap-all");

let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
let recordedBlob = null;
let recordStartTime = null;
let tapTimestampsMs = [];

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

function renderTapStats() {
  tapCountEl.textContent = `N: ${tapTimestampsMs.length}`;

  const tempos = [];
  for (let i = 1; i < tapTimestampsMs.length; i++) {
    const intervalS = (tapTimestampsMs[i] - tapTimestampsMs[i - 1]) / 1000;
    tempos.push(60 / intervalS);
  }
  const valid = tempos.slice(WARMUP);

  if (valid.length === 0) {
    tapLastEl.textContent = "Last: —";
    tapRecentEl.textContent = `Recent (${RECENT_N}): —`;
    tapAllEl.textContent = "All — Mean: —   Median: —   Std: —";
    return;
  }

  tapLastEl.textContent = `Last: ${valid[valid.length - 1].toFixed(1)} BPM`;

  const recent = valid.slice(-RECENT_N);
  tapRecentEl.textContent = `Recent (${RECENT_N}): ${mean(recent).toFixed(1)} BPM`;

  const allMean = mean(valid);
  tapAllEl.textContent =
    `All — Mean: ${allMean.toFixed(1)}   ` +
    `Median: ${median(valid).toFixed(1)}   ` +
    `Std: ${std(valid, allMean).toFixed(2)}`;
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
  const tapTimesS = tapTimestampsMs.map((t) => (t - recordStartTime) / 1000);

  await addPendingCapture(recordedBlob, tapTimesS, dataset, trackName || null);

  recordedBlob = null;
  saveBtn.disabled = true;
  trackNameInput.value = "";
  resetTapState();
  statusEl.textContent = "Saved locally.";
  await refreshPendingCount();
});

loadDatasetOptions();
refreshPendingCount();
renderTapStats();
