# Implementation Plan: Mobile Field Capture Companion

> Progress tracker. See `docs/mobilecompanionfeasibility.md` for the design rationale.
> Steps are implemented one at a time — check with the user before starting the next one.

## Context

`docs/mobilecompanionfeasibility.md` recommends building an **offline-first Progressive Web App** rather than a native/PySide6 port: record audio + tap tempo on a phone, store captures locally on-device, and sync them to the same `data/{dataset}/tracks/` + `annotations/*.beats` structure the desktop annotator (`tools/annotator/`) already reads — reusing `tools/annotator/data.py` directly rather than inventing a new format.

This needs **true offline capture**: the phone must be usable with zero network at the recording location (e.g. a venue), syncing back only once it's on WiFi again. That requirement shapes the whole frontend architecture below — it's not a bolt-on.

**Housekeeping note (unrelated, found during research)**: `tests/test_annotator_data.py` currently fails to even collect — it imports `load_annotations` from `tools/annotator/data.py`, which doesn't exist there (`save_annotations` there writes plain-text `.beats`, not the JSON format the test expects). Pre-existing on this branch and on `main`, unrelated to this work — not fixed as part of this plan unless asked.

## Architecture

- **New module**: `tools/mobile_companion/`, parallel to `tools/annotator/` — a FastAPI server that is a thin wrapper around the existing `tools/annotator/data.py` functions (`list_datasets`, `save_annotations`, `annotation_path`, `tempo_from_beats`, `DATA_DIR`) plus a static frontend it serves.
- **Backend reuse, zero changes to `tools/annotator/`**: every write goes through the same functions the desktop app already uses, so the desktop annotator picks up phone-captured tracks with no changes on its side.
- **Frontend**: vanilla HTML/JS PWA (manifest + service worker for installability and offline app-shell caching) — no npm/`package.json` introduced, no JS test runner. Automated tests are backend-only (pytest, mirroring `tests/test_annotator_data.py`'s pure-function style); frontend and full-device flows are verified manually.
- **Offline flow**: install the PWA once while on the same WiFi as the laptop (so the service worker can cache the app shell). At the venue, with no network, record + tap — both get written straight to IndexedDB, never attempted-and-failed over the network. Back on WiFi, hit "Sync" to push queued captures to the backend.

## Steps

### 1. Dependencies + FastAPI skeleton — ✅ DONE
- `uv add fastapi uvicorn python-multipart`.
- `tools/mobile_companion/__init__.py` (empty), `tools/mobile_companion/server.py`: `FastAPI()` app with `GET /health` → `{"status": "ok"}`.
- Test: `tests/test_mobile_companion_server.py` — `TestClient(app).get("/health")` returns 200 and `{"status": "ok"}`. **Passing.**

### 2. Local HTTPS / LAN dev setup — ✅ DONE
- Installed `mkcert` via Homebrew; generated a dev CA + cert covering `localhost`, `127.0.0.1`, and the LAN IP, at `tools/mobile_companion/certs/dev-{cert,key}.pem` (gitignored — machine-local, regenerate if the LAN IP changes).
- Full setup + phone-trust instructions documented in `tools/mobile_companion/README.md`.
- **Manual step still owed by the user** (can't be scripted): run `mkcert -install` in a real terminal to trust the CA in the Mac's system keychain (needs an interactive sudo password), and AirDrop `rootCA.pem` (path via `mkcert -CAROOT`) to the phone to trust it there too.
- Verified: `curl --cacert <mkcert CAROOT>/rootCA.pem https://<lan-ip>:8443/health` returns `{"status":"ok"}` cleanly over TLS.

### 3. Dataset listing endpoint — NOT STARTED
- `GET /datasets` in `server.py`, reusing `list_datasets()` (`tools/annotator/data.py:226`) unchanged. Returns dataset names (+ track/annotation counts) for the frontend's dataset picker.
- Test: `TestClient` call against a monkeypatched `tools.annotator.data.DATA_DIR` pointed at `tmp_path` with a fake dataset folder; assert the response reflects it.

### 4. Track naming helper (pure function) — NOT STARTED
- Extract the sanitization regex already in `recorder.py:105` (`re.sub(r"[^\w\-]", "_", name.strip()) or "recording"`) into a shared pure function so both `recorder.py` and the new endpoint use identical logic — no duplicated regex.
- Add `generate_track_id() -> str`, a timestamp-based fallback id (e.g. `field_20260715_143201`) for quick captures where typing a name on a phone is friction.
- Test: pure unit tests, same style as `tests/test_annotator_data.py` — no I/O, no mocks.

### 5. Audio upload endpoint — NOT STARTED
- `POST /datasets/{dataset}/tracks` — multipart file + optional name. Decode with `librosa.load(io.BytesIO(raw), sr=44100, mono=True)` and write via `soundfile.write` at 44.1kHz mono — matching `recorder.py:13`'s `_SR = 44100` convention, so `load_track()`'s hardcoded `tracks_dir / f"{track_id}.wav"` (`data.py:297`) keeps working untouched.
- Test: `TestClient` posting synthetic WAV bytes (generated in-test with `soundfile`/`numpy`) against a tmp `DATA_DIR`; assert the file lands at the exact path `load_track()` expects and is re-decodable.
- **Deferred to step 10**: real device recordings are webm/opus (Android) or mp4/aac (iOS), not WAV — decoding those depends on `ffmpeg` being installed on the machine running the server. Verify on real devices before trusting the pipeline end-to-end.

### 6. Tap-annotation endpoint — NOT STARTED
- `POST /datasets/{dataset}/tracks/{track_id}/annotations` — JSON body `{"tap_times": [...]}` (seconds, relative to recording start). Builds a `TrackData` and calls the existing `save_annotations(track, annotation_path(track))` (`data.py:341`, `data.py:188`) and `tempo_from_beats` (`data.py:51`) — zero changes to `data.py`.
- Test: `TestClient` posting tap times, then reading back via `load_track()` and asserting `beat_times` round-trips.

### 7. PWA shell — NOT STARTED
- `tools/mobile_companion/static/manifest.json` + a minimal service worker that precaches the app's own HTML/JS/CSS/icons, so the already-installed app opens with zero network. `server.py` mounts `static/` and serves `index.html` at `/`.
- Verification: manual — install to home screen while on WiFi, switch to airplane mode, confirm the app still opens.

### 8. IndexedDB capture queue (JS module) — NOT STARTED
- `static/queue.js`: `addPendingCapture(blob, tapTimes, dataset, trackName)`, `listPending()`, `markSynced(id)`, `deletePending(id)` — isolated from recording/network code.
- No automated JS tests (no JS toolchain in this repo); verified manually alongside step 9.

### 9. Recording + tap-tempo capture UI — NOT STARTED
- `static/app.js`: dataset picker (fetches `/datasets`), record button (`MediaRecorder` → blob), tap-tempo button mirroring `tap_tempo_widget.py`'s stats (last / recent-8 / mean / median / std, `_RECENT_N = 8`, `_WARMUP = 4`). "Save" writes straight to the step 8 queue — never attempts a network call from here.
- Verification: manual, airplane mode — record, tap, save, confirm entries appear in IndexedDB (devtools) with no network activity.

### 10. Sync engine — NOT STARTED
- Manual "Sync" button (plus an opportunistic attempt on page load if online — iOS Safari's Background Sync support is unreliable, so this can't be the only trigger) that iterates the step 8 queue, POSTing each capture to step 5 then step 6, marking synced/removing on success and leaving failures queued for retry.
- Verification (end-to-end, on real devices): install the PWA on an iPhone and an Android phone while on WiFi → airplane mode → record a clip, tap a tempo, save → re-enable WiFi → Sync → confirm a `.wav` and `.beats` file land in `data/{dataset}/` in the exact structure `tools/annotator` reads → open the desktop annotator and confirm the new track appears with correct beats.

## Verification summary
- Steps 1, 3, 4, 5, 6: automated pytest, `TestClient`-based, following `tests/test_annotator_data.py`'s conventions (tmp paths, monkeypatched `DATA_DIR`, no real mirdata/network).
- Steps 2, 7, 8, 9, 10: manual, on real iPhone + Android hardware.
- Full loop closes only at step 10: a phone-captured track must be indistinguishable, from the desktop annotator's point of view, from a track recorded directly in `tools/annotator`.
