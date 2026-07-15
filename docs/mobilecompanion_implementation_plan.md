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

### 2. Remote connectivity + HTTPS (via Tailscale) — ✅ DONE

**Original plan assumed the phone would always be on the same home WiFi as the laptop — that's wrong.** The real requirement is: phone on mobile data, anywhere, sending captures back to the laptop. A plain LAN IP (`192.168.178.96`) only resolves inside the home network, so mobile data can't reach it at all. Revised approach: **Tailscale** — a private mesh VPN app installed on both the laptop and phone, giving the phone a stable way to reach the laptop from any network (WiFi or cellular), plus auto-issued trusted HTTPS certificates for the laptop's Tailscale address (no more manual mkcert cert generation or AirDropping a root CA to the phone).

- **Superseded**: the mkcert-based local cert setup (`tools/mobile_companion/certs/dev-{cert,key}.pem`, the phone-trust instructions in `tools/mobile_companion/README.md`) is no longer the primary connectivity path. Left in place for now as a same-WiFi fallback for quick local dev without Tailscale running; may be removed later once Tailscale is confirmed sufficient on its own.
- **New steps**:
  1. ✅ Install Tailscale on the laptop (`brew install --cask tailscale`) and sign in. Laptop is live: Tailscale IP `100.103.0.56`, hostname `ge-mb001.tail288fcf.ts.net`.
  2. ✅ Phone joined Tailscale as `iphone-13-mini` (100.88.20.31), confirmed via `tailscale status`.
  3. ⬜ Enable Tailscale's HTTPS certificates feature — one-time toggle at https://login.tailscale.com/admin/dns, needs the user's interactive login in a browser. Attempted `tailscale cert` before this was on; got `500: your Tailscale account does not support getting TLS certs`, confirming it's off. Once enabled, run `tailscale cert ge-mb001.tail288fcf.ts.net` from `tools/mobile_companion/certs/` to get a real trusted cert.
  4. ✅ Updated `tools/mobile_companion/README.md` with full Tailscale setup + cert instructions, and the run command using the Tailscale hostname. Old mkcert/LAN approach kept as a documented same-WiFi fallback only.
- Offline-first design (steps 7–10) is unaffected — recordings still queue locally on the phone regardless of connectivity, since even mobile data can be a dead zone at a venue. This step only changes *how the phone finds and trusts the laptop when it does have a connection*.
- **Status**: blocked on two manual user steps (phone login, HTTPS toggle in admin console) before this can be verified end-to-end. Backend feature work (steps 3+) can proceed independently in the meantime since it doesn't depend on connectivity being finished.

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
