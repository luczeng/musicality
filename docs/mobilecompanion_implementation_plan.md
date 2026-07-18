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
  3. ✅ Enabled Tailscale's HTTPS certificates feature in the admin console, then ran `tailscale cert ge-mb001.tail288fcf.ts.net` from `tools/mobile_companion/certs/` — produced `ge-mb001.tail288fcf.ts.net.crt` / `.key` (gitignored, machine-local like the mkcert files).
  4. ✅ Updated `tools/mobile_companion/README.md` with full Tailscale setup + cert instructions (fixed the filename pattern to match what `tailscale cert` actually outputs: `<name>.crt` / `<name>.key`, not `<name>-key.pem`), and the run command using the Tailscale hostname. Old mkcert/LAN approach kept as a documented same-WiFi fallback only.
- Offline-first design (steps 7–10) is unaffected — recordings still queue locally on the phone regardless of connectivity, since even mobile data can be a dead zone at a venue. This step only changes *how the phone finds and trusts the laptop when it does have a connection*.
- **Verified**: ran the server with `--ssl-keyfile`/`--ssl-certfile` pointed at the Tailscale cert, `curl -v https://ge-mb001.tail288fcf.ts.net:8443/health` came back with `SSL certificate verify ok`, issuer `Let's Encrypt`, and `{"status":"ok"}` — a real trusted cert, no custom CA needed, no phone-side trust setup required. Not yet tested from the phone's Tailscale/Safari app directly (should work identically since Tailscale + Let's Encrypt handle trust for any device on the tailnet), worth a quick manual check whenever convenient.

### 3. Dataset listing endpoint — ✅ DONE
- `GET /datasets` in `server.py`, reusing `list_datasets()` (`tools/annotator/data.py:226`) unchanged. Returns each dataset's `name`, `n_tracks`, `n_annotations` for the frontend's dataset picker.
- Test: `tests/test_mobile_companion_server.py::TestDatasets` — `TestClient` calls against a monkeypatched `tools.annotator.data.DATA_DIR` pointed at `tmp_path`, covering an empty data dir and a fake dataset folder with tracks + annotations. **Passing.**

### 4. Track naming helper (pure function) — ✅ DONE
- New `tools/annotator/naming.py`: `sanitize_track_name(name)` (the regex formerly inlined at `recorder.py:105`) and `generate_track_id() -> str`, a timestamp-based fallback id (`field_YYYYMMDD_HHMMSS`) for quick captures where typing a name on a phone is friction.
- `recorder.py` now imports `sanitize_track_name` instead of inlining the regex — single source of truth, ready for the step 5 upload endpoint to import the same module.
- Test: `tests/test_annotator_naming.py`, pure unit tests, same style as `tests/test_annotator_data.py` — no I/O, no mocks. **Passing (9 tests).**

### 5. Audio upload endpoint — ✅ DONE
- `POST /datasets/{dataset}/tracks` in `server.py` — multipart `file` + optional `name` form field. Decodes with `librosa.load(io.BytesIO(raw), sr=44100, mono=True)` and writes via `soundfile.write` at 44.1kHz mono — matching `recorder.py`'s `_SR = 44100` convention, so `load_track()`'s hardcoded `tracks_dir / f"{track_id}.wav"` (`data.py:297`) keeps working untouched. Track id is `sanitize_track_name(name)` when a name is given, else `generate_track_id()` (both from step 4's `naming.py`). Creates `{dataset}/tracks/` if the dataset doesn't exist yet. Malformed audio returns `400`.
- Test: `tests/test_mobile_companion_server.py::TestUploadTrack` — `TestClient` posting synthetic WAV bytes (generated in-test with `soundfile`/`numpy`, at a different input sample rate to exercise resampling) against a tmp `DATA_DIR`; covers the exact output path, resampling to 44.1kHz, named vs. auto-generated ids, new-dataset creation, and the 400 error path. **Passing (6 tests).**
- **Deferred to step 10**: real device recordings are webm/opus (Android) or mp4/aac (iOS), not WAV — decoding those depends on `ffmpeg` being installed on the machine running the server. Verify on real devices before trusting the pipeline end-to-end.

### 6. Tap-annotation endpoint — ✅ DONE
- `POST /datasets/{dataset}/tracks/{track_id}/annotations` in `server.py` — JSON body `{"tap_times": [...]}` (seconds, relative to recording start), validated via a `TapAnnotation` pydantic model. Sorts the tap times (`np.sort`, matching `add_beat`'s convention that `beat_times` stays sorted ascending), builds a `TrackData`, and calls the existing `save_annotations(track, annotation_path(track))` (`data.py:341`, `data.py:188`) and `tempo_from_beats` (`data.py:51`) — zero changes to `data.py`. Returns the resulting `tempo`.
- Test: `tests/test_mobile_companion_server.py::TestUploadAnnotation` — `TestClient` posts tap times (after uploading a clip via step 5's endpoint so `load_track()` can resolve the track), then reads back via `load_track()` asserting `beat_times` round-trips, out-of-order taps get sorted, and the returned tempo matches `tempo_from_beats`. **Passing (4 tests).**

### 7. PWA shell — ✅ DONE
- `tools/mobile_companion/static/`: `manifest.json` (name/icons/`display: standalone`), `sw.js` (cache-first service worker, precaches `/`, `/manifest.json`, and both icon sizes on install, purges stale cache versions on activate), `index.html` (minimal placeholder shell that registers the service worker — no recording UI yet, that's step 9), and generated `icon-192.png`/`icon-512.png`.
- `server.py`: mounts `static/` at `/static` (`StaticFiles`) for the icons, and serves `index.html`/`manifest.json`/`sw.js` each from their own root-scoped route (`sw.js` must be served from `/`, not `/static/sw.js`, so its default scope covers the whole app rather than just `/static/`).
- **Verified on real hardware**: loaded the page in Safari on the iPhone over Tailscale HTTPS, waited for the service worker to register, switched to Airplane Mode, reloaded the same tab — page still loaded from cache instead of Safari's offline error. Confirms the core mechanism works. (Tested via reload-in-tab rather than full "Add to Home Screen" — that install step is decoupled from the caching behavior and can be revisited later if a real home-screen icon is wanted; not required for steps 8–10.)

### 8. IndexedDB capture queue (JS module) — ✅ DONE
- `static/queue.js`: `addPendingCapture(blob, tapTimes, dataset, trackName)`, `listPending()` (unsynced only, oldest first), `markSynced(id)`, `deletePending(id)` — isolated from recording/network code, single object store (`captures`, keyed by a `crypto.randomUUID()` id), Blobs stored directly (IndexedDB supports this natively).
- No automated JS tests (no JS toolchain in this repo, per the project's stated approach) — checked with `node --check` for syntax only, and confirmed `server.py`'s static mount serves it (`GET /static/queue.js` → 200, `text/javascript`). Full functional verification (exercising it against real captures) deferred to step 9, once the recording UI actually imports and calls it — per plan.
- **Note for step 9**: `queue.js` isn't referenced from `index.html` yet and isn't in `sw.js`'s `APP_SHELL` precache list — both need updating once `app.js` (step 9) imports it, so it's still available offline.

### 9. Recording + tap-tempo capture UI — ✅ IMPLEMENTED, ON-DEVICE CHECK PENDING
- `static/app.js`: dataset field (`<input list>` fed by `/datasets` when reachable, but always usable as free text — so it degrades gracefully offline instead of blocking capture), track-name field (optional — blank lets the server auto-generate an id), Record/Stop button (`MediaRecorder`, format left to the browser default per step 5's deferred webm/opus vs. mp4/aac note), Tap button (enabled only while recording), live stats display (last / recent-8 / mean / median / std) that's a line-for-line port of `tap_tempo_widget.py`'s math (same `_RECENT_N = 8` / `_WARMUP = 4`, same population `std`) — cross-checked against the Python widget's numpy output on synthetic tap data and confirmed matching to floating-point precision. Save converts the raw tap timestamps to elapsed seconds since recording start and calls `queue.js`'s `addPendingCapture` — no `fetch` in that path at all.
- `index.html` rewritten with the actual UI markup + inline CSS (no separate stylesheet needed); `sw.js`'s `APP_SHELL` now also precaches `/static/app.js` and `/static/queue.js`, and its cache name bumped to `v2` so the phone that already installed the v1 shell (step 7) picks up this update rather than staying stuck on the placeholder page.
- Verified so far: `node --check` on all three JS files, a numeric cross-check of the tap-stats math against the Python widget (matched to float precision), and the running server — `/`, `/static/app.js`, `/static/queue.js` all 200, `/sw.js` reports `v2`, `/datasets` still returns real data. Full test suite still 88 passed.
- **Still needed (manual, on real hardware, airplane mode)**: reload the app on the phone (picks up the v2 shell), record a clip, tap along, hit Save, and confirm an entry appears in IndexedDB (Safari Web Inspector → Storage, if connected to a Mac) with no network activity during the whole flow.

### 10. Sync engine — ✅ IMPLEMENTED, ON-DEVICE END-TO-END CHECK PENDING
- `static/app.js`: `syncPendingCaptures()` drains `queue.js`'s pending list — for each capture, POSTs the blob to step 5's endpoint, then the tap times to step 6's using the returned `track_id`, then `markSynced` + `deletePending` on success. Failures (any thrown error) leave the capture queued for retry, no data lost. Wired to both a manual **Sync** button and an opportunistic call on page load when `navigator.onLine` (per the plan's note: iOS Safari's Background Sync API is unreliable, so the button has to stay the primary trigger regardless).
- `index.html` gained the Sync button + a status line; `sw.js`'s cache bumped to `v3` (again, so an already-installed phone picks up the new button rather than being stuck on the step-9 shell).
- **Bug found and fixed during smoke-testing**: `upload_track` (step 5) was decoding straight from an in-memory `BytesIO`, which works for WAV (libsndfile handles it natively) but silently breaks for the compressed formats real phones actually record — webm/opus (Android) or mp4/aac (iOS) — because those need soundfile's audioread/ffmpeg fallback, and ffmpeg is invoked as a subprocess against a real file path, not a byte stream. Confirmed the failure by generating a synthetic `.webm` clip (`ffmpeg -c:a libopus`) and POSTing it exactly as `app.js`'s `syncOneCapture` would — got `400: Format not recognised`. Fixed by writing the upload to a `tempfile.NamedTemporaryFile` before decoding (`server.py`); re-ran the same webm upload → annotation → dataset-listing sequence end-to-end afterward and it now works correctly (`.wav` + `.beats` land at the right paths, dataset counts update). This is exactly the risk step 5 flagged as deferred — now verified for webm specifically; not yet independently confirmed for iOS's mp4/aac, though the fix isn't format-specific so it should behave the same way.
- Full pytest suite: 88 passed (existing WAV-based tests unaffected by the tempfile change).
- **Still needed (manual, on real hardware, full loop)**: on the iPhone (and ideally an Android phone) — WiFi → airplane mode → record + tap + Save → back on WiFi → tap Sync → confirm the `.wav`/`.beats` land in `data/{dataset}/` → open the desktop annotator and confirm the track appears with correct beats. This is the point where the whole plan's loop actually closes.

## Verification summary
- Steps 1, 3, 4, 5, 6: automated pytest, `TestClient`-based, following `tests/test_annotator_data.py`'s conventions (tmp paths, monkeypatched `DATA_DIR`, no real mirdata/network).
- Steps 2, 7, 8, 9, 10: manual, on real iPhone + Android hardware.
- Full loop closes only at step 10: a phone-captured track must be indistinguishable, from the desktop annotator's point of view, from a track recorded directly in `tools/annotator`.
