# Feasibility Study: Mobile Companion for Tempo Annotation ("Field Capture Companion")

## Context

The desktop annotator (`tools/annotator/`) is a 2,013-line PySide6 app used to record audio, tap-tempo, and hand-annotate beats for the tempo-estimation dataset. The user wants to explore taking this to iPhone/Android — not for full feature parity, but scoped down to a **field capture companion**: record audio and tap tempo on a phone while out and about, then get that data into the same `data/{dataset}/tracks/` + `annotations/*.beats` structure the desktop app already reads (via `tools/annotator/data.py` and `musicality/dataformats`). This report lays out what that requires, the realistic effort, and the big risks to know about before committing.

## Why not just port the PySide6 app?

Researched current tooling (mid-2026) directly, since this determines everything else:

- **PySide6 on Android**: real support exists (`pyside6-android-deploy`, uses buildozer), but the deploy tooling **only runs from a Linux host**, and packaging Qt Widgets apps for Android is still relatively fragile/manual.
- **PySide6 on iOS**: effectively **unsupported**. The only path is the very old Qt 5.15.2 + `pyqtdeploy`, and the Qt team lists iOS support as "under investigation," not shipped.
- **Kivy** (the main Python-native mobile UI alternative) supports both platforms, but **audio recording on iOS via Kivy is explicitly reported as unsupported / stalled** in the ecosystem — a dealbreaker since recording is the entire point of this companion app.
- **BeeWare/Briefcase** supports both platforms with native widgets, but has the same immaturity around mobile audio capture, and iOS builds still require Xcode on a Mac regardless of framework.

**Conclusion**: none of the "reuse the Python UI code" paths are solid for iOS today, and even Android support means real native-toolchain overhead (buildozer, Linux build host, APK signing). Given the scope is deliberately narrow (record + tap tempo, not the full waveform editor), a native/PySide6 port is the wrong tool for this job.

## Recommended approach: mobile web app (PWA), not a native app

Because the actual requirements are just "record audio," "tap a button to capture timestamps," and "get files to the laptop," a **browser-based Progressive Web App** sidesteps the entire native mobile toolchain problem:

- **Recording**: `MediaRecorder` API is well-supported on modern iOS Safari (14.3+) and Android Chrome — no native audio bindings needed.
- **Tap tempo**: trivial in JS — `performance.now()` timestamps on button taps, same math as `tools/annotator/tap_tempo_widget.py` (BPM from last/recent-8/mean/median), just ported to ~30 lines of JS.
- **Sync/storage**: the "backend" is a small FastAPI endpoint added to this same repo. Neither `fastapi` nor `flask`/`uvicorn` is a current project dependency (checked `pyproject.toml`), so standing this up starts with `uv add fastapi uvicorn`. The endpoint can then **directly reuse `tools/annotator/data.py`'s existing beats-parsing/writing functions and `dataformats` path resolution** — specifically `save_annotations`, `annotation_path`, `tempo_from_beats`, and `DATA_DIR` (resolved via `musicality/dataformats/load()`, currently `data_dir: data` per `dataformat.yaml`) — no new annotation format needed.
- **Distribution**: visit a URL, "Add to Home Screen." No App Store / Play Store review, no Apple Developer account, no Xcode/Android Studio required to ship v1.

This is additive — it does not touch the existing PySide6 desktop annotator at all.

### Concrete risks to validate early (spike before building further)
1. **Mic permission over LAN**: hitting a local dev server by `http://<laptop-ip>:port` from a phone may be blocked by mobile browsers' secure-context requirements for mic access. Likely needs a local HTTPS cert (mkcert) or a tunnel (e.g. Tailscale) — worth a 30-minute spike on both an iPhone and Android device before scoping the rest.
2. **Audio format mismatch — re-encode to WAV server-side is required, not optional.** Mobile browsers record compressed containers, not WAV (Android Chrome → webm/opus, iOS Safari → mp4/aac). It's true that the desktop *playback/analysis* path (`librosa.load(path, sr=None, mono=True)`, used in `main_window.py:370` and `tools/inspect_track.py`) can decode arbitrary containers, so that alone wouldn't force a conversion. But `tools/annotator/data.py::load_track()` hardcodes the audio path for custom datasets as `tracks_dir / f"{track_id}.wav"` (`data.py:297`) — it does not glob for extension the way `list_datasets()`/`load_dataset_tracks()` do (those scan `{.wav, .mp3, .flac, .ogg, .aiff}`). A `.webm` or `.m4a` file dropped straight into `tracks/` would show up in dataset listings but fail to load in the desktop annotator. Since `recorder.py`'s `Recorder.stop()` already always writes 44.1kHz mono `.wav` via `soundfile`, matching that format is the path of least resistance: the upload endpoint should re-encode every incoming clip to `.wav` (via `soundfile`/`ffmpeg`) before writing into `tracks/`. This keeps `tools/annotator/data.py` and `main_window.py` completely untouched — zero changes needed there — which preserves the "purely additive" property of this whole approach.
3. **True "field" use (no WiFi)**: if the phone needs to work away from the studio's network (e.g. recording a live band at a venue), the app needs offline-first storage (IndexedDB) with deferred sync when back on WiFi — this is the single biggest swing factor in effort. If "field" really just means "same building, different room," a live upload to a local server is enough and this whole risk disappears.

## Rough effort estimate

| Task | Estimate |
|---|---|
| Spike: validate mic recording + tap timing precision on real iPhone/Android in browser | 0.5–1 day |
| Backend: `uv add fastapi uvicorn`, then upload + tap-annotation endpoints reusing `tools/annotator/data.py` / `musicality/dataformats` (including the required WAV re-encode step) | 0.5–1 day |
| Frontend: record button, tap-tempo button, upload UI (single HTML/JS page) | 1–2 days |
| Networking: local HTTPS/tunnel setup, OR offline-first IndexedDB sync if true no-WiFi field use is needed | 1–3 days (the swing factor) |
| Testing/polish on both iOS Safari and Android Chrome | 1–2 days |
| **Total** | **~4–9 days**, solo, vs. months for any native/full-parity approach |

## What to decide before building
- Does field use genuinely mean "no WiFi at the recording location," or just "not at the desk"? This alone moves the estimate by several days (offline sync vs. simple LAN upload).
- Is App Store/Play Store presence ever needed later, or is "installable web app on my own phone" sufficient indefinitely? (If store presence becomes a hard requirement down the line, that's a separate, much larger native rewrite — not an extension of the PWA.)

## Verification path if this proceeds
- Manual: run the FastAPI server locally, hit it from an actual iPhone and Android phone over the same network, record a short clip, tap a tempo, confirm a `.wav` and `.beats` file land in `data/{dataset}/` in the exact structure `tools/annotator` already reads — then open the desktop annotator and confirm the new track shows up with correct beats.
- No new automated test infrastructure needed beyond extending coverage for any new server endpoints, if this moves into implementation, following the existing pattern in `tests/test_annotator_data.py` (pure-function tests, no I/O or mirdata).
