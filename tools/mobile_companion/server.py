"""FastAPI server for the mobile field-capture companion.

Thin wrapper around ``tools/annotator/data.py`` — every read/write goes
through the same functions the desktop annotator uses, so captures made
from a phone are indistinguishable from ones made in the desktop app.
"""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import musicality.dataformats as dataformats
import tools.annotator.data as annotator_data
from tools.annotator.naming import generate_track_id, sanitize_track_name

_SR = 44100
_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="musicality mobile companion")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


class TapAnnotation(BaseModel):
    tap_times: list[float]
    structure: str | None = None
    device: str | None = None
    duration_s: float | None = None
    bpm_mean: float | None = None
    bpm_median: float | None = None
    bpm_std: float | None = None
    section_aligned: bool | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/manifest.json")
def manifest() -> FileResponse:
    return FileResponse(
        _STATIC_DIR / "manifest.json", media_type="application/manifest+json"
    )


@app.get("/sw.js")
def service_worker() -> FileResponse:
    return FileResponse(_STATIC_DIR / "sw.js", media_type="application/javascript")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/datasets")
def datasets() -> list[dict]:
    return [
        {
            "name": info.name,
            "n_tracks": info.n_tracks,
            "n_annotations": info.n_annotations,
        }
        for info in annotator_data.list_datasets()
    ]


@app.get("/datasets/{dataset}/tracks")
def list_tracks(dataset: str) -> list[dict]:
    try:
        track_ids = annotator_data.load_dataset_tracks(dataset)
    except Exception:
        # No tracks/ dir yet and not a recognized mirdata name either — the
        # normal state for a dataset nobody has saved a phone recording to.
        return []

    return [
        {
            "track_id": track_id,
            "has_annotation": annotator_data.has_annotation(dataset, track_id),
            "meter": annotator_data.annotation_meter_label(dataset, track_id),
        }
        for track_id in track_ids
    ]


@app.post("/datasets/{dataset}/tracks")
async def upload_track(
    dataset: str, file: UploadFile = File(...), name: str | None = Form(None)
) -> dict[str, str]:
    raw = await file.read()
    try:
        # app.js always uploads a WAV — raw PCM captured via Web Audio, no
        # compressed container — so soundfile reads it directly from memory;
        # no ffmpeg/tempfile fallback needed. This also means the endpoint
        # fails loudly on anything else instead of silently limping through
        # a codec-decode fallback.
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"could not decode audio: {exc}"
        ) from exc

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mixdown, in case a future client sends stereo

    if sr != _SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=_SR).astype(np.float32)

    track_id = sanitize_track_name(name) if name else generate_track_id()
    tracks_dir = annotator_data.DATA_DIR / dataset / dataformats.FORMAT.tracks_dirname
    tracks_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(tracks_dir / f"{track_id}.wav"), audio, _SR)

    return {"dataset": dataset, "track_id": track_id}


@app.post("/datasets/{dataset}/tracks/{track_id}/annotations")
def upload_annotation(dataset: str, track_id: str, body: TapAnnotation) -> dict:
    beat_times = np.sort(np.array(body.tap_times, dtype=float))
    beat_positions = annotator_data.cycle_positions(
        len(beat_times), annotator_data.DEFAULT_N_BEATS
    )
    track = annotator_data.TrackData(
        dataset_name=dataset,
        track_id=track_id,
        audio_path=str(
            annotator_data.DATA_DIR
            / dataset
            / dataformats.FORMAT.tracks_dirname
            / f"{track_id}.wav"
        ),
        tempo=annotator_data.tempo_from_beats(beat_times),
        beat_times=beat_times,
        beat_positions=beat_positions,
    )
    annotator_data.save_annotations(track, annotator_data.annotation_path(track))

    metadata_fields = (
        body.structure,
        body.device,
        body.duration_s,
        body.bpm_mean,
        body.bpm_median,
        body.bpm_std,
        body.section_aligned,
    )
    if any(field is not None for field in metadata_fields):
        metadata = (
            annotator_data.load_metadata(dataset, track_id)
            or annotator_data.TrackMetadata()
        )
        if body.structure is not None:
            metadata.structure = body.structure
        if body.device is not None:
            metadata.device = body.device
        if body.duration_s is not None:
            metadata.duration_s = body.duration_s
        if body.bpm_mean is not None:
            metadata.bpm_mean = body.bpm_mean
        if body.bpm_median is not None:
            metadata.bpm_median = body.bpm_median
        if body.bpm_std is not None:
            metadata.bpm_std = body.bpm_std
        if body.section_aligned is not None:
            metadata.section_aligned = body.section_aligned
        annotator_data.save_metadata(dataset, track_id, metadata)

    return {"dataset": dataset, "track_id": track_id, "tempo": track.tempo}


@app.get("/datasets/{dataset}/tracks/{track_id}/annotations")
def get_annotation(dataset: str, track_id: str) -> dict:
    track = annotator_data.load_track(dataset, track_id)
    metadata = annotator_data.load_metadata(dataset, track_id)
    return {
        "track_id": track_id,
        "tap_times": track.beat_times.tolist(),
        "tempo": annotator_data.tempo_from_beats(track.beat_times),
        "structure": metadata.structure if metadata else None,
        "device": metadata.device if metadata else None,
        "duration_s": metadata.duration_s if metadata else None,
        "bpm_mean": metadata.bpm_mean if metadata else None,
        "bpm_median": metadata.bpm_median if metadata else None,
        "bpm_std": metadata.bpm_std if metadata else None,
        "section_aligned": metadata.section_aligned if metadata else None,
    }


@app.get("/datasets/{dataset}/tracks/{track_id}/audio")
def get_audio(dataset: str, track_id: str) -> FileResponse:
    path = (
        annotator_data.DATA_DIR
        / dataset
        / dataformats.FORMAT.tracks_dirname
        / f"{track_id}.wav"
    )
    if not path.exists():
        raise HTTPException(status_code=404, detail="track audio not found")

    return FileResponse(path, media_type="audio/wav")
