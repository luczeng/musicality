"""FastAPI server for the mobile field-capture companion.

Thin wrapper around ``tools/annotator/data.py`` — every read/write goes
through the same functions the desktop annotator uses, so captures made
from a phone are indistinguishable from ones made in the desktop app.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import tools.annotator.data as annotator_data
from tools.annotator.naming import generate_track_id, sanitize_track_name

_SR = 44100
_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="musicality mobile companion")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


class TapAnnotation(BaseModel):
    tap_times: list[float]


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


@app.post("/datasets/{dataset}/tracks")
async def upload_track(
    dataset: str, file: UploadFile = File(...), name: str | None = Form(None)
) -> dict[str, str]:
    raw = await file.read()
    try:
        audio, _ = librosa.load(io.BytesIO(raw), sr=_SR, mono=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"could not decode audio: {exc}"
        ) from exc

    track_id = sanitize_track_name(name) if name else generate_track_id()
    tracks_dir = annotator_data.DATA_DIR / dataset / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(tracks_dir / f"{track_id}.wav"), audio, _SR)

    return {"dataset": dataset, "track_id": track_id}


@app.post("/datasets/{dataset}/tracks/{track_id}/annotations")
def upload_annotation(dataset: str, track_id: str, body: TapAnnotation) -> dict:
    beat_times = np.sort(np.array(body.tap_times, dtype=float))
    track = annotator_data.TrackData(
        dataset_name=dataset,
        track_id=track_id,
        audio_path=str(
            annotator_data.DATA_DIR / dataset / "tracks" / f"{track_id}.wav"
        ),
        tempo=annotator_data.tempo_from_beats(beat_times),
        beat_times=beat_times,
        beat_positions=None,
    )
    annotator_data.save_annotations(track, annotator_data.annotation_path(track))

    return {"dataset": dataset, "track_id": track_id, "tempo": track.tempo}
