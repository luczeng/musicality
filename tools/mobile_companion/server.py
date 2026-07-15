"""FastAPI server for the mobile field-capture companion.

Thin wrapper around ``tools/annotator/data.py`` — every read/write goes
through the same functions the desktop annotator uses, so captures made
from a phone are indistinguishable from ones made in the desktop app.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="musicality mobile companion")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
