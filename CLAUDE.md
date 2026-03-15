# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`musicality` is a Python library for working with the [Million Song Dataset (MSD)](http://millionsongdataset.com/) — a collection of audio features and metadata for a million contemporary music tracks. The core data format is HDF5 (via PyTables).

## Commands

```bash
# Install with test dependencies
pip install -e ".[test]"

# Run tests
pytest tests/

# Run a single test
pytest tests/test_get_beat.py

# Format code
black musicality/

# Display a song's HDF5 contents
python musicality/display_song.py <path-to-.h5-file>
```

## Architecture

### HDF5 Data Layer

The primary interface for reading song data is `hdf5_getters.py`, which exposes ~55 getter functions (e.g. `get_tempo`, `get_beats_start`, `get_artist_name`). All getters accept an open HDF5 file handle and an optional `songidx` integer for aggregate files.

`hdf5_descriptors.py` defines the PyTables schema:
- `SongMetaData` — artist/song identifiers and text fields
- `SongAnalysis` — audio features (tempo, loudness, key, mode, energy, danceability) and time-series arrays (beats, bars, segments, sections, tatums)
- `SongMusicBrainz` — MusicBrainz-sourced IDs and tags

`hdf5_utils.py` handles file creation and low-level read/write; `hdf5_getters.py` is the public API.

### Aggregate vs. Single-Song Files

Single-song `.h5` files hold one track. `create_aggregate_file.py` merges multiple single-song files into one aggregate `.h5`, where each table row is a song. Getters use `songidx` to index into aggregate files.

### Dataset Creation (`DatasetCreation/`)

`dataset_creator.py` downloads and builds the dataset by querying the Echo Nest API and an optional local MusicBrainz PostgreSQL database (`MBrainzDB/query.py`). It uses multiprocessing with file-based locking for parallel downloads.

### Format Converters

- `enpyapi_to_hdf5.py` — Echo Nest API response → HDF5
- `hdf5_to_matfile.py` — HDF5 → MATLAB `.mat`
- `create_summary_file.py` — generates summary statistics across the dataset

### File Discovery

`utils.py` provides `get_all_files(basedir, ext='.h5')` for recursively walking the dataset directory tree (files are organized in a `A/A/A/...` three-level prefix hierarchy).

## Dependencies

- `numpy` — array handling for time-series audio features
- `tables` (PyTables) — HDF5 file I/O

## Notes

- The codebase was originally written for Python 2 (`print` statements, `Queue` module, old-style exception syntax). Some Python 3 compatibility issues may exist.
- The sample dataset lives in `data/MillionSongSubset/`; the full subset archive is `millionsongsubset.tar.gz`.
