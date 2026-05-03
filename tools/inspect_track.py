#!/usr/bin/env python3
"""Inspect a dataset track: visualise waveform + beat annotations and/or
play back the audio with an audible click track and a live metronome window.

Usage
-----
    # both visual and audio (default when neither flag is given)
    uv run python tools/inspect_track.py --dataset ballroom

    # pick a specific track
    uv run python tools/inspect_track.py --dataset ballroom --track Media-105901

    # visual only
    uv run python tools/inspect_track.py --dataset ballroom --visual

    # audio only
    uv run python tools/inspect_track.py --dataset brid --audio
"""

import argparse
import random
import sys
import time
from pathlib import Path

import mirdata
import numpy as np

import musicality.dataformats as dataformats

DATA_DIR = Path(__file__).parent.parent / dataformats.load().data_dir

COLOR_INACTIVE = "#444444"
COLOR_BEAT     = "#FFD700"   # yellow
COLOR_DOWNBEAT = "#44CC44"   # green


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_track(dataset_name: str, track_id: str):
    ds = mirdata.initialize(dataset_name, data_home=str(DATA_DIR / dataset_name))
    track = ds.track(track_id)

    beat_times = None
    beat_positions = None
    if track.beats is not None:
        beat_times = track.beats.times
        beat_positions = getattr(track.beats, "positions", None)

    return track.audio_path, track.tempo, beat_times, beat_positions


def pick_track(dataset_name: str, track_id: str | None) -> str:
    ds = mirdata.initialize(dataset_name, data_home=str(DATA_DIR / dataset_name))
    if track_id is not None:
        if track_id not in ds.track_ids:
            print(f"Track '{track_id}' not found in '{dataset_name}'.")
            print(f"Available IDs (sample): {ds.track_ids[:5]}")
            sys.exit(1)
        return track_id
    return random.choice(ds.track_ids)


# ---------------------------------------------------------------------------
# Visual inspection
# ---------------------------------------------------------------------------

def plot_track(audio_path: str, tempo: float, beat_times, beat_positions):
    import librosa
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(audio) / sr
    times = np.linspace(0, duration, len(audio))

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(times, audio, color="steelblue", linewidth=0.4, alpha=0.8)
    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    title = Path(audio_path).stem
    if tempo is not None:
        title += f"  —  {tempo:.1f} BPM"
    ax.set_title(title)

    if beat_times is not None:
        for i, t in enumerate(beat_times):
            pos = beat_positions[i] if beat_positions is not None else None
            is_downbeat = (pos == 1) if pos is not None else False
            ax.axvline(t, color="red" if is_downbeat else "orange",
                       linewidth=1.2 if is_downbeat else 0.7, alpha=0.8)

        handles = [
            Line2D([0], [0], color="red",    linewidth=1.2, label="downbeat"),
            Line2D([0], [0], color="orange", linewidth=0.7, label="beat"),
        ] if beat_positions is not None else [
            Line2D([0], [0], color="orange", linewidth=0.7, label="beat"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8)

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(axis="x", which="major", linestyle="--", alpha=0.3)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Metronome window
# ---------------------------------------------------------------------------

def _beats_per_bar(beat_positions) -> int:
    if beat_positions is None or len(beat_positions) == 0:
        return 4
    return int(max(beat_positions))


def run_metronome(beat_times, beat_positions, total_duration: float):
    """Animate a row of dots in sync with already-started audio playback.

    Assumes sd.play() has been called immediately before this function.
    Returns when playback is done.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = _beats_per_bar(beat_positions)
    dot_r = 0.28
    fig_w = max(n * 1.4, 3.0)

    fig, ax = plt.subplots(figsize=(fig_w, 1.6))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.3)

    circles = []
    for i in range(n):
        c = mpatches.Circle((i, 0), dot_r, color=COLOR_INACTIVE, zorder=2)
        ax.add_patch(c)
        circles.append(c)

    plt.ion()
    plt.show()

    start = time.perf_counter()
    active_beat = -1

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= total_duration:
            break

        # find the most recent beat index
        idx = int(np.searchsorted(beat_times, elapsed, side="right")) - 1

        if idx != active_beat:
            active_beat = idx
            for c in circles:
                c.set_color(COLOR_INACTIVE)
            if idx >= 0:
                pos = int(beat_positions[idx]) if beat_positions is not None else (idx % n) + 1
                dot = (pos - 1) % n
                circles[dot].set_color(COLOR_DOWNBEAT if pos == 1 else COLOR_BEAT)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)

    for c in circles:
        c.set_color(COLOR_INACTIVE)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.ioff()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Audio playback with click track
# ---------------------------------------------------------------------------

def play_with_clicks(audio_path: str, beat_times, beat_positions,
                     click_volume: float = 0.3):
    import librosa
    import sounddevice as sd

    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    if beat_times is not None and len(beat_times) > 0:
        click = librosa.clicks(times=beat_times, sr=sr, length=len(audio),
                               click_freq=1000, click_duration=0.02)
        mixed = audio + click_volume * click
        mixed = mixed / max(np.abs(mixed).max(), 1.0)
    else:
        mixed = audio
        print("No beat annotations found — playing audio without clicks.")

    print(f"Playing '{Path(audio_path).name}'  (Ctrl-C to stop)")

    sd.play(mixed, samplerate=sr)
    run_metronome(beat_times, beat_positions, len(mixed) / sr)
    sd.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect a dataset track.")
    parser.add_argument("--dataset", required=True, help="mirdata dataset name (e.g. ballroom)")
    parser.add_argument("--track", default=None, help="Track ID (random if omitted)")
    parser.add_argument("--visual", action="store_true", help="Show waveform + beat plot")
    parser.add_argument("--audio",  action="store_true", help="Play audio with beat clicks and metronome")
    parser.add_argument("--click-volume", type=float, default=0.3, metavar="VOL",
                        help="Click track volume relative to audio (default: 0.3)")
    args = parser.parse_args()

    if not args.visual and not args.audio:
        args.visual = True
        args.audio = True

    track_id = pick_track(args.dataset, args.track)
    print(f"[inspect] dataset={args.dataset}  track={track_id}")

    audio_path, tempo, beat_times, beat_positions = load_track(args.dataset, track_id)
    print(f"[inspect] tempo={tempo}  beats={len(beat_times) if beat_times is not None else 0}")

    if args.visual:
        plot_track(audio_path, tempo, beat_times, beat_positions)

    if args.audio:
        play_with_clicks(audio_path, beat_times, beat_positions,
                         click_volume=args.click_volume)


if __name__ == "__main__":
    main()
