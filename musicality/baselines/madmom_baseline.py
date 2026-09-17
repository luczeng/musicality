"""madmom's RNN + DBN beat/downbeat tracker, wrapped as a :class:`Baseline`.

This is the reference system the beat-tracking literature has been measured
against for a decade, and the one ``plans/08`` §1.2 compares our parameter
count to: a small bidirectional RNN ensemble producing beat and downbeat
activations at 100 fps, decoded by a dynamic Bayesian network over a
bar-pointer state space. Two things about it are the actual point of running it
here — the DBN decodes beats and bar positions *jointly* under an explicit
tempo/meter model, where we peak-pick and then label; and its context is the
whole track, where ours is 17.8 seconds.

Not installed by default. See :mod:`musicality.baselines` for why, and note
that madmom's model files are CC BY-NC-SA — non-commercial use only, unlike
madmom's own BSD-licensed code.
"""

import numpy as np

from musicality.baselines.base import Baseline

_INSTALL_HINT = (
    "madmom is not installed. The PyPI release (0.16.1) does not import on "
    "Python 3.10+; install the maintained git version instead:\n\n"
    "    uv sync --extra baselines\n\n"
    "or, standalone:\n\n"
    "    uv pip install setuptools cython numpy\n"
    "    uv pip install --no-build-isolation 'madmom @ git+https://github.com/CPJKU/madmom'"
)

# madmom's activation functions run at 100 frames per second and its DBNs are
# parameterised in those units, so this is not a knob — it is the rate the
# pretrained models were trained at.
FPS = 100


class MadmomBaseline(Baseline):
    """madmom's pretrained tracker.

    :param variant: Which system to run.

        - ``"downbeat"`` (default) — ``RNNDownBeatProcessor`` +
          ``DBNDownBeatTrackingProcessor``. Estimates beats *and* bar
          positions, so it is the only variant that can be scored on
          ``position_acc``/``f_one``.
        - ``"beat"`` — ``RNNBeatProcessor`` + ``DBNBeatTrackingProcessor``.
          Beats only. Worth running because the beat F-measures quoted in the
          literature for "madmom" are usually this system, not the downbeat
          one, and the two do not agree on beats.

    :param beats_per_bar: Bar lengths the downbeat DBN is allowed to choose
        between. madmom's own default is ``(3, 4)`` and changing it is a real
        intervention: pinning it to ``(4,)`` hands the tracker the meter, which
        our model never gets, so leave it alone for a fair comparison.
    :param min_bpm: Lower tempo bound for the DBN.
    :param max_bpm: Upper tempo bound. madmom's default of 215 is a genuine
        constraint on jtd, whose median tempo is around 185 BPM — a
        double-time reading is outside the state space and cannot be chosen,
        which flatters ``cmlt`` relative to a tracker without that prior.
    :param transition_lambda: Tempo-change penalty in the DBN's transition
        model. Higher is stiffer.
    :raises ImportError: If madmom is not installed, with install instructions.
    """

    name = "madmom"

    def __init__(
        self,
        variant: str = "downbeat",
        beats_per_bar: tuple[int, ...] = (3, 4),
        min_bpm: float = 55.0,
        max_bpm: float = 215.0,
        transition_lambda: float = 100.0,
    ):
        if variant not in ("downbeat", "beat"):
            raise ValueError(
                f"Unknown madmom variant {variant!r} — expected 'downbeat' or 'beat'"
            )

        try:
            from madmom.features.beats import (
                DBNBeatTrackingProcessor,
                RNNBeatProcessor,
            )
            from madmom.features.downbeats import (
                DBNDownBeatTrackingProcessor,
                RNNDownBeatProcessor,
            )
        except ImportError as error:
            raise ImportError(_INSTALL_HINT) from error

        self.variant = variant
        self.beats_per_bar = tuple(beats_per_bar)
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.transition_lambda = transition_lambda

        # Both processors load an 8-model RNN ensemble at construction time, so
        # they are built once and reused across every track.
        if variant == "downbeat":
            self.activations = RNNDownBeatProcessor()
            self.decoder = DBNDownBeatTrackingProcessor(
                beats_per_bar=list(self.beats_per_bar),
                min_bpm=min_bpm,
                max_bpm=max_bpm,
                transition_lambda=transition_lambda,
                fps=FPS,
            )
        else:
            self.activations = RNNBeatProcessor()
            self.decoder = DBNBeatTrackingProcessor(
                min_bpm=min_bpm,
                max_bpm=max_bpm,
                transition_lambda=transition_lambda,
                fps=FPS,
            )

    @property
    def config(self) -> dict:
        config = {
            "variant": self.variant,
            "min_bpm": self.min_bpm,
            "max_bpm": self.max_bpm,
            "transition_lambda": self.transition_lambda,
        }

        if self.variant == "downbeat":
            config["beats_per_bar"] = list(self.beats_per_bar)

        return config

    def predict(self, audio_path: str) -> tuple[np.ndarray, np.ndarray | None]:
        """Run the RNN over the file and decode it with the DBN.

        madmom loads the audio itself, at its own 44.1 kHz; non-WAV files go
        through ffmpeg, which must be on ``PATH``.

        :returns: ``(beats, downbeats)``. ``downbeats`` is ``None`` for the
            beat-only variant, and the subset of *beats* the DBN numbered ``1``
            otherwise.
        """

        activations = self.activations(str(audio_path))
        decoded = self.decoder(activations)

        if self.variant == "beat":
            return np.asarray(decoded, dtype=float), None

        # (n_beats, 2): time, then the beat's 1-based position in its bar.
        decoded = np.atleast_2d(np.asarray(decoded, dtype=float))
        if decoded.size == 0:
            return np.zeros(0), np.zeros(0)

        beats = decoded[:, 0]
        downbeats = decoded[decoded[:, 1] == 1, 0]

        return beats, downbeats
