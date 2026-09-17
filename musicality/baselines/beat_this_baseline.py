"""Beat This! (Foscarin, Schlüter & Widmer, ISMIR 2024), wrapped as a
:class:`Baseline`.

The current state of the art, and the system ``plans/08`` §3.3 and §4.1 argue
we should be learning from rather than competing with. Three of its choices are
the ones worth measuring against ours: a transformer over a log-mel
spectrogram at 50 fps with no recurrent or convolutional temporal model; a
shift-tolerant loss that stops punishing a prediction one frame early; and
**no DBN at all** by default — peak-picking the frame probabilities is enough,
which is the direct counterexample to "our decoder is the bottleneck".

Not installed by default; see :mod:`musicality.baselines`.

Checkpoints download themselves on first use (about 78 MB for a ``final*``
model) and are cached by ``torch.hub``, so the first call needs network access.
"""

import numpy as np

from musicality.baselines.base import Baseline

_INSTALL_HINT = (
    "beat_this is not installed. Install it with:\n\n"
    "    uv sync --extra baselines\n\n"
    "or, standalone:\n\n"
    "    uv pip install 'beat-this @ git+https://github.com/CPJKU/beat_this'"
)


class BeatThisBaseline(Baseline):
    """Beat This!'s pretrained tracker.

    :param checkpoint: Upstream checkpoint name (or a local path/URL).
        ``final0``/``final1``/``final2`` are the main models, trained on every
        corpus the authors used **except GTZAN**; ``small0..2`` are the ~8 MB
        variant. Because most of our validation split is in their training
        data, prefer ``single_final0`` or a ``fold*`` checkpoint when a
        non-GTZAN corpus is the one being argued about — those hold out a
        documented validation split instead of nothing. See
        :data:`~musicality.baselines.base.CORPUS_EXPOSURE`.
    :param dbn: Replace the default peak-picking postprocessor with madmom's
        DBN. Requires madmom, and is the closest thing to an apples-to-apples
        decoder comparison available: the same frame probabilities, decoded two
        ways.
    :param device: Torch device for inference.
    :param float16: Run the transformer in half precision. Faster on GPU,
        pointless on CPU.
    :raises ImportError: If beat_this is not installed, with install instructions.
    """

    name = "beat_this"

    def __init__(
        self,
        checkpoint: str = "final0",
        dbn: bool = False,
        device: str = "cpu",
        float16: bool = False,
    ):
        try:
            from beat_this.inference import File2Beats
        except ImportError as error:
            raise ImportError(_INSTALL_HINT) from error

        self.checkpoint = checkpoint
        self.dbn = dbn
        self.device = device
        self.float16 = float16

        self.tracker = File2Beats(
            checkpoint_path=checkpoint,
            device=device,
            float16=float16,
            dbn=dbn,
        )

    @property
    def config(self) -> dict:
        # `device` and `float16` are deliberately absent: they change how the
        # same weights are executed, not what they predict, so a cache built on
        # GPU is valid for a CPU run.
        return {"checkpoint": self.checkpoint, "dbn": self.dbn}

    def predict(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Track one file.

        Beat This! loads and resamples the audio itself, to its own 22.05 kHz.
        Its postprocessor snaps every downbeat onto the nearest beat, so the
        returned downbeats are always an exact subset of the beats.

        :returns: ``(beats, downbeats)``, times in seconds.
        """

        beats, downbeats = self.tracker(str(audio_path))

        return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)
