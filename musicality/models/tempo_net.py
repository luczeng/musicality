import torch
import torch.nn as nn
import torchaudio.transforms as T


class TempoNet(nn.Module):
    """Lightweight CNN that maps a raw waveform to a single BPM value.

    Internally applies a log-mel spectrogram transform before the CNN.

    Input:  (B, 1, T)
    Output: (B,)

    :param n_mels: Number of mel filterbanks.
    :param sample_rate: Audio sample rate, used to build the mel transform.
    :param hop_length: Hop length for the mel transform.
    :param dropout: Dropout probability in the regression head.
    :param n_outputs: Output dimension. ``1`` for scalar regression; > 1 for
        classification over tempo bins (returns logits without softmax).
    """

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 22050,
        hop_length: int = 512,
        dropout: float = 0.3,
        n_outputs: int = 1,
    ):

        super().__init__()
        self.n_outputs = n_outputs

        self.mel = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=2048,
                hop_length=hop_length,
            ),
            T.AmplitudeToDB(),
        )

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_outputs),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:

        mel = self.mel(wav)  # (B, 1, n_mels, T)

        # Per-sample normalisation — stabilises inputs across varying loudness
        mean = mel.mean(dim=(-2, -1), keepdim=True)
        std = mel.std(dim=(-2, -1), keepdim=True)
        mel = (mel - mean) / (std + 1e-6)

        out = self.head(self.encoder(mel))  # (B, n_outputs)
        return out.squeeze(-1) if self.n_outputs == 1 else out
