import torch.nn as nn
import torch
import torchaudio.transforms as T


class TCNTempoNet(nn.Module):
    """Dilated TCN for tempo regression (Davies & Böck, 2019).

    Applies a log-mel transform, projects to the TCN channel width, then runs a
    stack of dilated 1D residual convolutions with exponentially growing dilation
    (1, 2, 4, …, 2^(n_layers-1)). Globally pools over time before the regression head.

    Receptive field ≈ kernel_size × (2^n_layers − 1) frames.

    Input:  (B, 1, T)
    Output: (B,)

    :param n_mels: Number of mel filterbanks.
    :param sample_rate: Audio sample rate used to build the mel transform.
    :param hop_length: Hop length for the mel transform. Controls temporal resolution
        (smaller = more frames per second). Defaults to 512 (≈43 fps at 22050 Hz).
    :param channels: Channel width for the TCN.
    :param n_layers: Number of dilated layers. Keep receptive field
        (3 × (2^n_layers − 1) frames) within the input sequence length.
    :param dropout: Dropout probability in the regression head.
    """

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 22050,
        hop_length: int = 512,
        channels: int = 32,
        n_layers: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.mel = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=2048,
                hop_length=hop_length,
            ),
            T.AmplitudeToDB(),
        )

        self.input_proj = nn.Conv1d(n_mels, channels, kernel_size=1)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels, channels, kernel_size=3, padding=2**i, dilation=2**i
                    ),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                )
                for i in range(n_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:

        x = self.mel(wav).squeeze(1)  # (B, 1, n_mels, T) → (B, n_mels, T)

        # Per-sample normalisation — stabilises inputs across varying loudness
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        x = (x - mean) / (std + 1e-6)

        x = self.input_proj(x)  # (B, channels, T)

        for layer in self.layers:
            x = x + layer(x)  # dilated residual

        x = x.mean(dim=-1)  # (B, channels) — global average pool over time

        return self.head(x).squeeze(1)
