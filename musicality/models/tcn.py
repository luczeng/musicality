import torch.nn as nn
import torch
import torchaudio.transforms as T


class TCNTempoNet(nn.Module):
    """Dilated TCN for tempo regression (Davies & Böck, 2019), or per-frame beat-phase detection.

    Applies a log-mel transform, projects to the TCN channel width, then runs a
    stack of dilated 1D residual convolutions with exponentially growing dilation
    (1, 2, 4, …, 2^(n_layers-1)).

    Two output modes, controlled by ``frame_level``:

    - ``frame_level=False`` (default): globally pools over time, then a small
      FC head produces scalar/bin regression or classification logits.
      Input: (B, 1, T) → Output: (B,) or (B, n_outputs).
    - ``frame_level=True``: skips the pool; a 1x1 conv head produces
      per-frame logits instead (e.g. beat/one/last for beat-phase detection).
      Input: (B, 1, T) → Output: (B, n_outputs, T') or (B, T') if n_outputs == 1,
      where T' is the mel transform's frame count. Sigmoid is *not* applied —
      pair with ``BCEWithLogitsLoss`` downstream, matching the classification
      mode's convention of returning raw logits.

    Receptive field ≈ kernel_size × (2^n_layers − 1) frames — the same trunk is
    shared between both modes, so this is unaffected by ``frame_level``.

    :param n_mels: Number of mel filterbanks.
    :param sample_rate: Audio sample rate used to build the mel transform.
    :param hop_length: Hop length for the mel transform. Controls temporal resolution
        (smaller = more frames per second). Defaults to 512 (≈43 fps at 22050 Hz).
    :param channels: Channel width for the TCN.
    :param n_layers: Number of dilated layers. Keep receptive field
        (3 × (2^n_layers − 1) frames) within the input sequence length.
    :param dropout: Dropout probability applied right before the final head layer —
        the pooled regression head's last ``Linear``, or the frame head's 1x1 conv.
    :param n_outputs: Output dimension. In pooled mode, ``1`` for scalar regression,
        > 1 for classification over tempo bins. In frame-level mode, the number of
        per-frame target channels (e.g. 3 for beat/one/last).
    :param frame_level: If ``True``, produce per-frame outputs instead of pooling
        over time.
    """

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 22050,
        hop_length: int = 512,
        channels: int = 32,
        n_layers: int = 8,
        dropout: float = 0.3,
        n_outputs: int = 1,
        frame_level: bool = False,
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.frame_level = frame_level

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

        if frame_level:
            self.frame_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Conv1d(channels, n_outputs, kernel_size=1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(channels, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_outputs),
            )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:

        x = self.mel(wav).squeeze(1)  # (B, 1, n_mels, T) → (B, n_mels, T')

        # Per-sample normalisation — stabilises inputs across varying loudness
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        x = (x - mean) / (std + 1e-6)

        x = self.input_proj(x)  # (B, channels, T')

        for layer in self.layers:
            x = x + layer(x)  # dilated residual

        if self.frame_level:
            out = self.frame_head(x)  # (B, n_outputs, T')
            return out.squeeze(1) if self.n_outputs == 1 else out

        x = x.mean(dim=-1)  # (B, channels) — global average pool over time

        out = self.head(x)  # (B, n_outputs)

        return out.squeeze(-1) if self.n_outputs == 1 else out
