"""Tempo estimation backbones: custom CNN, torchaudio wav2vec2, and HuggingFace models."""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModel


class TempoNet(nn.Module):
    """Lightweight CNN that maps a raw waveform to a single BPM value.

    Internally applies a log-mel spectrogram transform before the CNN.

    Input:  (B, 1, T)
    Output: (B,)

    :param n_mels: Number of mel filterbanks.
    :param sample_rate: Audio sample rate, used to build the mel transform.
    :param hop_length: Hop length for the mel transform.
    :param dropout: Dropout probability in the regression head.
    """

    def __init__(self, n_mels: int = 128, sample_rate: int = 22050, hop_length: int = 512, dropout: float = 0.3):

        super().__init__()

        self.mel = nn.Sequential(
            T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=2048, hop_length=hop_length),
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
            nn.Linear(128, 1),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:

        mel = self.mel(wav)  # (B, 1, n_mels, T)

        # Per-sample normalisation — stabilises inputs across varying loudness
        mean = mel.mean(dim=(-2, -1), keepdim=True)
        std = mel.std(dim=(-2, -1), keepdim=True)
        mel = (mel - mean) / (std + 1e-6)

        return self.head(self.encoder(mel)).squeeze(1)


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
            T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=2048, hop_length=hop_length),
            T.AmplitudeToDB(),
        )

        self.input_proj = nn.Conv1d(n_mels, channels, kernel_size=1)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3,
                          padding=2 ** i, dilation=2 ** i),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            )
            for i in range(n_layers)
        ])

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

        x = self.input_proj(x)        # (B, channels, T)

        for layer in self.layers:
            x = x + layer(x)       # dilated residual

        x = x.mean(dim=-1)         # (B, channels) — global average pool over time

        return self.head(x).squeeze(1)


class TorchAudioTempoNet(nn.Module):
    """TorchAudio wav2vec2 encoder with a tempo regression head.

    Uses a ``torchaudio.pipelines`` bundle as backbone (e.g. ``WAV2VEC2_BASE``).
    Expects raw waveforms at the pipeline's native sample rate (typically 16 kHz).
    The backbone is frozen by default.

    Input:  (B, 1, T)
    Output: (B,)

    :param pipeline: ``torchaudio.pipelines`` attribute name (e.g. ``"WAV2VEC2_BASE"``).
    :param dropout: Dropout probability in the regression head.
    :param freeze_backbone: Whether to freeze the backbone weights.
    """

    def __init__(
        self,
        pipeline: str = "WAV2VEC2_BASE",
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        bundle = getattr(torchaudio.pipelines, pipeline)
        self.backbone = bundle.get_model()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Infer hidden size via a dummy forward pass
        with torch.no_grad():
            features, _ = self.backbone(torch.zeros(1, 16000))
            hidden_size = features.shape[-1]

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:

        wav = wav.squeeze(1)  # (B, 1, T) -> (B, T)

        features, _ = self.backbone(wav)  # (B, T, hidden_size)

        hidden = features.mean(dim=1)  # (B, hidden_size)

        return self.head(hidden).squeeze(1)


class WaveformTempoNet(nn.Module):
    """Any HuggingFace waveform encoder with a tempo regression head.

    Loads any model via ``AutoModel.from_pretrained``, mean-pools its last
    hidden state, and feeds it through a small regression head.
    Expects raw waveforms at the sample rate required by the chosen backbone
    (typically 16 kHz). The backbone is frozen by default.

    Input:  (B, 1, T)
    Output: (B,)

    :param model_name: HuggingFace model identifier.
    :param dropout: Dropout probability in the regression head.
    :param freeze_backbone: Whether to freeze the backbone weights.
    """

    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:

        wav = wav.squeeze(1)  # (B, 1, T) -> (B, T)

        hidden = self.backbone(wav).last_hidden_state.mean(dim=1)  # (B, hidden_size)

        return self.head(hidden).squeeze(1)
