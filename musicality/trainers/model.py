"""Tempo estimation backbones: custom CNN, wav2vec2, and BEaT."""

import torch
import torch.nn as nn
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2Model


class TempoNet(nn.Module):
    """Lightweight CNN that maps a raw waveform to a single BPM value.

    Internally applies a log-mel spectrogram transform before the CNN.

    Input:  (B, 1, T)
    Output: (B,)

    :param n_mels: Number of mel filterbanks.
    :param sample_rate: Audio sample rate, used to build the mel transform.
    :param dropout: Dropout probability in the regression head.
    """

    def __init__(self, n_mels: int = 128, sample_rate: int = 22050, dropout: float = 0.3):

        super().__init__()

        self.mel = nn.Sequential(
            T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=2048),
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

        mel = self.mel(wav)  # (B, n_mels, T)
        mel = mel.unsqueeze(1)  # (B, 1, n_mels, T)

        return self.head(self.encoder(mel)).squeeze(1)


class Wav2Vec2TempoNet(nn.Module):
    """wav2vec 2.0 encoder with a tempo regression head.

    Expects raw waveforms at 16 kHz. The backbone is frozen by default.

    Input:  (B, 1, T)
    Output: (B,)

    :param model_name: HuggingFace model identifier.
    :param dropout: Dropout probability in the regression head.
    :param freeze_backbone: Whether to freeze the backbone weights.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = Wav2Vec2Model.from_pretrained(model_name)

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


class BEaTTempoNet(nn.Module):
    """BEaT encoder with a tempo regression head.

    Expects raw waveforms at 16 kHz. The backbone is frozen by default.

    Input:  (B, 1, T)
    Output: (B,)

    :param model_name: HuggingFace model identifier.
    :param dropout: Dropout probability in the regression head.
    :param freeze_backbone: Whether to freeze the backbone weights.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BEaT-iter3-AS2M",
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
