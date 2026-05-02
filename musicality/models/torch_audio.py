import torch.nn as nn
import torchaudio
import torch


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
