import torch
import torch.nn as nn
from transformers import AutoModel

"""Tempo estimation backbones: custom CNN, torchaudio wav2vec2, and HuggingFace models."""


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
