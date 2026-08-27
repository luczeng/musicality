import torch.nn as nn
import torch
import torchaudio.transforms as T


class PositionalEncoding(nn.Module):
    """Additive sinusoidal positional encoding (Vaswani et al., 2017).

    Self-attention has no built-in notion of frame order (unlike convolution
    or recurrence), so this injects one: each position gets a fixed
    sin/cos pattern that varies by position and by channel pair, added
    directly to the input.

    :param channels: Channel width. Must be even — sin/cos are paired per
        two channels.
    """

    def __init__(self, channels: int):
        super().__init__()

        if channels % 2 == 0:
            self.num_channels = channels
        else:
            raise ValueError("channels must be even")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: ``(B, T, C)``
        :returns: ``x`` plus the positional encoding, same shape.
        """

        B, T, C = x.shape
        vec_channels = torch.arange(0, C, 1)
        vec_times = torch.arange(0, T, 1)
        channels, times = torch.meshgrid(vec_channels, vec_times, indexing="xy")

        denom = torch.pow(10000, 2 * (channels // 2) / self.num_channels)
        grid = times / denom

        grid[:, ::2] = torch.sin(grid[:, ::2])
        grid[:, 1::2] = torch.cos(grid[:, 1::2])
        grid = grid.unsqueeze(0).expand(B, -1, -1)

        return x + grid


class SelfAttentionBlock(nn.Module):
    """One transformer-encoder-style block: self-attention sublayer, then a
    feedforward sublayer, each wrapped in its own residual connection and
    LayerNorm. Lets every frame's representation draw on every other frame
    in the sequence, unlike the TCN trunk's fixed dilated-conv receptive
    field (see docs/beat_phase_context_ideas.md).

    :param channels: Channel width (attention embedding dim).
    :param n_heads: Number of attention heads.
    """

    def __init__(self, channels: int, n_heads: int):
        super().__init__()

        self.num_channels = channels
        self.n_heads = n_heads

        self.mha = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.l1 = nn.Linear(channels, 4 * channels)
        self.l2 = nn.Linear(4 * channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.nl = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x -> multihead attention + res + layernorm -> h
        h -> MLP -> residual -> layernorm -> output

        :param x: ``(B, T, C)``
        """

        h, _ = self.mha(x, x, x)
        h = h + x
        h = self.norm1(h)

        h = self.l2(self.nl(self.l1(h))) + h
        h = self.norm2(h)

        return h


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
    :param dropout: Dropout probability applied right before each head's final
        ``Conv1d``/``Linear`` — the pooled regression head's last ``Linear``,
        the frame head's 1x1 conv, or (when ``use_self_attention=True``) the
        ``beat_head``/``phase_head`` 1x1 convs.
    :param n_outputs: Output dimension. In pooled mode, ``1`` for scalar regression,
        > 1 for classification over tempo bins. In frame-level mode, the number of
        per-frame target channels (e.g. 3 for beat/one/last).
    :param frame_level: If ``True``, produce per-frame outputs instead of pooling
        over time.
    :param use_self_attention: Frame-level mode only. If ``True``, splits the
        frame head in two: ``beat_head`` reads straight off the TCN trunk
        (unchanged, already accurate), while ``phase_head`` routes the
        ``one``/``last`` channels through a positional encoding + a stack of
        :class:`SelfAttentionBlock`, giving them context beyond the trunk's
        fixed dilated-conv receptive field. Output channel order is always
        ``(beat, one, last)``, matching :func:`musicality.losses.beat_phase_loss`.
        See docs/beat_phase_context_ideas.md.
    :param n_attn_layers: Number of stacked :class:`SelfAttentionBlock` in
        ``phase_head``. Only used when ``use_self_attention=True``.
    :param n_attn_heads: Attention heads per :class:`SelfAttentionBlock`. Only
        used when ``use_self_attention=True``.
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
        use_self_attention: bool = False,
        n_attn_layers: int = 1,
        n_attn_heads: int = 4,
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.frame_level = frame_level
        self.use_self_attention = use_self_attention

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
            if use_self_attention:
                self.phase_head = nn.ModuleDict(
                    {
                        "pos_enc": PositionalEncoding(channels),
                        "attn_blocks": nn.ModuleList(
                            SelfAttentionBlock(channels, n_attn_heads)
                            for _ in range(n_attn_layers)
                        ),
                        "out": nn.Sequential(
                            nn.Dropout(dropout),
                            nn.Conv1d(channels, 2, kernel_size=1),
                        ),
                    }
                )

                self.beat_head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Conv1d(channels, 1, kernel_size=1),
                )

            else:
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
            if self.use_self_attention:
                h = x.transpose(1, 2)  # (B, channels, T') → (B, T', channels)
                h = self.phase_head["pos_enc"](h)
                for block in self.phase_head["attn_blocks"]:
                    h = block(h)
                h = h.transpose(1, 2)  # (B, T', channels) → (B, channels, T')

                out_phase = self.phase_head["out"](h)  # (B, 2, T') — one, last
                out_beat = self.beat_head(
                    x
                )  # (B, 1, T') — reads straight off the trunk
                return torch.cat(
                    (out_beat, out_phase), dim=1
                )  # (B, 3, T') — beat, one, last
            out = self.frame_head(x)  # (B, n_outputs, T')
            return out.squeeze(1) if self.n_outputs == 1 else out

        x = x.mean(dim=-1)  # (B, channels) — global average pool over time

        out = self.head(x)  # (B, n_outputs)

        return out.squeeze(-1) if self.n_outputs == 1 else out
