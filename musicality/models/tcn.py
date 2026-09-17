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

        # Cache keyed by (T, device, dtype) — the encoding is a deterministic
        # function of those alone, so it's cheap to reuse across forward calls
        # instead of rebuilding the meshgrid/pow/sin/cos every time. A plain
        # attribute (not a registered buffer) on purpose: it's derived, not
        # learned, so it shouldn't appear in state_dict/checkpoints, and this
        # way a stale cache from before a `.to(device)` call is naturally
        # detected and rebuilt below rather than silently going stale.
        self._cached_encoding: torch.Tensor | None = None

    def _build_encoding(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        vec_channels = torch.arange(0, self.num_channels, device=device)
        vec_times = torch.arange(0, T, device=device)
        channels, times = torch.meshgrid(vec_channels, vec_times, indexing="xy")

        denom = torch.pow(10000, 2 * (channels // 2) / self.num_channels)
        grid = times / denom

        grid[:, ::2] = torch.sin(grid[:, ::2])
        grid[:, 1::2] = torch.cos(grid[:, 1::2])

        return grid.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: ``(B, T, C)``
        :returns: ``x`` plus the positional encoding, same shape.
        """

        _, T, _ = x.shape
        cache = self._cached_encoding
        if (
            cache is None
            or cache.shape[0] != T
            or cache.device != x.device
            or cache.dtype != x.dtype
        ):
            cache = self._build_encoding(T, x.device, x.dtype)
            self._cached_encoding = cache

        return x + cache.unsqueeze(0)


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

        # No key_padding_mask: assumes every frame in x is real audio, no
        # padding — true today since BeatDataset always crops/pads to a fixed
        # duration before batching. If variable-length batches or chunked
        # inference (see docs/beat_phase_context_ideas.md) are added later,
        # a mask needs to be threaded through here, or attention will
        # silently attend into padding frames.
        h, _ = self.mha(x, x, x)
        h = h + x
        h = self.norm1(h)

        h = self.l2(self.nl(self.l1(h))) + h
        h = self.norm2(h)

        return h


class Conv2dStem(nn.Module):
    """Local time-frequency processing in front of the 1D trunk.

    ``TCNTempoNet``'s first operation used to be
    ``Conv1d(n_mels, channels, kernel_size=1)``: at every frame, a fixed linear
    mixture of all mel bands, after which the band axis is gone and every
    remaining layer convolves over time only. Nothing in the model ever saw a
    time-frequency neighbourhood.

    That is the wrong first operation for beat tracking, for two reasons
    (``plans/08_rethinking_the_approach.md`` §2.1 has the long version):

    - **A 1x1 mixer is frequency-absolute; onsets are frequency-relative.** It
      learns one weight per band, applied identically at every frame — so it can
      learn "these bands matter", but not "energy rose in *whichever* band it
      rose in". A kick drum and a walking bass note are the same event shape at
      different absolute frequencies, and a mixer must spend separate output
      channels on each register to detect the same thing twice. A ``Conv2d``
      shares one kernel across frequency and gets that equivariance for free,
      which is the reason to put audio on a log-frequency axis at all.
    - **Mixing before differencing lets onsets cancel.** Onset strength is a
      difference across time *within* a band. With the bands summed first, a
      band rising and another falling by the same weighted amount produces a
      flat mixture, and the event is gone before any layer could see it. That is
      exactly a harmonic change with no percussive attack — the dominant
      downbeat cue wherever no drum marks the bar.

    Every published tracker does local spectro-temporal processing first
    (madmom's TCN opens with 3x3 convolutions and frequency max-pooling; Beat
    This! uses frequency-wise partial attention), and both reach their numbers
    with a weak decoder or none at all, which is what points at the front end.

    Shape, with the defaults and ``n_mels=128``::

        (B, 128, T)                 log-mel, as the trunk used to receive it
        (B, 1, 128, T)              band axis promoted to a spatial axis
        (B, 16, 128, T)             Conv2d 3x3 + BN + GELU
        (B, 16,  42, T)             MaxPool2d((3, 1)) — frequency only
        (B, 16,  42, T)             Conv2d 3x3 + BN + GELU
        (B, 16,  14, T)             MaxPool2d((3, 1))
        (B, 16,  14, T)             Conv2d 3x3 + BN + GELU
        (B, 224, T)                 frequency folded into channels

    **Time is never pooled.** The frame rate is the output resolution, and the
    trunk's dilations are what buy context — pooling time here would spend
    precision the 70 ms evaluation tolerance cannot afford. The 3x3 kernels do
    widen the receptive field by ``2 * n_layers`` frames, which is negligible
    beside the trunk's ``1 + 2 * sum(dilations)``.

    :param n_mels: Number of input mel bands.
    :param channels: Feature maps per 2D layer. 16 is madmom-scale; the stem is
        meant to be cheap next to the trunk, not to hold capacity.
    :param n_layers: Number of ``Conv2d`` blocks. Frequency is pooled after
        every block *except the last*, so ``n_layers=3`` pools twice.
    :param freq_pool: Frequency pooling factor per pool.
    :raises ValueError: If *n_layers* is below 1, or if the pooling schedule
        would leave fewer than one frequency bin.
    """

    def __init__(
        self,
        n_mels: int,
        channels: int = 16,
        n_layers: int = 3,
        freq_pool: int = 3,
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if freq_pool < 1:
            raise ValueError(f"freq_pool must be >= 1, got {freq_pool}")

        blocks = []
        in_channels = 1
        n_freq = n_mels

        for layer in range(n_layers):
            blocks += [
                nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            ]
            in_channels = channels

            # No pool after the last block: the frequency axis is about to be
            # folded into channels anyway, and one more pool would throw away
            # resolution the trunk could have used.
            if layer < n_layers - 1:
                blocks.append(
                    nn.MaxPool2d(kernel_size=(freq_pool, 1), stride=(freq_pool, 1))
                )
                n_freq //= freq_pool

                if n_freq < 1:
                    raise ValueError(
                        f"Frequency axis collapses to {n_freq} bins: n_mels={n_mels} "
                        f"cannot survive {n_layers - 1} pool(s) of {freq_pool}. "
                        "Lower n_layers/freq_pool or raise n_mels."
                    )

        self.blocks = nn.Sequential(*blocks)

        #: Frequency bins surviving the pooling schedule.
        self.n_freq = n_freq

        #: Channel count the trunk's ``input_proj`` must expect.
        self.out_channels = channels * n_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Normalised log-mel, shape ``(B, n_mels, T)``.
        :returns: ``(B, out_channels, T)`` — *T* unchanged.
        """

        x = self.blocks(x.unsqueeze(1))  # (B, 1, n_mels, T) → (B, C, n_freq, T)

        batch, channels, n_freq, frames = x.shape

        # Fold frequency into channels rather than pooling it away: the trunk is
        # 1D, and which band a feature fired in is information the bar-position
        # head has no other way to recover. `reshape`, not `view` — MaxPool2d's
        # output is not guaranteed contiguous.
        return x.reshape(batch, channels * n_freq, frames)


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

    Receptive field is ``1 + (kernel_size - 1) * sum(dilations)`` frames — with
    the default schedule ``1, 2, ..., 2^(n_layers-1)`` that is
    ``2^(n_layers+1) - 1``, so 511 frames ≈ 11.9 s at ``n_layers=8``,
    ``hop_length=512``. (This docstring used to quote
    ``kernel_size × (2^n_layers − 1)``, a loose upper bound ~1.5x the truth; see
    ``docs/beat_phase_context_ideas.md`` and ``plans/08`` §1.2/§3.1.) The same
    trunk is shared between both modes, so this is unaffected by
    ``frame_level``. A ``conv2d_stem`` adds ``2 × stem_layers`` frames to that,
    which is noise beside it.

    ``input_norm`` is the other. See that parameter, and
    :mod:`musicality.input_stats` for the measurement behind the default.

    **The receptive field is only real if the input is at least that long.**
    Every trunk conv uses ``padding=dilation``, so a layer whose dilation exceeds
    the input length has both off-centre taps in zero padding at every frame and
    collapses to a 1x1 conv. On a 16 s clip at ``hop_length=512`` (689 frames)
    that is any layer past the ninth. See ``plans/08`` §3.1.

    ``conv2d_stem`` is one of two structural options here. Off, the first
    operation is a 1x1 mix over mel bands and no layer ever sees a
    time-frequency neighbourhood; on, a small :class:`Conv2dStem` runs first.
    See that class for why it exists.

    :param n_mels: Number of mel filterbanks.
    :param sample_rate: Audio sample rate used to build the mel transform.
    :param hop_length: Hop length for the mel transform. Controls temporal resolution
        (smaller = more frames per second). Defaults to 512 (≈43 fps at 22050 Hz).
    :param channels: Channel width for the TCN.
    :param n_layers: Number of dilated layers; dilation doubles per layer. Keep
        the *dilation* of the deepest layer (``2^(n_layers-1)`` frames) below the
        input sequence length, or that layer only ever convolves padding.
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
        remaining ``n_outputs - 1`` channels through a positional encoding +
        a stack of :class:`SelfAttentionBlock`, giving them context beyond the
        trunk's fixed dilated-conv receptive field. Output channel order is
        always ``beat`` first, then the phase channels — ``(beat, one, last)``
        for :func:`musicality.losses.beat_phase_loss`, or
        ``(beat, pos_1, ..., pos_G)`` for
        :func:`musicality.losses.beat_position_loss`.
        See docs/beat_phase_context_ideas.md.
    :param n_attn_layers: Number of stacked :class:`SelfAttentionBlock` in
        ``phase_head``. Only used when ``use_self_attention=True``.
    :param n_attn_heads: Attention heads per :class:`SelfAttentionBlock`. Only
        used when ``use_self_attention=True``.
    :param conv2d_stem: Run a :class:`Conv2dStem` between the mel and the trunk
        instead of projecting the raw bands. Defaults to ``False`` *here* so
        that checkpoints predating the stem reconstruct into identical parameter
        shapes from their own saved hyperparameters; the shipped frame-level
        configs (``configs/model/tcn_frames*.yaml``) turn it on. Measured cost
        at those configs' ``channels=32``: 32,805 parameters to 40,773 — the
        stem itself is 4.9 k, the rest is ``input_proj`` widening from 128 to
        224 inputs.
    :param stem_channels: Feature maps per stem layer. ``conv2d_stem`` only.
    :param stem_layers: ``Conv2d`` blocks in the stem; frequency is pooled after
        all but the last. ``conv2d_stem`` only.
    :param stem_freq_pool: Frequency pooling factor per stem pool.
        ``conv2d_stem`` only.
    :param input_norm: How the log-mel is normalised before the stem/trunk.

        - ``"global"`` (the code default, and what every checkpoint up to v6
          was trained with): one mean and one std over *both* axes of the input
          tensor. Cheap, but the statistics depend on how much audio is in the
          tensor, so a 16 s training crop and a whole-track inference pass
          normalise the same bars differently.
        - ``"fixed"`` (what the shipped frame-level configs use): one frozen
          mean and std *per mel band*, measured once over the training data and
          carried in the checkpoint as buffers. Removes the length dependence by
          construction.

        Under ``"fixed"`` the buffers start at 0/1 — an identity transform — and
        must be filled before training, by
        :func:`musicality.input_stats.fit_input_stats` or
        :meth:`set_input_stats`. Both training entry points do this
        automatically; :attr:`input_stats_fitted` reports the state.
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
        conv2d_stem: bool = False,
        stem_channels: int = 16,
        stem_layers: int = 3,
        stem_freq_pool: int = 3,
        input_norm: str = "global",
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.frame_level = frame_level
        self.use_self_attention = use_self_attention

        if input_norm not in ("global", "fixed"):
            raise ValueError(
                f"input_norm must be 'global' or 'fixed', got {input_norm!r}"
            )

        self.input_norm = input_norm

        self.mel = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=2048,
                hop_length=hop_length,
            ),
            T.AmplitudeToDB(),
        )

        # Registered unconditionally under `fixed` (not only once measured) so
        # that a checkpoint's state_dict and a freshly constructed model always
        # agree on their keys, and `load_state_dict` restores the statistics
        # along with the weights.
        if input_norm == "fixed":
            self.register_buffer("norm_mean", torch.zeros(n_mels))
            self.register_buffer("norm_std", torch.ones(n_mels))
            self.register_buffer("norm_fitted", torch.zeros((), dtype=torch.bool))

        self.stem = (
            Conv2dStem(
                n_mels,
                channels=stem_channels,
                n_layers=stem_layers,
                freq_pool=stem_freq_pool,
            )
            if conv2d_stem
            else None
        )

        # Unchanged when there is no stem, so a checkpoint trained without one
        # loads into exactly the same parameter shapes.
        proj_in = n_mels if self.stem is None else self.stem.out_channels

        self.input_proj = nn.Conv1d(proj_in, channels, kernel_size=1)

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
                            nn.Conv1d(channels, n_outputs - 1, kernel_size=1),
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

    @property
    def input_stats_fitted(self) -> bool:
        """Whether ``input_norm="fixed"`` statistics have been measured.

        Always ``False`` under ``input_norm="global"``, which needs none.
        """

        return self.input_norm == "fixed" and bool(self.norm_fitted)

    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install per-band normalisation statistics and mark them fitted.

        :param mean: Per-band mean, shape ``(n_mels,)``.
        :param std: Per-band standard deviation, shape ``(n_mels,)``. Must be
            strictly positive — :func:`musicality.input_stats.compute_band_stats`
            clamps it for exactly this reason.
        :raises RuntimeError: If ``input_norm`` is not ``"fixed"``.
        :raises ValueError: On a shape mismatch, or a non-positive ``std``.
        """

        if self.input_norm != "fixed":
            raise RuntimeError(
                f"set_input_stats() needs input_norm='fixed', got {self.input_norm!r}"
            )

        mean = torch.as_tensor(mean, dtype=self.norm_mean.dtype)
        std = torch.as_tensor(std, dtype=self.norm_std.dtype)

        if mean.shape != self.norm_mean.shape or std.shape != self.norm_std.shape:
            raise ValueError(
                f"expected statistics of shape {tuple(self.norm_mean.shape)}, got "
                f"mean {tuple(mean.shape)} and std {tuple(std.shape)}"
            )

        if not bool((std > 0).all()):
            raise ValueError("every per-band std must be > 0")

        self.norm_mean.copy_(mean.to(self.norm_mean.device))
        self.norm_std.copy_(std.to(self.norm_std.device))
        self.norm_fitted.fill_(True)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:

        x = self.mel(wav).squeeze(1)  # (B, 1, n_mels, T) → (B, n_mels, T')

        if self.input_norm == "fixed":
            # Frozen per-band statistics: the same audio normalises identically
            # whether it arrives as a 16 s crop or inside a whole track. See
            # musicality.input_stats for the measurement that motivates this.
            x = (x - self.norm_mean[None, :, None]) / self.norm_std[None, :, None]
        else:
            # Per-sample normalisation over both axes — stabilises inputs across
            # varying loudness, but the statistics depend on the length of the
            # input, so training (a crop) and inference (a whole track) disagree.
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True)
            x = (x - mean) / (std + 1e-6)

        if self.stem is not None:
            x = self.stem(x)  # (B, n_mels, T') → (B, stem.out_channels, T')

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

                out_phase = self.phase_head["out"](h)  # (B, n_outputs - 1, T')
                out_beat = self.beat_head(
                    x
                )  # (B, 1, T') — reads straight off the trunk
                return torch.cat(
                    (out_beat, out_phase), dim=1
                )  # (B, n_outputs, T') — beat first, then the phase channels
            out = self.frame_head(x)  # (B, n_outputs, T')
            return out.squeeze(1) if self.n_outputs == 1 else out

        x = x.mean(dim=-1)  # (B, channels) — global average pool over time

        out = self.head(x)  # (B, n_outputs)

        return out.squeeze(-1) if self.n_outputs == 1 else out
