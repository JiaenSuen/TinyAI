"""Sequence classification models for predictive maintenance.

Includes recurrent, attention, state-space-inspired, convolutional, patch,
frequency-domain, and lightweight classification backbones.

Add future models in three steps:
    1. Define the model class.
    2. Register the class in MODEL_CLASSES.
    3. Add its configuration to models_router.

All models receive:
    [batch_size, sequence_length, input_size]

All models return:
    [batch_size, num_classes]
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# Model router
# =============================================================================

models_router: dict[str, dict[str, Any]] = {
    "lstm": {
        "class_name": "LSTMClassifier",
        "model_kwargs": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": False,
        },
        "training_kwargs": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    "transformer": {
        "class_name": "TransformerClassifier",
        "model_kwargs": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.2,
        },
        "training_kwargs": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
    },
    "linear_transformer": {
        "class_name": "LinearTransformerClassifier",
        "model_kwargs": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.2,
            "attention_eps": 1e-6,
        },
        "training_kwargs": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
    },
    "retnet": {
        "class_name": "RetNetClassifier",
        "model_kwargs": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.2,
            "norm_eps": 1e-6,
        },
        "training_kwargs": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
    },
    "mamba": {
        "class_name": "MambaLiteClassifier",
        "model_kwargs": {
            "d_model": 96,
            "num_layers": 3,
            "expand": 2,
            "conv_kernel": 5,
            "long_dilation": 4,
            "dropout": 0.2,
            "norm_eps": 1e-6,
        },
        "training_kwargs": {
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
        },
    },
    "modern_tcn": {
        "class_name": "ModernTCNClassifier",
        "model_kwargs": {
            "d_model": 96,
            "patch_size": 8,
            "patch_stride": 4,
            "num_blocks": 4,
            "large_kernel": 31,
            "small_kernel": 5,
            "expansion": 2,
            "dropout": 0.2,
        },
        "training_kwargs": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    "patch_tst": {
        "class_name": "PatchTSTClassifier",
        "model_kwargs": {
            "d_model": 96,
            "patch_length": 24,
            "stride": 12,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 192,
            "dropout": 0.2,
        },
        "training_kwargs": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
    },
    "timesnet": {
        "class_name": "TimesNetClassifier",
        "model_kwargs": {
            "d_model": 64,
            "num_layers": 2,
            "top_k": 3,
            "dim_feedforward": 128,
            "dropout": 0.2,
        },
        "training_kwargs": {
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
        },
    },
    "tslanet": {
        "class_name": "TSLANetClassifier",
        "model_kwargs": {
            "d_model": 96,
            "patch_length": 16,
            "stride": 8,
            "num_layers": 3,
            "expansion": 2,
            "dropout": 0.2,
            "norm_eps": 1e-6,
        },
        "training_kwargs": {
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
        },
    },
    "lite": {
        "class_name": "LITEMVClassifier",
        "model_kwargs": {
            "d_model": 96,
            "num_blocks": 3,
            "branch_channels": 48,
            "dropout": 0.2,
        },
        "training_kwargs": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
    },
}


def _validate_sequence_input(
    x: torch.Tensor,
    *,
    input_size: int,
    sequence_length: int | None = None,
) -> None:
    """Validate the shared [batch, sequence, feature] input."""

    if x.ndim != 3:
        raise ValueError(
            "Expected [batch_size, sequence_length, input_size], "
            f"but received {tuple(x.shape)}."
        )

    if x.size(-1) != input_size:
        raise ValueError(
            f"Expected {input_size} input features, "
            f"but received {x.size(-1)}."
        )

    if sequence_length is not None and x.size(1) > sequence_length:
        raise ValueError(
            f"Expected at most {sequence_length} timesteps, "
            f"but received {x.size(1)}."
        )


class RMSNorm(nn.Module):
    """Portable RMS normalization for PyTorch versions without nn.RMSNorm."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        if self.weight is not None:
            normalized = normalized * self.weight
        return normalized



class TemporalAttentionPooling(nn.Module):
    """Learn a weighted sequence summary instead of uniformly averaging time."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class LSTMClassifier(nn.Module):
    """LSTM classifier using the final recurrent hidden state."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        del sequence_length

        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )

        representation_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.LayerNorm(representation_size),
            nn.Linear(representation_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence_input(x, input_size=self.input_size)
        _, (hidden_state, _) = self.lstm(x)

        if self.bidirectional:
            representation = torch.cat(
                [hidden_state[-2], hidden_state[-1]], dim=-1
            )
        else:
            representation = hidden_state[-1]

        return self.classifier(representation)


class TransformerClassifier(nn.Module):
    """Transformer encoder classifier with learnable position embeddings."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(input_size, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence_input(
            x,
            input_size=self.input_size,
            sequence_length=self.sequence_length,
        )
        tokens = self.input_projection(x)
        tokens = tokens + self.position_embedding[:, : x.size(1)]
        encoded = self.encoder(tokens)
        return self.classifier(encoded.mean(dim=1))


# =============================================================================
# Linear Transformer
# =============================================================================

class LinearSelfAttention(nn.Module):
    """Multi-head ELU+1 linear attention without a [T, T] matrix."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.eps = eps

        self.q_projection = nn.Linear(d_model, d_model, bias=False)
        self.k_projection = nn.Linear(d_model, d_model, bias=False)
        self.v_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_dropout = nn.Dropout(dropout)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        return x.view(
            batch_size, sequence_length, self.nhead, self.head_dim
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        queries = self._feature_map(
            self._split_heads(self.q_projection(x))
        )
        keys = self._feature_map(
            self._split_heads(self.k_projection(x))
        )
        values = self._split_heads(self.v_projection(x))

        key_value = torch.einsum("bhtd,bhte->bhde", keys, values)
        key_sum = keys.sum(dim=2)
        normalizer = torch.einsum(
            "bhtd,bhd->bht", queries, key_sum
        ).clamp_min(self.eps).reciprocal()

        attended = torch.einsum(
            "bhtd,bhde,bht->bhte",
            queries,
            key_value,
            normalizer,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.d_model
        )
        return self.output_dropout(self.output_projection(attended))


class LinearTransformerEncoderLayer(nn.Module):
    """Pre-normalized Linear Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attention_eps: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = LinearSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            eps=attention_eps,
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feedforward(self.feedforward_norm(x))
        return x


class LinearTransformerClassifier(nn.Module):
    """Linear-attention Transformer classifier for full-window encoding."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        attention_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(input_size, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                LinearTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_eps=attention_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence_input(
            x,
            input_size=self.input_size,
            sequence_length=self.sequence_length,
        )
        tokens = self.input_projection(x)
        tokens = tokens + self.position_embedding[:, : x.size(1)]
        tokens = self.input_dropout(tokens)

        for layer in self.layers:
            tokens = layer(tokens)

        representation = self.final_norm(tokens).mean(dim=1)
        return self.classifier(representation)


# =============================================================================
# RetNet
# =============================================================================

def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class MultiScaleRetention(nn.Module):
    """RetNet parallel multi-scale causal retention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        if self.head_dim % 2 != 0:
            raise ValueError(
                "RetNet head dimension must be even for rotary positions."
            )

        self.scaling = self.head_dim**-0.5
        self.q_projection = nn.Linear(d_model, d_model, bias=False)
        self.k_projection = nn.Linear(d_model, d_model, bias=False)
        self.v_projection = nn.Linear(d_model, d_model, bias=False)
        self.gate_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

        decay = torch.log(
            1.0
            - 2.0
            ** (-5.0 - torch.arange(nhead, dtype=torch.float32))
        )
        angle = 1.0 / (
            10000.0
            ** torch.linspace(0.0, 1.0, self.head_dim // 2)
        )

        self.register_buffer("decay", decay)
        self.register_buffer("angle", angle.repeat_interleave(2))
        self.head_norm = RMSNorm(
            self.head_dim,
            eps=norm_eps,
            elementwise_affine=False,
        )
        self.output_dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_projection.weight, gain=2.0**-2.5)
        nn.init.xavier_uniform_(self.k_projection.weight, gain=2.0**-2.5)
        nn.init.xavier_uniform_(self.v_projection.weight, gain=2.0**-2.5)
        nn.init.xavier_uniform_(self.gate_projection.weight, gain=2.0**-2.5)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=2.0**-1.0)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        return x.view(
            batch_size, sequence_length, self.nhead, self.head_dim
        ).transpose(1, 2)

    def _relative_position_terms(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            sequence_length, device=device, dtype=torch.float32
        )
        angles = positions[:, None] * self.angle.float()[None, :]
        sine = torch.sin(angles).to(dtype=dtype)[None, None]
        cosine = torch.cos(angles).to(dtype=dtype)[None, None]

        distance = positions[:, None] - positions[None, :]
        causal = distance >= 0
        decay_mask = torch.exp(
            self.decay.float()[:, None, None]
            * distance.clamp_min(0.0)[None]
        )
        decay_mask = decay_mask * causal[None]
        decay_mask = decay_mask / decay_mask.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6).sqrt()

        return sine, cosine, decay_mask.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        queries = self._split_heads(self.q_projection(x))
        keys = self._split_heads(self.k_projection(x)) * self.scaling
        values = self._split_heads(self.v_projection(x))
        gates = self.gate_projection(x)

        sine, cosine, decay_mask = self._relative_position_terms(
            sequence_length,
            device=x.device,
            dtype=x.dtype,
        )
        queries = queries * cosine + _rotate_every_two(queries) * sine
        keys = keys * cosine + _rotate_every_two(keys) * sine

        retention_scores = torch.matmul(
            queries, keys.transpose(-1, -2)
        )
        retention_scores = retention_scores * decay_mask[None]
        score_scale = retention_scores.detach().abs().sum(
            dim=-1, keepdim=True
        ).clamp(min=1.0, max=5e4)
        retention_scores = retention_scores / score_scale

        retained = torch.matmul(retention_scores, values).transpose(1, 2)
        retained = self.head_norm(retained).reshape(
            batch_size, sequence_length, self.d_model
        )
        retained = F.silu(gates) * retained
        return self.output_dropout(self.output_projection(retained))


class RetNetFeedForward(nn.Module):
    """SwiGLU-style feed-forward block."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(d_model, dim_feedforward)
        self.value_projection = nn.Linear(d_model, dim_feedforward)
        self.output_projection = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_projection(x))
        hidden = hidden * self.value_projection(x)
        hidden = self.dropout(hidden)
        return self.dropout(self.output_projection(hidden))


class RetNetEncoderLayer(nn.Module):
    """Pre-normalized RetNet layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.retention_norm = RMSNorm(d_model, eps=norm_eps)
        self.retention = MultiScaleRetention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            norm_eps=norm_eps,
        )
        self.feedforward_norm = RMSNorm(d_model, eps=norm_eps)
        self.feedforward = RetNetFeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.retention(self.retention_norm(x))
        x = x + self.feedforward(self.feedforward_norm(x))
        return x


class RetNetClassifier(nn.Module):
    """RetNet classifier using parallel multi-scale causal retention."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if (d_model // nhead) % 2 != 0:
            raise ValueError("d_model // nhead must be even for RetNet.")

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(input_size, d_model)
        self.input_norm = RMSNorm(d_model, eps=norm_eps)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                RetNetEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, eps=norm_eps)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence_input(
            x,
            input_size=self.input_size,
            sequence_length=self.sequence_length,
        )
        tokens = self.input_dropout(
            self.input_norm(self.input_projection(x))
        )

        for layer in self.layers:
            tokens = layer(tokens)

        representation = self.final_norm(tokens).mean(dim=1)
        return self.classifier(representation)


# =============================================================================
# Fast Windows-safe Mamba-lite model
# =============================================================================


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise temporal convolution with explicit left-only padding."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1.")
        if dilation < 1:
            raise ValueError("dilation must be at least 1.")

        self.left_padding = dilation * (kernel_size - 1)
        self.convolution = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C] -> [B, C, T]
        channels_first = x.transpose(1, 2)
        channels_first = F.pad(
            channels_first,
            (self.left_padding, 0),
        )
        convolved = self.convolution(channels_first)
        return convolved.transpose(1, 2)


class MambaLiteMixer(nn.Module):
    """Fast Mamba-inspired gated temporal mixer using standard PyTorch.

    This block intentionally avoids selective-scan loops, cumulative products,
    Triton, and custom CUDA extensions. It retains the lightweight structure
    useful in Mamba-style sequence models:

    * expanded value and gate streams,
    * causal depthwise temporal mixing,
    * short- and long-range paths,
    * input-dependent multiplicative gating,
    * learnable skip connection,
    * output projection back to ``d_model``.

    It is designed for fair predictive-maintenance comparisons on Windows and
    should be described as ``Mamba-Lite`` rather than official Mamba/Mamba-3.
    """

    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        conv_kernel: int = 5,
        long_dilation: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if expand < 1:
            raise ValueError("expand must be at least 1.")

        self.d_inner = d_model * expand

        # One projection creates the content and gate streams.
        self.input_projection = nn.Linear(
            d_model,
            2 * self.d_inner,
        )

        # Local and dilated paths approximate multiple temporal timescales.
        self.local_mixer = CausalDepthwiseConv1d(
            channels=self.d_inner,
            kernel_size=conv_kernel,
            dilation=1,
        )
        self.long_mixer = CausalDepthwiseConv1d(
            channels=self.d_inner,
            kernel_size=conv_kernel,
            dilation=long_dilation,
        )

        # Channel-wise controls keep the block inexpensive.
        self.path_mix = nn.Parameter(torch.zeros(self.d_inner))
        self.skip = nn.Parameter(torch.ones(self.d_inner))

        self.output_projection = nn.Linear(
            self.d_inner,
            d_model,
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, gates = self.input_projection(x).chunk(2, dim=-1)

        local_features = self.local_mixer(values)
        long_features = self.long_mixer(values)

        mix = torch.sigmoid(self.path_mix)[None, None, :]
        temporal_features = (
            (1.0 - mix) * local_features
            + mix * long_features
        )

        temporal_features = F.silu(temporal_features)
        temporal_features = (
            temporal_features
            + self.skip[None, None, :] * values
        )

        # Input-dependent output selection, analogous to Mamba's gate branch.
        gated = temporal_features * F.silu(gates)
        output = self.output_projection(gated)
        return self.output_dropout(output)


class MambaLiteEncoderLayer(nn.Module):
    """Pre-normalized residual layer using the Mamba-lite mixer."""

    def __init__(
        self,
        d_model: int,
        expand: int,
        conv_kernel: int,
        long_dilation: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()

        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.mixer = MambaLiteMixer(
            d_model=d_model,
            expand=expand,
            conv_kernel=conv_kernel,
            long_dilation=long_dilation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MambaLiteClassifier(nn.Module):
    """Small, fast Mamba-inspired classifier with no external dependencies.

    The default configuration is deliberately close to the existing LSTM in
    parameter count and below the Linear Transformer. It runs with ordinary
    PyTorch on native Windows, CPU, or CUDA.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
        d_model: int = 128,
        num_layers: int = 2,
        expand: int = 2,
        conv_kernel: int = 5,
        long_dilation: int = 4,
        dropout: float = 0.2,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.input_size = input_size
        self.sequence_length = sequence_length

        self.input_projection = nn.Linear(input_size, d_model)
        self.input_norm = RMSNorm(d_model, eps=norm_eps)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                MambaLiteEncoderLayer(
                    d_model=d_model,
                    expand=expand,
                    conv_kernel=conv_kernel,
                    long_dilation=long_dilation * (2**layer_index),
                    dropout=dropout,
                    norm_eps=norm_eps,
                )
                for layer_index in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(d_model, eps=norm_eps)
        self.pooling = TemporalAttentionPooling(d_model, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence_input(
            x,
            input_size=self.input_size,
            sequence_length=self.sequence_length,
        )

        tokens = self.input_dropout(
            self.input_norm(self.input_projection(x))
        )

        for layer in self.layers:
            tokens = layer(tokens)

        representation = self.pooling(self.final_norm(tokens))
        return self.classifier(representation)



# =============================================================================
# ModernTCN adapted for fixed-window classification
# =============================================================================


class SamePadDepthwiseConv1d(nn.Module):
    """Odd-kernel depthwise convolution preserving temporal length."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same padding.")
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ModernTCNBlock(nn.Module):
    """Large-kernel depthwise temporal mixing plus inverted bottleneck."""

    def __init__(
        self,
        d_model: int,
        large_kernel: int,
        small_kernel: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.large_kernel = SamePadDepthwiseConv1d(d_model, large_kernel)
        self.small_kernel = SamePadDepthwiseConv1d(d_model, small_kernel)
        self.norm = nn.GroupNorm(1, d_model)
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.layer_scale = nn.Parameter(torch.full((d_model,), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.large_kernel(x) + self.small_kernel(x)
        x = self.channel_mixer(self.norm(x))
        return residual + self.layer_scale[None, :, None] * x


class ModernTCNClassifier(nn.Module):
    """ModernTCN-style large-kernel convolutional sequence classifier."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
        d_model: int = 96,
        patch_size: int = 8,
        patch_stride: int = 4,
        num_blocks: int = 4,
        large_kernel: int = 31,
        small_kernel: int = 5,
        expansion: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.patch_embedding = nn.Conv1d(
            input_size,
            d_model,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=patch_size // 2,
        )
        self.blocks = nn.ModuleList(
            [
                ModernTCNBlock(
                    d_model=d_model,
                    large_kernel=large_kernel,
                    small_kernel=small_kernel,
                    expansion=expansion,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.pooling = TemporalAttentionPooling(d_model, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence_input(
            x,
            input_size=self.input_size,
            sequence_length=self.sequence_length,
        )
        tokens = self.patch_embedding(x.transpose(1, 2))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens.transpose(1, 2))
        return self.classifier(self.pooling(tokens))


# =============================================================================
# PatchTST adapted for multivariate classification
# =============================================================================


class PatchTSTClassifier(nn.Module):
    """Channel-independent patch encoder with light cross-variable attention."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
        d_model: int = 96,
        patch_length: int = 24,
        stride: int = 12,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 192,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if patch_length < 1 or stride < 1:
            raise ValueError("patch_length and stride must be positive.")

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.stride = stride
        padded_length = max(sequence_length, patch_length)
        self.max_patches = 1 + (padded_length - patch_length) // stride

        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 1, self.max_patches, d_model)
        )
        self.variable_embedding = nn.Parameter(
            torch.zeros(1, input_size, d_model)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.variable_embedding, std=0.02)

        patch_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.patch_encoder = nn.TransformerEncoder(
            patch_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        variable_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.variable_encoder = nn.TransformerEncoder(
            variable_layer,
            num_layers=1,
            norm=nn.LayerNorm(d_model),
        )
        self.variable_pooling = TemporalAttentionPooling(
            d_model,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence_input(
            x,
            input_size=self.input_size,
            sequence_length=self.sequence_length,
        )
        batch_size, sequence_length, channels = x.shape
        channel_first = x.transpose(1, 2)

        if sequence_length < self.patch_length:
            channel_first = F.pad(
                channel_first,
                (0, self.patch_length - sequence_length),
            )

        patches = channel_first.unfold(
            dimension=-1,
            size=self.patch_length,
            step=self.stride,
        )
        num_patches = patches.size(2)
        patch_tokens = self.patch_projection(patches)
        patch_tokens = patch_tokens + self.position_embedding[:, :, :num_patches]
        patch_tokens = patch_tokens.reshape(
            batch_size * channels,
            num_patches,
            -1,
        )
        patch_tokens = self.patch_encoder(patch_tokens)
        channel_tokens = patch_tokens.mean(dim=1).reshape(
            batch_size,
            channels,
            -1,
        )
        channel_tokens = channel_tokens + self.variable_embedding[:, :channels]
        channel_tokens = self.variable_encoder(channel_tokens)
        representation = self.variable_pooling(channel_tokens)
        return self.classifier(representation)


# =============================================================================
# Lightweight TimesNet adapted for classification
# =============================================================================


class DepthwiseSeparableConv2d(nn.Module):
    """Parameter-efficient 2D convolution used inside TimesBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class TimesInceptionBlock(nn.Module):
    """Multi-kernel 2D temporal variation extractor."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                DepthwiseSeparableConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                )
                for kernel_size in (1, 3, 5)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([branch(x) for branch in self.branches], dim=-1).mean(-1)


class TimesBlock(nn.Module):
    """Discover dominant periods with FFT and model them in 2D."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        top_k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            TimesInceptionBlock(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            TimesInceptionBlock(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = x.shape
        spectrum = torch.fft.rfft(x.float(), dim=1)
        amplitude = spectrum.abs()
        frequency_count = amplitude.size(1)

        available = min(self.top_k, max(0, frequency_count - 1))
        if available == 0:
            return x

        global_amplitude = amplitude.mean(dim=(0, 2))
        global_amplitude = global_amplitude.clone()
        global_amplitude[0] = -torch.inf
        frequency_indices = torch.topk(
            global_amplitude,
            k=available,
        ).indices
        sample_weights = amplitude.mean(dim=2)[:, frequency_indices]

        period_features: list[torch.Tensor] = []
        for frequency_index in frequency_indices:
            period = max(1, sequence_length // int(frequency_index.item()))
            padded_length = math.ceil(sequence_length / period) * period
            padded = x
            if padded_length > sequence_length:
                padded = F.pad(
                    x,
                    (0, 0, 0, padded_length - sequence_length),
                )

            variation_2d = padded.reshape(
                batch_size,
                padded_length // period,
                period,
                d_model,
            ).permute(0, 3, 1, 2)
            variation_2d = self.conv(variation_2d)
            restored = variation_2d.permute(0, 2, 3, 1).reshape(
                batch_size,
                padded_length,
                d_model,
            )[:, :sequence_length]
            period_features.append(restored)

        stacked = torch.stack(period_features, dim=-1)
        weights = torch.softmax(sample_weights, dim=-1).to(dtype=x.dtype)
        mixed = torch.sum(stacked * weights[:, None, None, :], dim=-1)
        return x + self.dropout(mixed)


class TimesNetClassifier(nn.Module):
    """TimesNet-style multi-period classifier with lightweight 2D kernels."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
        d_model: int = 64,
        num_layers: int = 2,
        top_k: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(input_size, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )
        self.pooling = TemporalAttentionPooling(d_model, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence_input(
            x,
            input_size=self.input_size,
            sequence_length=self.sequence_length,
        )
        tokens = self.input_projection(x)
        tokens = tokens + self.position_embedding[:, :x.size(1)]
        for block, norm in zip(self.blocks, self.norms):
            tokens = norm(block(tokens))
        return self.classifier(self.pooling(tokens))


# =============================================================================
# TSLANet adapted for supervised classification
# =============================================================================


class DepthwiseSeparableConv1d(nn.Module):
    """Length-preserving depthwise-separable convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class AdaptiveSpectralBlock(nn.Module):
    """Adaptive Fourier filtering for long-range and noise-robust features."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.base_weight = nn.Parameter(torch.randn(d_model, 2) * 0.02)
        self.high_weight = nn.Parameter(torch.randn(d_model, 2) * 0.02)
        self.threshold = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(1)
        spectrum = torch.fft.rfft(x.float(), dim=1, norm="ortho")
        base_weight = torch.view_as_complex(self.base_weight.contiguous())
        high_weight = torch.view_as_complex(self.high_weight.contiguous())

        energy = spectrum.abs().pow(2).mean(dim=-1, keepdim=True)
        energy_min = energy.amin(dim=1, keepdim=True)
        energy_max = energy.amax(dim=1, keepdim=True)
        normalized_energy = (energy - energy_min) / (
            energy_max - energy_min + 1e-6
        )
        threshold = torch.sigmoid(self.threshold)
        adaptive_mask = torch.sigmoid(
            12.0 * (normalized_energy - threshold)
        )

        filtered = spectrum * base_weight[None, None, :]
        filtered = filtered + (
            spectrum
            * adaptive_mask
            * high_weight[None, None, :]
        )
        restored = torch.fft.irfft(
            filtered,
            n=sequence_length,
            dim=1,
            norm="ortho",
        ).to(dtype=x.dtype)
        return self.dropout(restored)


class InteractiveConvBlock(nn.Module):
    """Two convolution paths interact multiplicatively as in TSLANet."""

    def __init__(
        self,
        d_model: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.branch_a = DepthwiseSeparableConv1d(
            d_model,
            hidden,
            kernel_size=3,
        )
        self.branch_b = DepthwiseSeparableConv1d(
            d_model,
            hidden,
            kernel_size=5,
        )
        self.output_projection = nn.Conv1d(hidden, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.transpose(1, 2)
        branch_a = self.branch_a(channels_first)
        branch_b = self.branch_b(channels_first)
        interacted = F.gelu(branch_a) * branch_b
        interacted = interacted + F.gelu(branch_b) * branch_a
        output = self.output_projection(interacted).transpose(1, 2)
        return self.dropout(output)


class TSLANetBlock(nn.Module):
    """Adaptive spectral block followed by interactive convolution."""

    def __init__(
        self,
        d_model: int,
        expansion: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.spectral_norm = RMSNorm(d_model, eps=norm_eps)
        self.spectral = AdaptiveSpectralBlock(d_model, dropout=dropout)
        self.conv_norm = RMSNorm(d_model, eps=norm_eps)
        self.interactive_conv = InteractiveConvBlock(
            d_model=d_model,
            expansion=expansion,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.spectral(self.spectral_norm(x))
        x = x + self.interactive_conv(self.conv_norm(x))
        return x


class TSLANetClassifier(nn.Module):
    """Lightweight spectral-convolutional classifier for noisy time series."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
        d_model: int = 96,
        patch_length: int = 16,
        stride: int = 8,
        num_layers: int = 3,
        expansion: int = 2,
        dropout: float = 0.2,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.patch_embedding = nn.Conv1d(
            input_size,
            d_model,
            kernel_size=patch_length,
            stride=stride,
            padding=patch_length // 2,
        )
        self.layers = nn.ModuleList(
            [
                TSLANetBlock(
                    d_model=d_model,
                    expansion=expansion,
                    dropout=dropout,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, eps=norm_eps)
        self.pooling = TemporalAttentionPooling(d_model, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence_input(
            x,
            input_size=self.input_size,
            sequence_length=self.sequence_length,
        )
        tokens = self.patch_embedding(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.classifier(
            self.pooling(self.final_norm(tokens))
        )


# =============================================================================
# LITEMV adapted for multivariate predictive-maintenance classification
# =============================================================================


class LITEBlock(nn.Module):
    """Multi-scale depthwise inception block with dilated branches."""

    def __init__(
        self,
        d_model: int,
        branch_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, d_model)
        self.branches = nn.ModuleList(
            [
                DepthwiseSeparableConv1d(
                    d_model,
                    branch_channels,
                    kernel_size=5,
                    dilation=1,
                ),
                DepthwiseSeparableConv1d(
                    d_model,
                    branch_channels,
                    kernel_size=9,
                    dilation=2,
                ),
                DepthwiseSeparableConv1d(
                    d_model,
                    branch_channels,
                    kernel_size=17,
                    dilation=4,
                ),
            ]
        )
        self.projection = nn.Sequential(
            nn.Conv1d(3 * branch_channels, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer_scale = nn.Parameter(torch.full((d_model,), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        features = torch.cat(
            [branch(normalized) for branch in self.branches],
            dim=1,
        )
        features = self.projection(features)
        return x + self.layer_scale[None, :, None] * features


class LITEMVClassifier(nn.Module):
    """LITE-inspired multivariate classifier with fixed signal filters."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
        d_model: int = 96,
        num_blocks: int = 3,
        branch_channels: int = 48,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length

        base_filters = torch.tensor(
            [
                [-0.5, 0.0, 0.5],
                [1.0, -2.0, 1.0],
                [0.25, 0.5, 0.25],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer(
            "custom_filters",
            base_filters.repeat(input_size, 1, 1),
        )

        self.stem = nn.Sequential(
            nn.Conv1d(4 * input_size, d_model, kernel_size=1),
            nn.GroupNorm(1, d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                LITEBlock(
                    d_model=d_model,
                    branch_channels=branch_channels,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.pooling = TemporalAttentionPooling(d_model, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _fixed_features(self, x: torch.Tensor) -> torch.Tensor:
        channel_first = x.transpose(1, 2)
        padded = F.pad(channel_first, (1, 1), mode="replicate")
        filtered = F.conv1d(
            padded,
            self.custom_filters.to(dtype=x.dtype),
            groups=self.input_size,
        )
        return torch.cat([channel_first, filtered], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence_input(
            x,
            input_size=self.input_size,
            sequence_length=self.sequence_length,
        )
        features = self.stem(self._fixed_features(x))
        for block in self.blocks:
            features = block(features)
        tokens = self.final_norm(features.transpose(1, 2))
        return self.classifier(self.pooling(tokens))


MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "LSTMClassifier": LSTMClassifier,
    "TransformerClassifier": TransformerClassifier,
    "LinearTransformerClassifier": LinearTransformerClassifier,
    "RetNetClassifier": RetNetClassifier,
    "MambaLiteClassifier": MambaLiteClassifier,
    "ModernTCNClassifier": ModernTCNClassifier,
    "PatchTSTClassifier": PatchTSTClassifier,
    "TimesNetClassifier": TimesNetClassifier,
    "TSLANetClassifier": TSLANetClassifier,
    "LITEMVClassifier": LITEMVClassifier,
}


def normalize_model_name(model_name: str) -> str:
    """Validate and normalize a routed model name."""
    normalized_name = model_name.strip().lower()

    if normalized_name not in models_router:
        available = ", ".join(sorted(models_router))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )

    return normalized_name


def build_model(
    model_name: str,
    input_size: int,
    num_classes: int,
    sequence_length: int,
) -> nn.Module:
    """Build one model using its models_router configuration."""
    normalized_name = normalize_model_name(model_name)
    route = models_router[normalized_name]
    class_name = str(route["class_name"])

    if class_name not in MODEL_CLASSES:
        raise ValueError(
            f"Model class '{class_name}' is missing from MODEL_CLASSES."
        )

    model_class = MODEL_CLASSES[class_name]
    model_kwargs = dict(route["model_kwargs"])

    return model_class(
        input_size=input_size,
        num_classes=num_classes,
        sequence_length=sequence_length,
        **model_kwargs,
    )


def get_training_kwargs(model_name: str) -> dict[str, float]:
    """Return optimizer defaults for one routed model."""
    normalized_name = normalize_model_name(model_name)
    return dict(models_router[normalized_name]["training_kwargs"])


def get_model_config(model_name: str) -> dict[str, Any]:
    """Return a copy of the complete routed model configuration."""
    normalized_name = normalize_model_name(model_name)
    route = models_router[normalized_name]

    return {
        "class_name": route["class_name"],
        "model_kwargs": dict(route["model_kwargs"]),
        "training_kwargs": dict(route["training_kwargs"]),
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    """Run one real dataset sample through every registered model."""
    from dataset import ID_TO_LABEL, create_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = create_dataloaders()
    feature_batch, _ = next(iter(train_loader))
    features = feature_batch[:1].to(device)

    input_size = features.shape[-1]
    sequence_length = features.shape[1]
    num_classes = len(ID_TO_LABEL)

    print("MODEL FORWARD TEST")
    print("=" * 80)
    print(f"Input shape: {tuple(features.shape)}")
    print(f"Device: {device}")

    for model_name in models_router:
        model = build_model(
            model_name=model_name,
            input_size=input_size,
            num_classes=num_classes,
            sequence_length=sequence_length,
        ).to(device)
        model.eval()

        with torch.no_grad():
            logits = model(features)

        expected_shape = (features.shape[0], num_classes)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"{model_name} returned {tuple(logits.shape)}; "
                f"expected {expected_shape}."
            )
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{model_name} produced non-finite logits.")

        print(
            f"{model_name:<20} | "
            f"parameters: {count_parameters(model):>10,} | "
            f"output: {tuple(logits.shape)}"
        )

    print("All registered models passed the forward test.")


if __name__ == "__main__":
    main()
