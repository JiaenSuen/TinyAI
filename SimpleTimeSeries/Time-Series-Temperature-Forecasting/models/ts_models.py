import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)



class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)





class ConvRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.conv_x = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_h = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x_t, h_prev):
        # x_t: [B, F] → [B, F, 1]
        x_t = x_t.unsqueeze(-1)

        # h_prev: [B, H] → [B, H, 1]
        h_prev = h_prev.unsqueeze(-1)

        h = torch.tanh(
            self.conv_x(x_t) + self.conv_h(h_prev)
        )

        return h.squeeze(-1)
class ConvRNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, kernel_size=3, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.cells = nn.ModuleList([
            ConvRNNCell(hidden_dim, hidden_dim, kernel_size)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, F]
        B, T, _ = x.shape

        x = self.input_proj(x)

        h = [torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
             for _ in range(self.num_layers)]

        for t in range(T):
            xt = x[:, t, :]

            for i, cell in enumerate(self.cells):
                h[i] = cell(xt, h[i])
                xt = h[i]

        out = self.norm(h[-1])
        out = self.dropout(out)

        return self.head(out).squeeze(-1)


# Common
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [B, T, F]
        if self.kernel_size <= 1:
            trend = x
            seasonal = x - trend
            return seasonal, trend

        pad = (self.kernel_size - 1) // 2
        x_t = x.transpose(1, 2)  # [B, F, T]
        x_pad = F.pad(x_t, (pad, pad), mode="replicate")
        trend = self.avg(x_pad).transpose(1, 2)  # [B, T, F]
        seasonal = x - trend
        return seasonal, trend


# 1) DLinear
class DLinearRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, kernel_size=25, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.decomp = SeriesDecomposition(kernel_size=kernel_size)
        self.seasonal_linear = nn.Sequential(
            nn.Linear(seq_len * input_dim, 1),
            nn.Dropout(dropout),
        )
        self.trend_linear = nn.Sequential(
            nn.Linear(seq_len * input_dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        s = self.seasonal_linear(seasonal.reshape(x.size(0), -1))
        t = self.trend_linear(trend.reshape(x.size(0), -1))
        return (s + t).squeeze(-1)


# 2) N-BEATS
class NBeatsBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, n_layers=4, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_dim, in_dim)
        self.forecast = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.mlp(x)
        return self.backcast(h), self.forecast(h)


class NBeatsRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, n_blocks=4, hidden_dim=256, n_layers=4, dropout=0.1):
        super().__init__()
        self.in_dim = seq_len * input_dim
        self.blocks = nn.ModuleList([
            NBeatsBlock(self.in_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        residual = x.reshape(x.size(0), -1)
        forecast = 0.0
        for block in self.blocks:
            backcast, f = block(residual)
            residual = residual - backcast
            forecast = forecast + f
        return forecast.squeeze(-1)


# 3) N-HiTS (practical proxy)
class NHiTSBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_dim, in_dim)
        self.forecast = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.mlp(x)
        return self.backcast(h), self.forecast(h)


class NHiTSRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, pool_sizes=(1, 2, 4), hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.in_dim = seq_len * input_dim
        self.pool_sizes = pool_sizes
        self.blocks = nn.ModuleList([
            NHiTSBlock(self.in_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
            for _ in pool_sizes
        ])

    def _downsample(self, x, pool):
        # x: [B, T, F]
        if pool <= 1:
            return x
        x_t = x.transpose(1, 2)  # [B, F, T]
        x_p = F.avg_pool1d(x_t, kernel_size=pool, stride=pool, ceil_mode=True)
        return x_p.transpose(1, 2)

    def forward(self, x):
        residual = x
        forecast = 0.0

        for pool, block in zip(self.pool_sizes, self.blocks):
            pooled = self._downsample(residual, pool)
            pooled = F.interpolate(
                pooled.transpose(1, 2),
                size=self.seq_len,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)

            flat = pooled.reshape(x.size(0), -1)
            backcast, f = block(flat)
            residual = residual - backcast.view_as(residual)
            forecast = forecast + f

        return forecast.squeeze(-1)


# 4) PatchTST
class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride

        self.max_patches = max(1, (seq_len - patch_len) // stride + 1)
        self.patch_proj = nn.Linear(patch_len * input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_patches, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: [B, T, F]
        B, T, Fdim = x.shape

        if T < self.patch_len:
            pad_len = self.patch_len - T
            x = F.pad(x, (0, 0, 0, pad_len), mode="replicate")
            T = x.size(1)

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B, N, P, F]
        N = patches.size(1)
        patches = patches.contiguous().view(B, N, -1)  # [B, N, P*F]

        z = self.patch_proj(patches)
        z = z + self.pos_emb[:, :N, :]
        z = self.encoder(z)
        z = z.mean(dim=1)
        return self.head(z).squeeze(-1)


# 5) TimesNet (practical proxy)
class TimesBlock(nn.Module):
    def __init__(self, input_dim, d_model=128, top_k=3, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.in_proj = nn.Linear(input_dim, d_model)
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _detect_periods(self, x):
        # x: [B, T, D]
        T = x.size(1)
        x_mean = x.mean(dim=2)  # [B, T]
        fft = torch.fft.rfft(x_mean, dim=1)
        amp = fft.abs().mean(dim=0)
        if amp.numel() > 1:
            amp[0] = 0
        k = min(self.top_k, amp.numel() - 1)
        idx = torch.topk(amp[1:], k=k).indices + 1
        periods = [max(1, int(round(T / max(1, int(i.item()))))) for i in idx]
        return periods

    def forward(self, x):
        # x: [B, T, F]
        x = self.in_proj(x)  # [B, T, D]
        B, T, D = x.shape
        periods = self._detect_periods(x)

        feats = []
        for p in periods:
            pad_len = (p - (T % p)) % p
            if pad_len > 0:
                x_pad = F.pad(x, (0, 0, 0, pad_len), mode="replicate")
            else:
                x_pad = x

            T2 = x_pad.size(1)
            H = T2 // p
            y = x_pad.view(B, H, p, D).permute(0, 3, 1, 2)  # [B, D, H, p]
            y = self.conv(y)
            y = self.pool(y).squeeze(-1).squeeze(-1)        # [B, D]
            feats.append(y)

        return torch.stack(feats, dim=0).mean(dim=0)


class TimesNetRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=128, depth=2, top_k=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TimesBlock(input_dim=input_dim, d_model=d_model, top_k=top_k, dropout=dropout)
            for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        feats = [blk(x) for blk in self.blocks]
        z = torch.stack(feats, dim=0).mean(dim=0)
        return self.head(z).squeeze(-1)


# 6) TCN
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, C, T]
        y = F.pad(x, (self.pad, 0))
        y = self.conv1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = F.pad(y, (self.pad, 0))
        y = self.conv2(y)
        y = self.relu(y)
        y = self.dropout(y)

        res = x if self.down is None else self.down(x)
        return self.relu(y + res)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, kernel_size=3, levels=4, dropout=0.1):
        super().__init__()
        blocks = []
        in_ch = input_dim
        for i in range(levels):
            out_ch = hidden_dim
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, F, T]
        z = self.tcn(x)
        return self.head(z).squeeze(-1)


# 7) xLSTM-style proxy
class xLSTMCellProxy(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x_proj = nn.Linear(input_dim, 4 * hidden_dim)
        self.h_proj = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t, h, c):
        gates = self.x_proj(x_t) + self.h_proj(h)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.exp(i)
        f = torch.exp(f)
        norm = i + f + 1e-6
        i = i / norm
        f = f / norm

        g = torch.tanh(g)
        c = f * c + i * g
        h = torch.sigmoid(o) * torch.tanh(self.out_proj(c))
        return h, c


class xLSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, depth=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cells = nn.ModuleList([xLSTMCellProxy(hidden_dim, hidden_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, F]
        x = self.input_proj(x)
        B, T, H = x.shape

        h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, H, device=x.device, dtype=x.dtype)

        for t in range(T):
            xt = x[:, t, :]
            for cell in self.cells:
                h, c = cell(xt, h, c)
                xt = h

        h = self.dropout(self.norm(h))
        return self.head(h).squeeze(-1)


# 8) ConvLSTM proxy
class ConvLSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, conv_channels=128, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_channels, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)          # [B, F, T]
        x = self.conv(x)               # [B, C, T]
        x = x.transpose(1, 2)          # [B, T, C]
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


# 9) RWKV-style proxy
class RWKVBlockProxy(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mix_k = nn.Parameter(torch.rand(1, 1, dim))
        self.mix_v = nn.Parameter(torch.rand(1, 1, dim))
        self.mix_r = nn.Parameter(torch.rand(1, 1, dim))

        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.receptance = nn.Linear(dim, dim)
        self.decay = nn.Parameter(torch.zeros(dim))

        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        # x: [B, T, D]
        z = self.ln1(x)
        shift = torch.zeros_like(z)
        shift[:, 1:, :] = z[:, :-1, :]

        k_in = z * self.mix_k + shift * (1 - self.mix_k)
        v_in = z * self.mix_v + shift * (1 - self.mix_v)
        r_in = z * self.mix_r + shift * (1 - self.mix_r)

        k = torch.tanh(self.key(k_in))
        v = self.value(v_in)
        r = torch.sigmoid(self.receptance(r_in))
        decay = torch.sigmoid(self.decay).view(1, 1, -1)

        s = torch.zeros(x.size(0), x.size(2), device=x.device, dtype=x.dtype)
        outs = []
        for t in range(x.size(1)):
            s = s * decay.squeeze(0).squeeze(0) + k[:, t, :] * v[:, t, :]
            outs.append(r[:, t, :] * s)
        y = torch.stack(outs, dim=1)

        x = x + y
        x = x + self.ffn(self.ln2(x))
        return x


class RWKVRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, depth=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([RWKVBlockProxy(hidden_dim, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


# 10) Transformer
class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        B, T, Fdim = x.shape
        z = self.in_proj(x)
        if T <= self.seq_len:
            z = z + self.pos_emb[:, :T, :]
        else:
            z = z + F.interpolate(
                self.pos_emb.transpose(1, 2),
                size=T,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)

        z = self.encoder(z)
        z = z.mean(dim=1)
        return self.head(z).squeeze(-1)