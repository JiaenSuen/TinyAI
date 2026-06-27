import torch
import torch.nn as nn
import torch.nn.functional as F


 
class LuongAttention(nn.Module):
    """
    Luong dot-product attention
    query: [B, H]
    keys : [B, T, H]
    context: [B, H]
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = hidden_dim ** 0.5

    def forward(self, query, keys):
        # scores: [B, T]
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / self.scale
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class MultiAttentionPool(nn.Module):
    """
    Use multiple independent attention points pooling
    hidden: [B, T, H]
    """
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_heads)
        ])

    def forward(self, hidden):
        contexts = []
        weights_all = []

        for attn in self.attn_layers:
            scores = attn(hidden).squeeze(-1)       # [B, T]
            weights = torch.softmax(scores, dim=1)   # [B, T]
            context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)  # [B, H]
            contexts.append(context)
            weights_all.append(weights)

        context = torch.cat(contexts, dim=1)  # [B, H*num_heads]
        return context, weights_all


# 1) Luong Attention + GRU
class LuongAttentionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = LuongAttention(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        outputs, hidden = self.gru(x)      # outputs: [B, T, H]
        query = hidden[-1]                 # [B, H]
        context, _ = self.attn(query, outputs)
        feat = torch.cat([query, context], dim=1)
        return self.head(feat).squeeze(-1)


# 2) Self-Attention GRU
class SelfAttentionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, num_heads=4, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        outputs, hidden = self.gru(x)              # [B, T, H]
        q = hidden[-1].unsqueeze(1)                # [B, 1, H]
        attn_out, _ = self.self_attn(q, outputs, outputs)
        context = attn_out.squeeze(1)              # [B, H]
        last = hidden[-1]                          # [B, H]
        feat = torch.cat([last, context], dim=1)
        return self.head(feat).squeeze(-1)


# 3) 多層 Attention GRU
class MultiAttentionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, attn_heads=4, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.multi_attn = MultiAttentionPool(hidden_dim, num_heads=attn_heads)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * (1 + attn_heads), hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, x):
        outputs, hidden = self.gru(x)
        last = hidden[-1]                       # [B, H]
        context, _ = self.multi_attn(outputs)    # [B, H * attn_heads]
        feat = torch.cat([last, context], dim=1)
        return self.head(feat).squeeze(-1)


# 4) CNN-GRU
class CNNGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, conv_channels=64, kernel_size=3, num_layers=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)     # [B, F, T]
        z = self.conv(x)          # [B, C, T]
        z = z.transpose(1, 2)     # [B, T, C]
        _, hidden = self.gru(z)
        feat = hidden[-1]
        return self.head(feat).squeeze(-1)


# 5) CNN-GRU-LSTM
class CNNGRULSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, conv_channels=64, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(conv_channels, hidden_dim, batch_first=True)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)     # [B, F, T]
        z = self.conv(x)          # [B, C, T]
        z = z.transpose(1, 2)     # [B, T, C]
        z, _ = self.gru(z)        # [B, T, H]
        z, (h, c) = self.lstm(z)  # [B, T, H]
        feat = h[-1]
        return self.head(feat).squeeze(-1)


# 6) Stacked GRU + LSTM
class StackedGRULSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        z, _ = self.gru(x)
        z, (h, c) = self.lstm(z)
        feat = h[-1]
        return self.head(feat).squeeze(-1)


# 7) GRU-Transformer
class GRUTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=128, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        z, _ = self.gru(x)  # [B, T, H]
        B, T, H = z.shape

        if T <= self.seq_len:
            z = z + self.pos_emb[:, :T, :]
        else:
            pos = F.interpolate(
                self.pos_emb.transpose(1, 2),
                size=T,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)
            z = z + pos

        z = self.encoder(z)
        feat = z.mean(dim=1)
        return self.head(feat).squeeze(-1)


# 8) Deep Fusion GRU
class DeepFusionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, conv_channels=64, attn_heads=4, dropout=0.15):
        super().__init__()

        # branch A: raw GRU
        self.gru_raw = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # branch B: CNN -> GRU
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru_conv = nn.GRU(conv_channels, hidden_dim, batch_first=True)

        # branch C: attention over raw GRU outputs
        self.attn_pool = MultiAttentionPool(hidden_dim, num_heads=attn_heads)

        fusion_dim = hidden_dim + hidden_dim + hidden_dim * attn_heads
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # branch A
        out_a, hid_a = self.gru_raw(x)
        feat_a = hid_a[-1]

        # branch B
        z = x.transpose(1, 2)       # [B, F, T]
        z = self.conv(z)            # [B, C, T]
        z = z.transpose(1, 2)       # [B, T, C]
        _, hid_b = self.gru_conv(z)
        feat_b = hid_b[-1]

        # branch C
        feat_c, _ = self.attn_pool(out_a)

        feat = torch.cat([feat_a, feat_b, feat_c], dim=1)
        return self.fusion(feat).squeeze(-1)


 