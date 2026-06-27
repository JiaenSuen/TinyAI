import torch
import torch.nn as nn
 
from models.ts_models import (
    LSTMRegressor,
    GRURegressor,
    ConvRNNRegressor,
    DLinearRegressor,
    NBeatsRegressor,
    NHiTSRegressor,
    PatchTSTRegressor,
    TimesNetRegressor,
    TCNRegressor,
    xLSTMRegressor,
    ConvLSTMRegressor,
    RWKVRegressor,
    TransformerRegressor,
)
from models.advanced_gru_models import (
    LuongAttentionGRU,
    SelfAttentionGRU,
    MultiAttentionGRU,
    CNNGRU,
    CNNGRULSTM,
    StackedGRULSTM,
    GRUTransformer,
    DeepFusionGRU,
)


class ITransformerRegressor(nn.Module):
    """
    A practical iTransformer-style proxy.

    Input shape: [B, T, F]
    The model attends over variables by treating each variable as a token.
    """

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
        self.input_dim = input_dim

        self.time_proj = nn.Linear(seq_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, input_dim, d_model))

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
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2).contiguous()
        z = self.time_proj(x)  # [B, F, D]
        z = z + self.pos_emb[:, : z.size(1), :]
        z = self.encoder(z)
        z = z.mean(dim=1)
        return self.head(z).squeeze(-1)


def build_model(model_name, input_dim, seq_len):
    model_name = model_name.lower()

    if model_name == "":
        return None

    elif model_name == "lstm":
        return LSTMRegressor(input_dim=input_dim, hidden_dim=128, num_layers=1, dropout=0.1)

    elif model_name == "gru":
        return GRURegressor(input_dim=input_dim, hidden_dim=128, num_layers=1, dropout=0.1)

    elif model_name == "ConvRNN": # ConvRNN cell
        return ConvRNNRegressor(input_dim=input_dim, hidden_dim=128, num_layers=2, kernel_size=3, dropout=0.1)
    
    elif model_name == "convlstm":
        return ConvLSTMRegressor(input_dim=input_dim, hidden_dim=128, conv_channels=128)

    elif model_name == "dlinear":
        return DLinearRegressor(input_dim=input_dim, seq_len=seq_len, kernel_size=25, dropout=0.0)

    elif model_name == "nbeats":
        return NBeatsRegressor(input_dim=input_dim, seq_len=seq_len, n_blocks=4, hidden_dim=256, n_layers=4, dropout=0.1)

    elif model_name == "nhits":
        return NHiTSRegressor(input_dim=input_dim, seq_len=seq_len, pool_sizes=(1, 2, 4), hidden_dim=256, n_layers=3, dropout=0.1)

    elif model_name == "patchtst":
        return PatchTSTRegressor(input_dim=input_dim, seq_len=seq_len, patch_len=16, stride=8, d_model=128)

    elif model_name == "itransformer":
        return ITransformerRegressor(input_dim=input_dim, seq_len=seq_len, d_model=128, n_heads=8, n_layers=3, dropout=0.1)

    elif model_name == "timesnet":
        return TimesNetRegressor(input_dim=input_dim, seq_len=seq_len, d_model=128, depth=2, top_k=3)

    elif model_name == "tcn":
        return TCNRegressor(input_dim=input_dim, hidden_dim=128, kernel_size=3, levels=4)

    elif model_name == "xlstm":
        return xLSTMRegressor(input_dim=input_dim, hidden_dim=128, depth=2)

    

    elif model_name == "rwkv":
        return RWKVRegressor(input_dim=input_dim, hidden_dim=128, depth=3)

    elif model_name == "transformer":
        return TransformerRegressor(input_dim=input_dim, seq_len=seq_len, d_model=128, n_heads=8, n_layers=3)

    elif model_name == "luong_gru":
        return LuongAttentionGRU(input_dim=input_dim, hidden_dim=128, num_layers=1, dropout=0.1)

    elif model_name == "selfattn_gru":
        return SelfAttentionGRU(input_dim=input_dim, hidden_dim=128, num_layers=1, num_heads=4, dropout=0.1)

    elif model_name == "multiattn_gru":
        return MultiAttentionGRU(input_dim=input_dim, hidden_dim=128, num_layers=2, attn_heads=4, dropout=0.1)

    elif model_name == "cnn_gru":
        return CNNGRU(input_dim=input_dim, hidden_dim=128, conv_channels=64, kernel_size=3, num_layers=1, dropout=0.1)

    elif model_name == "cnn_gru_lstm":
        return CNNGRULSTM(input_dim=input_dim, hidden_dim=128, conv_channels=64, kernel_size=3, dropout=0.1)

    elif model_name == "stacked_gru_lstm":
        return StackedGRULSTM(input_dim=input_dim, hidden_dim=128, dropout=0.1)

    elif model_name == "gru_transformer":
        return GRUTransformer(input_dim=input_dim, seq_len=seq_len, hidden_dim=128, n_heads=8, n_layers=2, dropout=0.1)

    elif model_name == "deep_fusion_gru":
        return DeepFusionGRU(input_dim=input_dim, hidden_dim=128, conv_channels=64, attn_heads=4, dropout=0.15)

    else:
        raise ValueError(f"Unknown model: {model_name}")