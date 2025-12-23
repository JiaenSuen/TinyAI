import torch
import torch.nn as nn
import torch.nn.functional as F

# LSTM 
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        return self.fc(h[-1])


# GRU 
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        return self.fc(h[-1])


# CNN Text 
class CNNTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv2d(1, hidden_dim, (3, embed_dim))
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)            # [B, T, E]
        emb = emb.unsqueeze(1)             # [B, 1, T, E]
        conv = F.relu(self.conv(emb))      # [B, H, T-2, 1]
        pooled = F.max_pool2d(conv, (conv.size(2), 1))
        pooled = pooled.squeeze(3)         
        pooled = pooled.squeeze(2)         
        return self.fc(pooled)            

