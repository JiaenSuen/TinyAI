import torch
import torch.nn as nn

class IMDBClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=2):
        super(IMDBClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        # embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        # LSTM
        output, (hidden, cell) = self.lstm(embedded)  # hidden: [1, batch_size, hidden_dim]
        final_hidden = hidden[-1]  # [batch_size, hidden_dim]
        logits = self.fc(final_hidden)  # [batch_size, num_classes]
        return logits
