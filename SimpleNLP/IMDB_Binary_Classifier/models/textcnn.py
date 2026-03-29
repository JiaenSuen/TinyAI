import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_classes=2,
                 kernel_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)                    # (batch_size, seq_len, embed_dim)
        embeds = embeds.unsqueeze(1)                  # (batch_size, 1, seq_len, embed_dim)
        

        conv_results = []
        for conv in self.convs:
            conv_out = F.relu(conv(embeds)).squeeze(3)        # (batch_size, num_filters, seq_len - k + 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, num_filters)
            conv_results.append(pooled)
        
  
        out = torch.cat(conv_results, dim=1)   # (batch_size, len(kernel_sizes)*num_filters)
        out = self.dropout(out)
        out = self.fc(out)
        return out