# models.py
import torch
import torch.nn as nn
import random

class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(LSTM_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)  # This is what we need for attention
        return outputs, hidden, cell  # Changed: Return outputs as well

class LSTM_Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(LSTM_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers, dropout=p)  # Input size includes context
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # For attention weights

    def forward(self, x, hidden, cell, encoder_outputs):
        # x shape: (N) -> unsqueeze to (1, N)
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        # Attention: Dot-product (Luong style)
        # Assume single layer for simplicity; for multi-layer, use hidden[-1]
        query = hidden[0].squeeze(0) if self.num_layers > 1 else hidden.squeeze(0)  # (N, hidden_size)
        # encoder_outputs: (seq_len_src, N, hidden_size)
        attn_scores = torch.einsum("nh, snh -> ns", query, encoder_outputs)  # (N, seq_len_src)
        attn_weights = self.softmax(attn_scores)  # (N, seq_len_src)
        context = torch.einsum("ns, snh -> nh", attn_weights, encoder_outputs)  # (N, hidden_size)
        context = context.unsqueeze(0)  # (1, N, hidden_size)

        # Concat embedding and context
        rnn_input = torch.cat((embedding, context), dim=2)  # (1, N, embedding_size + hidden_size)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)
        predictions = self.fc(outputs).squeeze(0)  # (N, output_size)
        return predictions, hidden, cell











class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teach_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(source.device)
        encoder_outputs, hidden, cell = self.encoder(source)  # Changed: Unpack encoder_outputs
        x = target[0]  # start token

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell, encoder_outputs)  # Changed: Pass encoder_outputs
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teach_force_ratio else best_guess

        return outputs

    def translate(self, source, max_len=50, bos_id=None, eos_id=None):
        """
        Greedy decoding for inference
        source: (seq_len, batch=1) tensor
        """
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(source)  # Changed: Unpack encoder_outputs
            batch_size = source.size(1)

            # Use the BOS token or source the 0th token.
            if bos_id is not None:
                x = torch.tensor([bos_id]*batch_size, device=source.device)
            else:
                x = source[0]

            outputs = []
            for _ in range(max_len):
                output, hidden, cell = self.decoder(x, hidden, cell, encoder_outputs)  # Changed: Pass encoder_outputs
                best_guess = output.argmax(1)  # (batch_size,)
                outputs.append(best_guess[0].item())   
                x = best_guess
                if eos_id is not None and best_guess[0].item() == eos_id:
                    break

        return outputs