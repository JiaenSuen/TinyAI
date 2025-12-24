# models.py
import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
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
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (N) where N is batch size, we want (1, N)
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, N, hidden_size)
        predictions = self.fc(outputs)
        # predictions shape: (1, N, output_size)
        predictions = predictions.squeeze(0)
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
        hidden, cell = self.encoder(source)
        x = target[0]  # start token

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
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
            hidden, cell = self.encoder(source)
            batch_size = source.size(1)

            # 使用 BOS token 或 source 第 0 個 token
            if bos_id is not None:
                x = torch.tensor([bos_id]*batch_size, device=source.device)
            else:
                x = source[0]

            outputs = []
            for _ in range(max_len):
                output, hidden, cell = self.decoder(x, hidden, cell)
                best_guess = output.argmax(1)  # (batch_size,)
                outputs.append(best_guess[0].item())  # 只取第一個句子
                x = best_guess
                if eos_id is not None and best_guess[0].item() == eos_id:
                    break

        return outputs