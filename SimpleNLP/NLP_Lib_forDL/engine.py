import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import SimpleTokenizer, build_vocab, text_to_ids, pad_sequences


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2id):
        self.vocab = vocab
        self.label2id = label2id
        self.tokenizer = SimpleTokenizer()
        self.X = [text_to_ids(self.tokenizer.tokenize(t), vocab) for t in texts]
        self.y = [label2id[l] for l in labels]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_fn(batch):
    X, y = zip(*batch)
    X = pad_sequences(list(X))
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.long)
    else:
        X = X.clone().detach()
    y = torch.tensor(y, dtype=torch.long)
    return X, y


class nlpDLEngine:
    def __init__(self, ModelClass, embed_dim=32, hidden_dim=64, lr=1e-3, batch_size=32, device=None):
        self.tokenizer = SimpleTokenizer()
        self.ModelClass = ModelClass
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.model_name = ModelClass.__name__
        self.param_path = f"params/{self.model_name}.pt"
        os.makedirs("params", exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, vocab_size, num_classes):
        self.model = self.ModelClass(
            vocab_size=vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=num_classes
        ).to(self.device)

    def train(self, texts, labels, label2id, epochs=10):
        tokenized = [self.tokenizer.tokenize(t) for t in texts]
        self.vocab = build_vocab(tokenized)

        dataset = TextDataset(texts, labels, self.vocab, label2id)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

        self._build_model(len(self.vocab), len(label2id))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for ep in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)

            print(f"[{self.model_name}] Epoch {ep+1}, Loss={total_loss/len(dataset):.4f}")

        torch.save(self.model.state_dict(), self.param_path)
        with open(self.param_path.replace(".pt", "_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

 
 


    def test(self, texts, labels, label2id):
        vocab = self.vocab
        dataset    = TextDataset(texts, labels, vocab, label2id)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)

        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        acc = correct / total
        print(f"[{self.model_name}] Test Accuracy: {acc:.4f}")
        return acc





    def predict(self, text, id2label):
        if not os.path.exists(self.param_path):
            raise RuntimeError("Model parameters not found.")


        with open(self.param_path.replace(".pt", "_vocab.json"), "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self._build_model(len(self.vocab), len(id2label))
        self.model.load_state_dict(torch.load(self.param_path, map_location=self.device, weights_only=True))
        self.model.eval()

        tokens = self.tokenizer.tokenize(text)
        X = pad_sequences([text_to_ids(tokens, self.vocab)]).to(self.device)

        with torch.no_grad():
            pred = torch.argmax(self.model(X), dim=1).item()

        return id2label[pred]
