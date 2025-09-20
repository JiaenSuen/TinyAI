import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import pickle

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定義 Dataset
class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = word_tokenize(text.lower())[:self.max_len]
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        if len(indices) < self.max_len:
            indices += [self.vocab['<pad>']] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 詞彙表建立
def build_vocab(texts, min_freq=1, specials=['<unk>', '<pad>']):
    counter = Counter()
    for text in texts:
        tokens = word_tokenize(text.lower())
        counter.update(tokens)
    vocab = {word: idx + len(specials) for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    for i, special in enumerate(specials):
        vocab[special] = i
    return vocab

# 載入 GloVe 向量
def load_glove_vectors(glove_file='glove/glove.6B.100d.txt'):
    glove_vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            glove_vectors[word] = vector
    return glove_vectors

# 建立嵌入矩陣
def create_embedding_matrix(vocab, glove_vectors, embed_dim=100):
    embedding_matrix = np.zeros((len(vocab), embed_dim))
    for word, idx in vocab.items():
        if word in glove_vectors:
            embedding_matrix[idx] = glove_vectors[word]
        else:
            embedding_matrix[idx] = np.random.normal(0, 0.1, embed_dim)
    return torch.tensor(embedding_matrix, dtype=torch.float32)

# 模型定義
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# 單次預測函數
def predict_single(text, model_path='agnews_model.pth', vocab_path='vocab.pkl',
                  class_map_path='class_map.pkl', max_len=100):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(class_map_path, 'rb') as f:
        class_map = pickle.load(f)

    model = TextClassifier(len(vocab), 100, 128, len(class_map)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tokens = word_tokenize(text.lower())[:max_len]
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if len(indices) < max_len:
        indices += [vocab['<pad>']] * (max_len - len(indices))
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item(), class_map[predicted_idx.item()]