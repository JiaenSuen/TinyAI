# Text to Tokenizer & Vocab
import torch
from torch.utils.data import Dataset
from dataset.vocab import build_vocab , tokens_to_ids

'''
label : 1/0 

MLP     : TF-IDF / BoW / Averaged Embedding  (batch, feature_dim)
SeqNet  : Token IDs + Embedding              (batch, max_len)

'''

import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class IMDBSequenceDataset(Dataset):
    def __init__(self, df, vocab, tokenizer, max_len=256, 
                 add_special_tokens=False):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['review']
        label = self.df.iloc[idx]['sentiment']

        tokens = self.tokenizer(text)
        token_ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]

        # Cut off / padding
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'length': min(len(tokens), self.max_len)   # pack_padded_sequence
        }


class IMDBVectorDataset(Dataset):
    def __init__(self, df, max_features=10000):
        self.df = df.reset_index(drop=True)
        self.labels = torch.tensor(self.df['sentiment'].values, dtype=torch.long)
        
 
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
 
        self.features = self.vectorizer.fit_transform(self.df['review']).toarray()
        self.features = torch.tensor(self.features, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx]
        }
    


def get_dataset(dataset_type: str, df, vocab=None, tokenizer=None, max_len=256, max_features=10000):
    if dataset_type == "sequence":
        return IMDBSequenceDataset(df, vocab, tokenizer, max_len)
    elif dataset_type == "vector":
        return IMDBVectorDataset(df, max_features)
    else:
        raise ValueError("dataset_type must be 'sequence' or 'vector'")