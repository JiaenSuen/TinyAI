# Text to Tokenizer & Vocab
import torch
from torch.utils.data import Dataset
from vocab import build_vocab , tokens_to_ids
class IMDBDataset(Dataset):
    def __init__(self, data, vocab, tokenizer, max_len=100):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['review']
        label = self.data.iloc[idx]['sentiment']

        tokens = self.tokenizer(text)  
        token_ids = tokens_to_ids(tokens, self.vocab)

        # 補齊 padding
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)
