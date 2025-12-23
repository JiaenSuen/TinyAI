 
from collections import Counter
import torch

def build_vocab(tokenized_texts, min_freq=1):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab

def tokens_to_ids(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]




PAD = "<PAD>"
UNK = "<UNK>"

def build_vocab(tokenized_texts):
    vocab = {PAD: 0, UNK: 1}
    counter = Counter()

    for tokens in tokenized_texts:
        counter.update(tokens)

    for token in counter:
        vocab[token] = len(vocab)

    return vocab

def text_to_ids(tokens, vocab):
    return [vocab.get(t, vocab[UNK]) for t in tokens]

def pad_sequences(seqs):
    max_len = max(len(s) for s in seqs)
    padded = [
        s + [0] * (max_len - len(s))
        for s in seqs
    ]
    return torch.tensor(padded)





import re

def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text) 
    text = text.replace('\u3000', ' ') 
    return text






import spacy
from typing import List
 

# python -m spacy download en_core_web_sm
class SpacyTokenizer:
    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name, disable=["tagger", "parser", "ner"])

    def tokenize(self, text: str) -> List[str]:
        text = normalize_text(text)
        doc = self.nlp(text)
        return [token.text for token in doc]

class SimpleTokenizer:
    def __init__(self, lang="en"):
        self.nlp = spacy.blank(lang)

    def tokenize(self, text: str):
        return [t.text.lower() for t in self.nlp(text)]


import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
class NLTKTokenizer:
    def tokenize(self, text: str) -> List[str]:
        text = normalize_text(text)
        return word_tokenize(text)





class CharTokenizer:
    def tokenize(self, text: str) -> List[str]:
        text = normalize_text(text)
        return list(text)
    


