# utils.py
from collections import Counter
import torch
import re
import spacy
from typing import List, Dict


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u3000', ' ')
    return text




PAD = "<PAD>"
UNK = "<UNK>"



def build_vocab(tokenized_texts: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    counter = Counter()

    for tokens in tokenized_texts:
        counter.update(tokens)

    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab


def text_to_ids(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(t, vocab[UNK]) for t in tokens]


def pad_sequences(seqs: List[List[int]]) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    padded = [s + [0] * (max_len - len(s)) for s in seqs]
    return torch.tensor(padded)



class SpacyTokenizer:
    def __init__(self, model_name: str):
        self.nlp = spacy.load(
            model_name,
            disable=["tagger", "parser", "ner", "lemmatizer"]
        )
    def tokenize(self, text: str) -> List[str]:
        text = normalize_text(text)
        return [t.text for t in self.nlp(text)]


class MultilingualTokenizer:
    def __init__(self):
        self.tokenizers = {
            "en": SpacyTokenizer("en_core_web_sm"),
            "de": SpacyTokenizer("de_core_news_sm"),
        }

    def tokenize(self, text: str, lang: str) -> List[str]:
        if lang not in self.tokenizers:
            raise ValueError(f"Unsupported language: {lang}")
        return self.tokenizers[lang].tokenize(text)


 
# Multilingual Vocab Builder
 

def build_multilingual_vocab(
    data_by_lang: Dict[str, List[str]],
    min_freq: int = 1
) -> Dict[str, Dict[str, int]]:
    """
    data_by_lang = {
        "en": [text1, text2, ...],
        "de": [text1, text2, ...]
    }
    """
    tokenizer = MultilingualTokenizer()
    vocabs = {}

    for lang, texts in data_by_lang.items():
        tokenized = [tokenizer.tokenize(t, lang) for t in texts]
        vocabs[lang] = build_vocab(tokenized, min_freq=min_freq)

    return vocabs
