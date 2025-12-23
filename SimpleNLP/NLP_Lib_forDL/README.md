# NLP DL Engine & Utilities
This library provides a simple yet flexible pipeline for building NLP models with deep learning. It includes tokenizers, vocabulary handling, and a unified engine for training, testing, and predicting with popular PyTorch NLP models like LSTM, GRU, and CNN for text.

  ---

## NLP Function
Tokenization & Vocabulary

```
from utils import SpacyTokenizer, NLTKTokenizer, CharTokenizer
from utils import build_vocab, tokens_to_ids

texts = [
    "Foxes are adorable creatures.",
    "Red Foxes and Arctic Foxes are super great !!!"
]

# SpaCy Tokenizer
spacy_tok = SpacyTokenizer()
spacy_tokens = [spacy_tok.tokenize(t) for t in texts]

# NLTK Tokenizer
nltk_tok = NLTKTokenizer()
nltk_tokens = [nltk_tok.tokenize(t) for t in texts]

# Char-level Tokenizer
char_tok = CharTokenizer()
char_tokens = [char_tok.tokenize(t) for t in texts]

# Build vocabulary
vocab = build_vocab(spacy_tokens)

# Convert tokens to IDs
ids = tokens_to_ids(spacy_tokens[0], vocab) 
```


## NLP Deep Learning Engine
nlpDLEngine is a unified interface for training, testing, and predicting with different NLP models.  
```
from engine import nlpDLEngine
from models import LSTMModel, GRUModel, CNNTextModel

texts = [
    "Foxes are adorable creatures.",
    "Red Foxes and Arctic Foxes are super great !!!",
    "I hate boring animals.",
    "Cats are cute too!",
    "I dislike rainy days."
]

labels = ["positive", "positive", "negative", "positive", "negative"]
label2id = {"positive": 0, "negative": 1}
id2label = {0: "positive", 1: "negative"}

epochs = 5
batch_size = 2

for Model in [LSTMModel, GRUModel, CNNTextModel]:
    print(f"\n=== Testing {Model.__name__} ===")
    engine = nlpDLEngine(Model, batch_size=batch_size)

    # Train the model
    engine.train(texts, labels, label2id, epochs=epochs)

    # Evaluate accuracy
    engine.test(texts, labels, label2id)

    # Predict new sentence
    test_sentence = "Foxes are lovely!"
    pred_label = engine.predict(test_sentence, id2label)
    print(f"Predict for '{test_sentence}': {pred_label}")

```

---

## Features

Supports LSTM, GRU, CNN for text classification  
Simple tokenization (SpaCy, NLTK, Char-level)  
Automatic vocabulary building and token-to-ID conversion  
Train / Test / Predict interface with PyTorch  
Works with GPU if available  
Batch-friendly for small or large datasets  

---