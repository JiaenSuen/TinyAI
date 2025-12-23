# test_demo.py
from utils import SpacyTokenizer, NLTKTokenizer, CharTokenizer
from utils import build_vocab, tokens_to_ids

texts = [
    "Foxes are adorable creatures.",
    "Red Foxes and Arctic Foxes are super great !!!"
]
 
print("=== SpaCy Tokenizer ===")
spacy_tok = SpacyTokenizer()
spacy_tokens = [spacy_tok.tokenize(t) for t in texts]
print(spacy_tokens)

print("\n=== NLTK Tokenizer ===")
nltk_tok = NLTKTokenizer()
nltk_tokens = [nltk_tok.tokenize(t) for t in texts]
print(nltk_tokens)

print("\n=== Char Tokenizer ===")
char_tok = CharTokenizer()
char_tokens = [char_tok.tokenize(t) for t in texts]
print(char_tokens)

print("\n=== Build Vocabulary ===")
vocab = build_vocab(spacy_tokens)
print(vocab)

print("\n=== Tokens → IDs ===")
ids = tokens_to_ids(spacy_tokens[0], vocab)
print(ids)







 
from engine import nlpDLEngine 
from models import LSTMModel, GRUModel, CNNTextModel


texts = [
    "Foxes are adorable creatures.",
    "Red Foxes and Arctic Foxes are super great !!!",
    "I hate boring animals.",
    "Cats are cute too!",
    "I dislike rainy days."
]

labels = [
    "positive",
    "positive",
    "negative",
    "positive",
    "negative"
]

label2id = {"positive": 0, "negative": 1}
id2label = {0: "positive", 1: "negative"}


epochs = 5
batch_size = 2  

for Model in [LSTMModel, GRUModel, CNNTextModel]:
    print(f"\n=== Testing {Model.__name__} ===")
    engine = nlpDLEngine(Model, batch_size=batch_size)
    
 
    engine.train(texts, labels, label2id, epochs=epochs)
    engine.test(texts, labels, label2id)
    
 
    test_sentence = "Foxes are lovely!"
    pred_label = engine.predict(test_sentence, id2label)
    print(f"Predict for '{test_sentence}': {pred_label}")