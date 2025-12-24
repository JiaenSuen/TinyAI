from utils import (
    MultilingualTokenizer,
    build_multilingual_vocab,
    text_to_ids
)

data_en = [
    "Foxes are adorable creatures."
]

data_de = [
    "Füchse sind entzückende Geschöpfe."
]

print("=== Tokenization ===")
tokenizer = MultilingualTokenizer()

tokens_en = [tokenizer.tokenize(t, "en") for t in data_en]
tokens_de = [tokenizer.tokenize(t, "de") for t in data_de]

print("EN:", tokens_en)
print("DE:", tokens_de)

print("\n=== Build Vocabulary ===")
vocabs = build_multilingual_vocab({
    "en": data_en,
    "de": data_de
})

print("EN vocab:", vocabs["en"])
print("DE vocab:", vocabs["de"])

print("\n=== Tokens → IDs ===")
ids_en = text_to_ids(tokens_en[0], vocabs["en"])
ids_de = text_to_ids(tokens_de[0], vocabs["de"])

print("EN ids:", ids_en)
print("DE ids:", ids_de)



'''

 
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


'''