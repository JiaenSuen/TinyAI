from engine import Seq2SeqEngine
from models import LSTM_Encoder  , LSTM_Decoder , Seq2Seq
from engine import Seq2SeqEngine

import os
data_path_root = "data/Multi30k"

train_en_path = os.path.join(data_path_root, "train.en")
train_de_path = os.path.join(data_path_root, "train.de")
test_en_path  = os.path.join(data_path_root, "test.en")
test_de_path  = os.path.join(data_path_root, "test.de")

 
def read_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

data_en_train = read_lines(train_en_path)
data_de_train = read_lines(train_de_path)
data_en_test  = read_lines(test_en_path)
data_de_test  = read_lines(test_de_path)

print(f"Train samples: {len(data_en_train)}")
print(f"Test samples : {len(data_en_test)}")

engine = Seq2SeqEngine(
    encoder_class=LSTM_Encoder,
    decoder_class=LSTM_Decoder,
    src_lang="en",
    tgt_lang="de",
    encoder_params={"embedding_size":32, "hidden_size":64, "num_layers":1, "p":0.0},
    decoder_params={"embedding_size":32, "hidden_size":64, "num_layers":1, "p":0.0},
    load_model=True
)

engine.train(data_en_train, data_de_train, epochs=50)
engine.evaluate(data_en_test, data_de_test)
print(engine.translate("Foxes are adorable creatures."))
print(engine.translate("A fox terrier leaps after a ball."))
print(engine.translate("Two dogs play by a tree."))
