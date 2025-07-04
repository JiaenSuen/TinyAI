
# Vocab : Dict 
from tokenizer import simple_tokenizer
from collections import Counter

def build_vocab(texts):
    counter = Counter()
    for text in texts:
        tokens = simple_tokenizer(text)
        counter.update(tokens)

    max_vocab_size = 10000
    most_common = counter.most_common(max_vocab_size - 2)  # minus PAD、UNK

    # 建立 vocab dict
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, freq) in enumerate(most_common, start=2):
        vocab[word] = idx

    return vocab

def tokens_to_ids(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]





# [('<PAD>', 0), ('<UNK>', 1), ('the', 2), ('and', 3), ('a', 4),
#  ('of', 5), ('to', 6), ('is', 7), ('it', 8), ('in', 9)]