# Machine Translation (English to German) , Sequence-to-Sequence
## Multi 30k Machine Translation Dataset
English to German
https://www.kaggle.com/datasets/hemanthkumar21/multi30k-de-en

## Seq2Seq
```
src_text ──→ src_ids ──→ Encoder  
 → Decoder ──→ tgt_ids ──→ tgt_text
tgt_text ──→ tgt_ids ──┘
```

## Seq2Seq Engine Test
"Foxes are adorable creatures."
```
[Encoder_en2de] Epoch 1, Loss=3.4927
[Encoder_en2de] Epoch 2, Loss=3.4673
[Encoder_en2de] Epoch 3, Loss=3.4479
[Encoder_en2de] Epoch 4, Loss=3.4281
[Encoder_en2de] Epoch 5, Loss=3.4116
[Encoder_en2de] Eval Loss=3.3885
<UNK> <UNK> Geschöpfe <UNK> <UNK> Löffelhund Midtown sind sind sind sind Graufüchse Graufüchse Allesfresser <UNK> <UNK> Löffelhund Midtown sind sind sind sind Graufüchse Graufüchse Allesfresser <UNK> <UNK> Löffelhund Midtown sind sind sind sind Graufüchse Graufüchse Allesfresser <UNK> <UNK> Löffelhund Midtown sind sind sind sind Graufüchse Graufüchse
```