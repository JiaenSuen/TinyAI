import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader , Dataset, random_split

import numpy as np
import pandas as pd
import re

from tokenizer import simple_tokenizer
from vocab import build_vocab , tokens_to_ids
from IMDBdataset import IMDBDataset
from model import IMDBClassifier

df = pd.read_csv('IMDB Dataset.csv')
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print(df[['review', 'sentiment']].head())



texts = df['review'].tolist()
# build vocab
vocab = build_vocab(df['review'].tolist())

# build dataset
dataset = IMDBDataset(df, vocab, simple_tokenizer, max_len=100)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# build dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True )
test_dataloader  = DataLoader(test_dataset , batch_size=32, shuffle=False)

 

'''
for batch in dataloader:
    print(batch['input_ids'].shape)  # torch.Size([32, 100])
    print(batch['label'].shape)      # torch.Size([32])
    break
'''


vocab_size = len(vocab)
model = IMDBClassifier(vocab_size=vocab_size, embedding_dim=128, hidden_dim=128, num_classes=2)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


for epoch in range(20):
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        # Forward
        outputs = model(input_ids)   
        # Loss
        loss = criterion(outputs, labels)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

 
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)  
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(train_dataloader)
    acc = total_correct / total_samples

    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")


model.eval()
for batch in test_dataloader:
    input_ids = batch['input_ids'].to(device)
    labels = batch['label'].to(device)


    outputs = model(input_ids)   
    preds = torch.argmax(outputs, dim=1)  
    total_correct += (preds == labels).sum().item()
    total_samples += labels.size(0)

acc = total_correct / total_samples
print(f"Test Accuracy = {acc:.4f}")



torch.save(model.state_dict(), 'imdb_classifier.pth')
model = IMDBClassifier(vocab_size=vocab_size, embedding_dim=128, hidden_dim=128, num_classes=2)
model.load_state_dict(torch.load('imdb_classifier.pth'))
model.to(device)




def predict_sentiment(model, vocab, tokenizer, text, max_len=100, device='cpu'):
    model.eval()  
    text = re.sub(r'<.*?>', '', text)
    tokens = tokenizer(text)
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(token_ids) < max_len:
        token_ids += [vocab['<PAD>']] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)  # shape: [1, max_len]

    #  forward
    with torch.no_grad():
        outputs = model(input_ids)  # shape: [1, 2]

        # softmax  
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    #  output ...
    label_map = {0: "negative", 1: "positive"}
    pred_label = label_map[pred_class]

    return pred_label, confidence

text = "This movie is fantastic! I really loved it." #input()
label, prob = predict_sentiment(model, vocab, simple_tokenizer, text, max_len=100, device=device)
print(f"Preduct Result: {label}, Prob: {prob:.4f}")
