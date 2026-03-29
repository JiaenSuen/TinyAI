import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd

from dataset.tokenizer import simple_tokenizer
from dataset.vocab import build_vocab
from dataset.IMDBdataset import get_dataset
from models._model_factory import build_model



def train_model(model_selection='lstm', epochs=8, batch_size=64, max_len=256, lr=1e-3):
    
 
    df = pd.read_csv('dataset/IMDB Dataset.csv')
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})


    if model_selection == 'mlp':
        dataset = get_dataset("vector", df, max_features=10000)
        vocab = None
    else:
        vocab = build_vocab(df['review'].tolist())
        dataset = get_dataset("sequence", df, vocab=vocab, tokenizer=simple_tokenizer, max_len=max_len)
        

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





    model = build_model(model_selection, vocab_size=len(vocab) if vocab else None)
    model = model.to(device)








    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    records = []
    for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

            for batch in progress_bar:
                if model_selection == 'mlp':
                    features = batch['features'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(features)
                else:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(input_ids)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

   
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

      
                progress_bar.set_postfix({
                    'Loss': f'{total_loss/(progress_bar.n+1):.4f}',
                    'Acc': f'{total_correct/total_samples:.4f}'
                })

 
            train_loss = total_loss / len(train_loader)
            train_acc = total_correct / total_samples

          
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for batch in test_loader:
                    if model_selection == 'mlp':
                        features = batch['features'].to(device)
                        labels = batch['label'].to(device)
                        outputs = model(features)
                    else:
                        input_ids = batch['input_ids'].to(device)
                        labels = batch['label'].to(device)
                        outputs = model(input_ids)

                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    test_correct += (preds == labels).sum().item()
                    test_total += labels.size(0)

            test_loss = test_loss / len(test_loader)
            test_acc = test_correct / test_total

            print(f"Epoch {epoch+1:2d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

            records.append([epoch+1, train_loss, train_acc, test_loss, test_acc, model_selection])

    record_df = pd.DataFrame(records,
                         columns=['epoch','train_loss','train_acc','test_loss','test_acc','model'])

    record_df.to_csv(f'record/{model_selection}_record.csv', mode='a', header=not pd.io.common.file_exists(f'record/{model_selection}_record.csv'), index=False)


    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            if model_selection == 'mlp':
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                outputs = model(features)
            else:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    test_loss = test_loss / len(test_loader)

    print(f"\nFinal Test Result")
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc:.4f}")
 


if __name__ == "__main__":

    train_model(model_selection='textcnn', epochs=20, batch_size=128, lr=1e-3)
   