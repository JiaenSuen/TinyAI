import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from dataset.tokenizer import simple_tokenizer
from dataset.vocab import build_vocab
from dataset.dataset import get_dataset
from models._model_factory import build_model


def load_reuters_data():
    import nltk
    nltk.download('reuters', quiet=True)
    from nltk.corpus import reuters

    train_fileids = [fid for fid in reuters.fileids() if fid.startswith('training/')]
    test_fileids  = [fid for fid in reuters.fileids() if fid.startswith('test/')]

    def load_docs(fileids):
        data = []
        for fid in fileids:
            text = reuters.raw(fid).strip()
            cats = reuters.categories(fid)
            if cats:                                
                data.append({'review': text, 'sentiment': cats[0]})
        return pd.DataFrame(data)

    train_df = load_docs(train_fileids)
    test_df  = load_docs(test_fileids)

    all_labels = sorted(reuters.categories())
    label_to_id = {lab: idx for idx, lab in enumerate(all_labels)}
    
    train_df['sentiment'] = train_df['sentiment'].map(label_to_id)
    test_df['sentiment']  = test_df['sentiment'].map(label_to_id)

    num_classes = len(all_labels)
    print(f" Reuters load")
    print(f"   Train: {len(train_df)} , Test: {len(test_df)} , Classes: {num_classes}")

    return train_df, test_df, num_classes


def train_model(model_selection='textcnn', epochs=15, batch_size=64, max_len=256, lr=1e-3):
    
    train_df, test_df, num_classes = load_reuters_data()

    if model_selection == 'mlp':
        train_dataset = get_dataset("vector", train_df, max_features=10000)
        test_dataset  = get_dataset("vector", test_df,  max_features=10000)
        vocab = None
    else:
        vocab = build_vocab(train_df['review'].tolist())
        train_dataset = get_dataset("sequence", train_df, vocab=vocab, tokenizer=simple_tokenizer, max_len=max_len)
        test_dataset  = get_dataset("sequence", test_df,  vocab=vocab, tokenizer=simple_tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(model_selection, vocab_size=len(vocab) if vocab else None, num_classes=num_classes)
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


    print(f"\nFinal Test Result - {model_selection.upper()}")
    print(f"Test Acc  : {test_acc:.4f}")


if __name__ == "__main__":

    train_model(model_selection='textcnn', epochs=20, batch_size=128, lr=1e-3)