# engine.py
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import MultilingualTokenizer, text_to_ids, pad_sequences, PAD, UNK
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class TranslationDataset(Dataset):
    """Dataset for Seq2Seq translation tasks"""
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, src_lang, tgt_lang):
        assert len(src_texts) == len(tgt_texts)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = MultilingualTokenizer()

        self.src_ids = [text_to_ids(self.tokenizer.tokenize(t, src_lang), src_vocab) for t in src_texts]
        self.tgt_ids = [text_to_ids(self.tokenizer.tokenize(t, tgt_lang), tgt_vocab) for t in tgt_texts]

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx]

def seq2seq_collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequences(list(src_batch))
    tgt_batch = pad_sequences(list(tgt_batch))
    return src_batch, tgt_batch

class Seq2SeqEngine:
    def __init__(
        self,
        encoder_class,
        decoder_class,
        src_lang,
        tgt_lang,
        encoder_params={},
        decoder_params={},
        lr=1e-3,
        batch_size=32,
        device=None,
        load_model=True
    ):
        self.encoder_class = encoder_class
        self.decoder_class = decoder_class
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.direction = f"{src_lang}2{tgt_lang}"

        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = f"{encoder_class.__name__}_{self.direction}"
        self.param_dir = "params"
        self.param_path = f"{self.param_dir}/{self.model_name}.pt"
        self.vocab_path = f"{self.param_dir}/{self.model_name}_vocabs.json"
        os.makedirs(self.param_dir, exist_ok=True)

        self.model = None
        self.src_vocab = None
        self.tgt_vocab = None
        self.tokenizer = MultilingualTokenizer()

        if load_model and os.path.exists(self.param_path) and os.path.exists(self.vocab_path):
          self.load()

    # --- Model Builder ---
    def _build_model(self):
        encoder = self.encoder_class(input_size=len(self.src_vocab), **self.encoder_params)
        decoder = self.decoder_class(input_size=len(self.tgt_vocab),
                                     output_size=len(self.tgt_vocab),
                                     **self.decoder_params)
        from models import Seq2Seq   
        self.model = Seq2Seq(encoder, decoder).to(self.device)

    # --- Training ---
    def train(self, src_texts, tgt_texts, epochs=10):
        
        # --- Tokenize ---
        tokenized_src = [self.tokenizer.tokenize(t, self.src_lang) for t in src_texts]
        tokenized_tgt = [self.tokenizer.tokenize(t, self.tgt_lang) for t in tgt_texts]

        # --- Build vocab automatically ---
        self.src_vocab = {PAD:0, UNK:1}
        for tokens in tokenized_src:
            for t in tokens:
                if t not in self.src_vocab:
                    self.src_vocab[t] = len(self.src_vocab)

        self.tgt_vocab = {PAD:0, UNK:1, "<BOS>":2, "<EOS>":3}
        for tokens in tokenized_tgt:
            for t in tokens:
                if t not in self.tgt_vocab:
                    self.tgt_vocab[t] = len(self.tgt_vocab)

        # --- Dataset ---
        dataset = TranslationDataset(
            src_texts, tgt_texts, self.src_vocab, self.tgt_vocab, self.src_lang, self.tgt_lang
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=seq2seq_collate_fn)

        # --- Build Model ---
        self._build_model()

        criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_vocab[PAD])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for ep in range(epochs):
            total_loss = 0
            for src_ids, tgt_ids in dataloader:
                src_ids = src_ids.transpose(0,1).to(self.device)  # seq_len x batch
                tgt_ids = tgt_ids.transpose(0,1).to(self.device)  # seq_len x batch

                optimizer.zero_grad()
                outputs = self.model(src_ids, tgt_ids)
                loss = criterion(
                    outputs[1:].reshape(-1, outputs.shape[-1]),
                    tgt_ids[1:].reshape(-1)
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[{self.model_name}] Epoch {ep+1}, Loss={total_loss:.4f}")

        self.save()

    # --- Evaluation ---
    def evaluate(self, src_texts, tgt_texts):
        """
        - Loss (CrossEntropyLoss)
        - Token-level Accuracy
        - BLEU-score 
        """
        dataset = TranslationDataset(
            src_texts, tgt_texts,
            self.src_vocab, self.tgt_vocab,
            self.src_lang, self.tgt_lang
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=seq2seq_collate_fn)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_vocab[PAD])

        self.model.eval()
        total_loss = 0
        total_tokens = 0
        correct_tokens = 0
        bleu_scores = []

        id2token = {v: k for k, v in self.tgt_vocab.items()}

        smooth_fn = SmoothingFunction().method1   

        with torch.no_grad():
            for src_ids, tgt_ids in dataloader:
                src_ids = src_ids.transpose(0,1).to(self.device)
                tgt_ids = tgt_ids.transpose(0,1).to(self.device)

                outputs = self.model(src_ids, tgt_ids)
                loss = criterion(
                    outputs[1:].reshape(-1, outputs.shape[-1]),
                    tgt_ids[1:].reshape(-1)
                )
                total_loss += loss.item()

                # Token-level accuracy
                pred_tokens = outputs.argmax(-1)
                mask = tgt_ids[1:] != self.tgt_vocab[PAD]
                correct_tokens += (pred_tokens[1:][mask] == tgt_ids[1:][mask]).sum().item()
                total_tokens += mask.sum().item()

                # BLEU per sentence
                batch_size = tgt_ids.shape[1]
                for i in range(batch_size):
                    ref_ids = tgt_ids[:, i].cpu().tolist()
                    pred_ids_i = pred_tokens[:, i].cpu().tolist()

                    # Remove PAD、BOS、EOS
                    ref_tokens = [id2token.get(t, UNK) for t in ref_ids if t not in (self.tgt_vocab.get(PAD), self.tgt_vocab.get("<BOS>"), self.tgt_vocab.get("<EOS>"))]
                    pred_tokens_i = [id2token.get(t, UNK) for t in pred_ids_i if t not in (self.tgt_vocab.get(PAD), self.tgt_vocab.get("<BOS>"), self.tgt_vocab.get("<EOS>"))]

                    if len(ref_tokens) == 0:
                        continue
                    bleu = sentence_bleu([ref_tokens], pred_tokens_i, smoothing_function=smooth_fn)
                    bleu_scores.append(bleu)

        avg_loss = total_loss / len(dataloader)
        token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if len(bleu_scores) > 0 else 0.0

        print(f"[{self.model_name}] Eval Loss={avg_loss:.4f}, Token Accuracy={token_acc:.4f}, BLEU={avg_bleu:.4f}")
        return avg_loss, token_acc, avg_bleu
        


    # --- Translate single sentence ---
    def translate(self, text, max_len=50):
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        tokens = self.tokenizer.tokenize(text, self.src_lang)
        src_ids = text_to_ids(tokens, self.src_vocab)
        src_tensor = pad_sequences([src_ids]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            if not hasattr(self.model, "translate"):
                raise RuntimeError("Model must implement a `translate` method for inference.")
            pred_ids = self.model.translate(
                src_tensor, max_len=max_len,
                bos_id=self.tgt_vocab.get("<BOS>"),
                eos_id=self.tgt_vocab.get("<EOS>")
            )

        id2token = {v:k for k,v in self.tgt_vocab.items()}
        tokens = [id2token.get(i, UNK) for i in pred_ids if i not in {self.tgt_vocab.get(PAD), self.tgt_vocab.get("<BOS>"), self.tgt_vocab.get("<EOS>")}]
        return " ".join(tokens)

    # --- Save / Load ---
    def save(self):
        torch.save(self.model.state_dict(), self.param_path)
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump({"src_vocab":self.src_vocab, "tgt_vocab":self.tgt_vocab}, f, ensure_ascii=False)

    def load(self):
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocabs = json.load(f)
        self.src_vocab = vocabs["src_vocab"]
        self.tgt_vocab = vocabs["tgt_vocab"]
        self._build_model()
        self.model.load_state_dict(torch.load(self.param_path, map_location=self.device,weights_only=True))
        self.model.eval()
