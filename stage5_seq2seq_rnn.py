# =========================================================
# Stage 5: Seq2Seq RNN Chatbot (Encoder–Decoder)
# =========================================================

import re
import random
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import TreebankWordTokenizer

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

DATA_PATH = Path("/Users/mac/Desktop/GenAI_LLM_ChatBot/data/DailyDialog/train.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 20
BATCH_SIZE = 32
EMBED_DIM = 128
HIDDEN_DIM = 256
EPOCHS = 5
TEACHER_FORCING_RATIO = 0.5

# ---------------------------------------------------------
# Load & preprocess data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)
tokenizer = TreebankWordTokenizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"']", "", text)
    return text.strip()

pairs = []

for dialog in df["dialog"]:
    if not isinstance(dialog, str):
        continue

    dialog = dialog.strip().lstrip("[").rstrip("]")
    utterances = re.split(r"'\s*\n\s*'", dialog)
    utterances = [clean_text(u.replace("'", "")) for u in utterances if u.strip()]

    for i in range(len(utterances) - 1):
        pairs.append((utterances[i], utterances[i + 1]))

pairs = pairs[:10000]  # subset for speed
print("Total pairs:", len(pairs))

# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

tokenized_inputs = [tokenizer.tokenize(p[0]) for p in pairs]
tokenized_outputs = [tokenizer.tokenize(p[1]) for p in pairs]

# ---------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<SOS>": 2,
    "<EOS>": 3
}

counter = Counter()
for sent in tokenized_inputs + tokenized_outputs:
    counter.update(sent)

vocab = {w: i + len(SPECIAL_TOKENS) for i, w in enumerate(counter)}
vocab.update(SPECIAL_TOKENS)

inv_vocab = {i: w for w, i in vocab.items()}
VOCAB_SIZE = len(vocab)

print("Vocab size:", VOCAB_SIZE)

# ---------------------------------------------------------
# Encoding + Padding
# ---------------------------------------------------------

def encode(tokens, add_sos=False, add_eos=False):
    ids = []

    if add_sos:
        ids.append(vocab["<SOS>"])

    ids += [vocab.get(t, vocab["<UNK>"]) for t in tokens]

    if add_eos:
        ids.append(vocab["<EOS>"])

    ids = ids[:MAX_LEN]
    ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))

    return torch.tensor(ids)

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class ChatDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src = encode(self.inputs[idx])
        tgt = encode(self.outputs[idx], add_sos=True, add_eos=True)
        return src, tgt

dataset = ChatDataset(tokenized_inputs, tokenized_outputs)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        _, (hidden, cell) = self.lstm(emb)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        emb = self.embedding(x)
        output, (hidden, cell) = self.lstm(emb, (hidden, cell))
        pred = self.fc(output.squeeze(1))
        return pred, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, tgt):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        outputs = torch.zeros(batch_size, tgt_len, VOCAB_SIZE).to(DEVICE)

        hidden, cell = self.encoder(src)

        x = tgt[:, 0]  # <SOS>

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t] = output
            teacher_force = random.random() < TEACHER_FORCING_RATIO
            x = tgt[:, t] if teacher_force else output.argmax(1)

        return outputs

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

model = Seq2Seq().to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training started...")

for epoch in range(EPOCHS):
    total_loss = 0

    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, tgt)

        loss = criterion(
            output[:, 1:].reshape(-1, VOCAB_SIZE),
            tgt[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

print("Training finished ✅")

# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

def chat(sentence):
    model.eval()
    tokens = tokenizer.tokenize(clean_text(sentence))
    src = encode(tokens).unsqueeze(0).to(DEVICE)

    hidden, cell = model.encoder(src)

    x = torch.tensor([vocab["<SOS>"]]).to(DEVICE)
    result = []

    for _ in range(MAX_LEN):
        output, hidden, cell = model.decoder(x, hidden, cell)
        token = output.argmax(1).item()

        if token == vocab["<EOS>"]:
            break

        result.append(inv_vocab.get(token, "<UNK>"))
        x = torch.tensor([token]).to(DEVICE)

    return " ".join(result)

# ---------------------------------------------------------
# Test Chat
# ---------------------------------------------------------

print("\nChatbot test:")
print("You: hello")
print("Bot:", chat("hello"))
