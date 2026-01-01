# =========================================================
# Stage 3: PyTorch NLP Pipeline
# Custom Dataset | Padding | Attention Masks | DataLoader
# =========================================================

import re
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import TreebankWordTokenizer

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

DATA_PATH = Path("/Users/mac/Desktop/GenAI_LLM_ChatBot/data/DailyDialog/train.csv")
MAX_LEN = 20
BATCH_SIZE = 32

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)
print("Dataset loaded")
print("Columns:", df.columns.tolist())

tokenizer = TreebankWordTokenizer()

# ---------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"']", "", text)
    return text.strip()

# ---------------------------------------------------------
# Parse dialogs → (input, response)
# ---------------------------------------------------------

pairs = []

for dialog in df["dialog"]:
    if not isinstance(dialog, str):
        continue

    dialog = dialog.strip().lstrip("[").rstrip("]")
    utterances = re.split(r"'\s*\n\s*'", dialog)
    utterances = [
        clean_text(u.replace("'", ""))
        for u in utterances
        if u.strip()
    ]

    for i in range(len(utterances) - 1):
        pairs.append((utterances[i], utterances[i + 1]))

print("Total (input, response) pairs:", len(pairs))

# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

tokenized_inputs = [tokenizer.tokenize(p[0]) for p in pairs]
tokenized_responses = [tokenizer.tokenize(p[1]) for p in pairs]

# ---------------------------------------------------------
# Build Vocabulary
# ---------------------------------------------------------

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1
}

counter = Counter()
for sent in tokenized_inputs + tokenized_responses:
    counter.update(sent)

vocab = {word: idx + len(SPECIAL_TOKENS) for idx, word in enumerate(counter)}
vocab.update(SPECIAL_TOKENS)

print("Vocabulary size:", len(vocab))

# ---------------------------------------------------------
# Numericalization + Padding + Attention Mask
# ---------------------------------------------------------

def encode(tokens):
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens][:MAX_LEN]
    attention_mask = [1] * len(ids)

    padding_length = MAX_LEN - len(ids)
    ids += [vocab["<PAD>"]] * padding_length
    attention_mask += [0] * padding_length

    return torch.tensor(ids), torch.tensor(attention_mask)

# ---------------------------------------------------------
# Custom PyTorch Dataset
# ---------------------------------------------------------

class ChatDataset(Dataset):
    def __init__(self, inputs, responses):
        self.inputs = inputs
        self.responses = responses

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids, input_mask = encode(self.inputs[idx])
        response_ids, response_mask = encode(self.responses[idx])

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "response_ids": response_ids,
            "response_mask": response_mask
        }

# ---------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------

dataset = ChatDataset(tokenized_inputs, tokenized_responses)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ---------------------------------------------------------
# Test a Batch
# ---------------------------------------------------------

batch = next(iter(dataloader))

print("\nBatch check:")
print("Input IDs shape:     ", batch["input_ids"].shape)
print("Input mask shape:   ", batch["input_mask"].shape)
print("Response IDs shape: ", batch["response_ids"].shape)
print("Response mask shape:", batch["response_mask"].shape)

print("\nStage 4 DONE ✅")
