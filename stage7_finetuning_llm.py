# =========================================================
# Stage 7: Fine-tuning LLM (T5) for Chatbot
# =========================================================

import re
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)

# ---------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "DailyDialog" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "t5_chatbot" / "final_model"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "t5-small"

MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64
BATCH_SIZE = 8
EPOCHS = 3

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)
print("Dataset loaded")

# ---------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------

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
        pairs.append({
            "input": "chat: " + utterances[i],
            "target": utterances[i + 1]
        })

pairs = pairs[:5000]   # subset for speed
print("Total pairs:", len(pairs))

# ---------------------------------------------------------
# Tokenizer & Model
# ---------------------------------------------------------

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

# ---------------------------------------------------------
# Custom Dataset
# ---------------------------------------------------------

class ChatDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source = self.pairs[idx]["input"]
        target = self.pairs[idx]["target"]

        source_enc = self.tokenizer(
            source,
            max_length=MAX_INPUT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_enc = self.tokenizer(
            target,
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_enc["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

dataset = ChatDataset(pairs, tokenizer)

# ---------------------------------------------------------
# Training Arguments
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=3e-4,
    logging_steps=100,
    save_strategy="no",
    report_to="none"
)

# ---------------------------------------------------------
# Trainer
# ---------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("Fine-tuning started...")
trainer.train()
print("Fine-tuning finished ✅")

# ---------------------------------------------------------
# Save FINAL model & tokenizer
# ---------------------------------------------------------

print("Saving model & tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Saved to:", OUTPUT_DIR)
print("Stage 7 DONE ✅")