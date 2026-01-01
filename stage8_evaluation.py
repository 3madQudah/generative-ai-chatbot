# =========================================================
# Stage 8: Evaluation
# Metrics: Perplexity | BLEU | Qualitative Analysis
# Dataset: DailyDialog
# Model: T5-small
# =========================================================

import re
import math
import torch
import pandas as pd
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

DATA_PATH = Path("/Users/mac/Desktop/GenAI_LLM_ChatBot/data/DailyDialog/train.csv")
MODEL_NAME = "t5-small"
DEVICE = "cpu"          # مهم لتفادي مشاكل MPS
MAX_LEN = 64
EVAL_SAMPLES = 20       # خليها صغيرة للتجربة السريعة

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    return text.lower().replace("\n", " ").strip()

# ---------------------------------------------------------
# Load Model & Tokenizer
# ---------------------------------------------------------

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.eval()

print("Model and tokenizer loaded")

# ---------------------------------------------------------
# Load Dataset & Build (input, response) pairs
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

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

pairs = pairs[:EVAL_SAMPLES]

if len(pairs) == 0:
    raise ValueError("No evaluation samples loaded. Check dialog parsing.")

print(f"Loaded {len(pairs)} evaluation samples")

# ---------------------------------------------------------
# 1️⃣ Perplexity
# ---------------------------------------------------------

def compute_perplexity(model, tokenizer, texts):
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN
            )

            labels = enc["input_ids"]
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=labels
            )

            loss = outputs.loss
            num_tokens = labels.numel()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

sample_inputs = [p[0] for p in pairs]
ppl = compute_perplexity(model, tokenizer, sample_inputs)

print("\n=== Perplexity ===")
print("Perplexity:", round(ppl, 3))

# ---------------------------------------------------------
# 2️⃣ BLEU Score
# ---------------------------------------------------------

smooth = SmoothingFunction().method4

def generate_response(text):
    enc = tokenizer(
        "chat: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    )

    output = model.generate(
        **enc,
        max_length=60,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

references = []
predictions = []

for inp, ref in pairs:
    pred = generate_response(inp)
    references.append(ref)
    predictions.append(pred)

def compute_bleu(refs, preds):
    scores = []
    for r, p in zip(refs, preds):
        score = sentence_bleu(
            [r.split()],
            p.split(),
            smoothing_function=smooth
        )
        scores.append(score)
    return sum(scores) / len(scores)

bleu = compute_bleu(references, predictions)

print("\n=== BLEU Score ===")
print("BLEU:", round(bleu, 4))

# ---------------------------------------------------------
# 3️⃣ Qualitative Evaluation
# ---------------------------------------------------------

print("\n=== Qualitative Evaluation (Samples) ===")

for i in range(min(5, len(pairs))):
    print("\nInput:     ", pairs[i][0])
    print("Reference: ", pairs[i][1])
    print("Model:     ", predictions[i])

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n=== Evaluation Summary ===")
print(f"Perplexity: {round(ppl, 3)} (lower is better)")
print(f"BLEU Score: {round(bleu, 4)} (higher is better)")
print("Qualitative analysis completed (fluency, relevance, repetition)")
print("Stage 8 DONE ✅")