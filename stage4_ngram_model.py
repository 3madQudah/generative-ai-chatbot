# =========================================================
# Stage 4: N-gram Language Model
# Unigram | Bigram | Trigram
# =========================================================

import re
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

DATA_PATH = Path("/Users/mac/Desktop/GenAI_LLM_ChatBot/data/DailyDialog/train.csv")
MAX_SENTENCES = 5000
N = 3   # change to 1, 2, or 3

# ---------------------------------------------------------
# Load & Clean Data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)
tokenizer = TreebankWordTokenizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"']", "", text)
    return text.strip()

sentences = []

for dialog in df["dialog"]:
    if not isinstance(dialog, str):
        continue

    dialog = dialog.strip().lstrip("[").rstrip("]")
    utterances = re.split(r"'\s*\n\s*'", dialog)

    for u in utterances:
        u = clean_text(u.replace("'", ""))
        if u:
            sentences.append(u)

sentences = sentences[:MAX_SENTENCES]

print("Total sentences:", len(sentences))

# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

tokenized_sentences = [tokenizer.tokenize(s) for s in sentences]

# ---------------------------------------------------------
# Build N-grams
# ---------------------------------------------------------

def build_ngrams(tokens_list, n):
    ngrams = defaultdict(list)

    for tokens in tokens_list:
        if len(tokens) < n:
            continue

        for i in range(len(tokens) - n + 1):
            context = tuple(tokens[i:i + n - 1])
            target = tokens[i + n - 1]
            ngrams[context].append(target)

    return ngrams

ngrams = build_ngrams(tokenized_sentences, N)
print(f"{N}-gram model built")
print("Number of contexts:", len(ngrams))

# ---------------------------------------------------------
# Text Generation
# ---------------------------------------------------------

def generate_text(ngrams, n, seed, max_len=20):
    result = list(seed)

    for _ in range(max_len):
        context = tuple(result[-(n - 1):]) if n > 1 else tuple()

        if context not in ngrams:
            break

        next_word = random.choice(ngrams[context])
        result.append(next_word)

    return " ".join(result)

# ---------------------------------------------------------
# Demo Generation
# ---------------------------------------------------------

if N == 1:
    seed = []
elif N == 2:
    seed = ["i"]
else:
    seed = ["i", "am"]

print("\nGenerated text:")
print(generate_text(ngrams, N, seed))
