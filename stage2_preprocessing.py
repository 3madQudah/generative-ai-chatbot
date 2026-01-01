# =========================================================
# NLP Pipeline with Flags
# Stage 2: Preprocessing & Tokenization
# Stage 3: Embeddings
# =========================================================

import re
import pandas as pd
from pathlib import Path
from nltk.tokenize import TreebankWordTokenizer
import spacy
from transformers import BertTokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec

# ---------------------------------------------------------
# FLAGS (تحكم بالتشغيل)
# ---------------------------------------------------------

RUN_PREPROCESSING = True
RUN_TOKENIZATION = True
RUN_ONEHOT = True
RUN_BOW = True
RUN_WORD2VEC = True

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

DATA_PATH = Path("/Users/mac/Desktop/GenAI_LLM_ChatBot/data/DailyDialog/train.csv")
MAX_UTTERANCES = 5000

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)
print("Dataset loaded")
print("Columns:", df.columns.tolist())

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

def build_pairs():
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
            pairs.append({
                "input": utterances[i],
                "response": utterances[i + 1]
            })

    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def run_preprocessing():
    pairs_df = build_pairs()
    print("\n=== Preprocessing ===")
    print(pairs_df.head())
    print("Total pairs:", len(pairs_df))
    return pairs_df

# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def run_tokenization(pairs_df):
    tokenizer = TreebankWordTokenizer()
    spacy_nlp = spacy.load("en_core_web_sm")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    sample_text = pairs_df["input"].iloc[0]

    print("\n=== Tokenization ===")
    print("Sample text:")
    print(sample_text)

    print("\nNLTK tokens:")
    print(tokenizer.tokenize(sample_text))

    print("\nspaCy tokens:")
    print([tok.text for tok in spacy_nlp(sample_text)])

    print("\nBERT tokens:")
    print(bert_tokenizer.tokenize(sample_text))

    texts = pairs_df["input"].tolist()[:MAX_UTTERANCES]
    tokens_list = [tokenizer.tokenize(t) for t in texts]

    return tokens_list

# ---------------------------------------------------------
# One-Hot Encoding
# ---------------------------------------------------------

def run_onehot(tokens_list):
    print("\n=== One-Hot Encoding ===")
    sentences = [" ".join(tokens) for tokens in tokens_list[:10]]
    encoder = OneHotEncoder(sparse_output=False)
    one_hot = encoder.fit_transform([[s] for s in sentences])
    print("One-hot shape:", one_hot.shape)

# ---------------------------------------------------------
# Bag of Words
# ---------------------------------------------------------

def run_bow(tokens_list):
    print("\n=== Bag of Words ===")
    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform([" ".join(t) for t in tokens_list])
    print("BoW shape:", bow.shape)
    print("Vocabulary size:", len(vectorizer.vocabulary_))

# ---------------------------------------------------------
# Word2Vec
# ---------------------------------------------------------

def run_word2vec(tokens_list):
    print("\n=== Word2Vec (Skip-gram) ===")
    w2v = Word2Vec(
        sentences=tokens_list,
        vector_size=100,
        window=5,
        min_count=3,
        sg=1,
        workers=4
    )

    if "beer" in w2v.wv:
        print("Similar words to 'beer':")
        print(w2v.wv.most_similar("beer"))

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    pairs_df = None
    tokens_list = None

    if RUN_PREPROCESSING:
        pairs_df = run_preprocessing()

    if RUN_TOKENIZATION and pairs_df is not None:
        tokens_list = run_tokenization(pairs_df)

    if RUN_ONEHOT and tokens_list is not None:
        run_onehot(tokens_list)

    if RUN_BOW and tokens_list is not None:
        run_bow(tokens_list)

    if RUN_WORD2VEC and tokens_list is not None:
        run_word2vec(tokens_list)

    print("\nPipeline finished ✅")
