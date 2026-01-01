import pandas as pd
from pathlib import Path
import re

# Path to CSV dataset
DATA_PATH = Path("/Users/mac/Desktop/GenAI_LLM_ChatBot/data/DailyDialog/train.csv")

# Load CSV
df = pd.read_csv(DATA_PATH)

print("Columns:", df.columns)
print("\nSample rows:")
print(df.head())

pairs = []

for dialog in df["dialog"]:
    if not isinstance(dialog, str):
        continue

    # Remove brackets
    dialog = dialog.strip()
    dialog = dialog.lstrip("[").rstrip("]")

    # Split utterances (this dataset uses line breaks between quotes)
    utterances = re.split(r"'\s*\n\s*'", dialog)

    # Clean utterances
    utterances = [
        u.replace("'", "").strip()
        for u in utterances
        if u.replace("'", "").strip()
    ]

    for i in range(len(utterances) - 1):
        pairs.append({
            "input": utterances[i],
            "response": utterances[i + 1]
        })

pairs_df = pd.DataFrame(pairs)

print("\nSample chat pairs:")
print(pairs_df.head())
print("\nTotal chat pairs:", len(pairs_df))

# Safety check
if len(pairs_df) == 0:
    raise ValueError("No chat pairs were created — parsing failed.")

# Analysis
pairs_df["input_len"] = pairs_df["input"].str.split().apply(len)
pairs_df["response_len"] = pairs_df["response"].str.split().apply(len)

print("\nLength statistics:")
print(pairs_df[["input_len", "response_len"]].describe())