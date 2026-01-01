# =========================================================
# Stage 9: Chatbot App (CLI) – CPU Safe Version
# =========================================================

import torch
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "t5_chatbot" / "final_model"

assert MODEL_PATH.exists(), "❌ Model path does not exist"

device = "cpu"
print("Using device:", device)

print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

def chat(user_input: str) -> str:
    input_text = "chat: " + user_input.strip()

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n🤖 Chatbot is ready! Type 'exit' to quit.\n")

while True:
    user_text = input("You: ")
    if user_text.lower() in {"exit", "quit"}:
        print("Bye 👋")
        break

    print("Bot:", chat(user_text))