from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Very naive baseline: take first N sentences as "summary"
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

def lead_3_baseline(text, n_sentences=3):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:n_sentences])

if __name__ == "__main__":
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    example = dataset["test"][0]
    article = example["article"]
    reference = example["highlights"]

    summary = lead_3_baseline(article, 3)
    print("ARTICLE:\n", article[:1000], "...")
    print("\nREFERENCE SUMMARY:\n", reference)
    print("\nBASELINE SUMMARY:\n", summary)
