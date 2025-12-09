from config import Config
from dataset import load_raw_dataset, get_tokenizer, preprocess_dataset
from transformers import AutoModelForSeq2SeqLM
import evaluate
import numpy as np

cfg = Config()

def main():
    tokenizer = get_tokenizer()
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.output_dir)

    rouge = evaluate.load("rouge")

    raw_datasets = load_raw_dataset()
    small_test = raw_datasets["test"].shuffle(seed=cfg.seed).select(range(1000))
    tokenized_test = preprocess_dataset({"test": small_test}, tokenizer)["test"]

    # generate summaries
    preds = []
    refs = []

    for i in range(0, len(tokenized_test), 8):
        batch = tokenized_test[i:i+8]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        import torch
        input_ids = torch.tensor(input_ids).to(model.device)
        attention_mask = torch.tensor(attention_mask).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=cfg.max_target_length,
                num_beams=4,
            )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(decoded_preds)

    # references
    raw_refs = small_test["highlights"]
    refs.extend(raw_refs)

    results = rouge.compute(predictions=preds, references=refs)
    print("ROUGE results:", results)

if __name__ == "__main__":
    main()
