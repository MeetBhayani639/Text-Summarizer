from datasets import load_dataset
from transformers import AutoTokenizer
from src.config import Config

cfg = Config()

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    return tokenizer

def load_raw_dataset():
    # CNN/DailyMail dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    return dataset

def preprocess_dataset(dataset_dict, tokenizer):
    """
    dataset_dict: a Hugging Face DatasetDict with splits 'train', 'validation', 'test'
    """

    def preprocess_function(batch):
        inputs = batch["article"]
        targets = batch["highlights"]

        # T5 prefix
        if "t5" in cfg.model_name:
            inputs = [f"summarize: {doc}" for doc in inputs]

        model_inputs = tokenizer(
            inputs,
            max_length=cfg.max_input_length,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=cfg.max_target_length,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )
    return tokenized
