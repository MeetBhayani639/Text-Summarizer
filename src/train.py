import numpy as np

from src.config import Config
from src.dataset import load_raw_dataset, get_tokenizer, preprocess_dataset

import evaluate
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    set_seed,
)

cfg = Config()

def compute_metrics(eval_pred):
    rouge = evaluate.load("rouge")
    predictions, labels = eval_pred

    tokenizer = get_tokenizer()
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )
    return {k: v.mid.fmeasure for k, v in result.items()}

def main():
    set_seed(cfg.seed)

    tokenizer = get_tokenizer()
    raw_datasets = load_raw_dataset()

    # shuffle first
    raw_datasets = raw_datasets.shuffle(seed=cfg.seed)

    # preprocess all splits
    tokenized_datasets = preprocess_dataset(raw_datasets, tokenizer)

    # take subsets for faster training
    tokenized_train = tokenized_datasets["train"].select(range(20000))
    tokenized_val   = tokenized_datasets["validation"].select(range(1000))

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
    
        # basic training configuration
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.lr,
        weight_decay=0.01,
        do_train=True,
        do_eval=True,
    
        # simple logging
        logging_steps=100,
    )





    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    main()
