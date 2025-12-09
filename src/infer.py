from config import Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

cfg = Config()

class Summarizer:
    def __init__(self, model_dir=None):
        model_dir = model_dir or cfg.output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.model.device

    def summarize(self, text, max_new_tokens=128):
        if "t5" in cfg.model_name:
            text = "summarize: " + text

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_input_length,
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
            )
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return summary
