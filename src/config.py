import os

class Config:
    model_name = "t5-small"  # or "facebook/bart-base"
    max_input_length = 512
    max_target_length = 128
    batch_size = 4
    num_train_epochs = 2
    lr = 5e-5
    output_dir = os.path.join("models", "t5-small-cnn")
    seed = 42
