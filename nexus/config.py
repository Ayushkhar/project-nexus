MODEL_CONFIG = {
    "vocab_size": 50257,        # GPT-2 tokenizer vocab
    "context_length": 256,      # max tokens model can see
    "emb_dim": 768,             # embedding dimension
    "n_heads": 12,              # attention heads
    "n_layers": 12,             # transformer blocks
    "drop_rate": 0.1,           # dropout
    "qkv_bias": False           # attention bias
}

TRAIN_CONFIG = {
    "batch_size": 8,
    "learning_rate": 3e-4,
    "num_epochs": 10,
    "eval_freq": 100,
    "eval_iter": 10
}

GENERATION_CONFIG = {
    "max_new_tokens": 50,
    "temperature": 0.7,
    "top_k": 40
}