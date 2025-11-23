from dataclasses import dataclass


# Think-Chat-Small: 46952160(47M)
# batch_size = 16, base_lr = 1e-3, warmup_ratio = 0.05, grad_clip = 1.0, accumulation_steps = 2
# total_step = 100000 * accumulation_steps, warmup_step = int(warmup_ratio * total_step)
class Config:
    # common
    eps: float = 1e-5
    is_flash: bool = True
    # text_dim
    vocab_size: int = 12000
    max_seq_len: int = 512
    text_dim: int = 512
    text_n_heads: int = 8
    text_n_kv_heads: int = 4
    text_n_layers: int = 12
    text_dropout_rate: float = 0.05


# Think-Chat:
# class Config:
#     # common
#     eps: float = 1e-5
#     is_flash: bool = True
#     # text_dim
#     vocab_size: int = 12000
#     max_seq_len: int = 512
#     text_dim: int = 896
#     text_n_heads: int = 14
#     text_n_kv_heads: int = 2
#     text_n_layers: int = 16
#     text_dropout_rate: float = 0.1