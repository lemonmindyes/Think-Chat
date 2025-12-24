from dataclasses import dataclass


'''
Think-Chat-Mini: 31997952(32M)
=================================
pretrain(Mini):
train_lines: 32 * 8 * 20000 = 5120000 ≈ (3.84 epoch)
batch_size = 32, base_lr = 3e-4, weight_decay = 0.1, grad_clip = 1.0
update_step = 20000, accumulation_steps = 8, warmup_ratio = 0.03
total_step = update_step * accumulation_steps = 160000
warmup_step = int(warmup_ratio * total_step) = 4800
=================================
sft(Nano):
train_lines: 32 * 1 * 10000 = 320000 ≈ (9.72 epoch)
batch_size = 32, base_lr = 2e-5, weight_decay = 0.1, grad_clip = 1.0
update_step = 10000, accumulation_steps = 1, warmup_ratio = 0.005
total_step = update_step * accumulation_steps = 10000
warmup_step = int(warmup_ratio * total_step) = 50
'''
# @dataclass
# class Config:
#     # common
#     eps: float = 1e-6
#     is_flash: bool = True
#     use_yarn: bool = False
#     # text_dim
#     vocab_size: int = 8192
#     max_seq_len: int = 512
#     text_dim: int = 512
#     text_n_heads: int = 8
#     text_n_kv_heads: int = 4
#     text_n_layers: int = 8
#     text_dropout_rate: float = 0.0


'''
Think-Chat-Medium: 90466304(90M)
=================================
pretrain(Mini):
train_lines: 32 * 8 * 20000 = 5120000 ≈ (3.84 epoch)
batch_size = 32, base_lr = 3e-4, weight_decay = 0.1, grad_clip = 1.0
update_step = 20000, accumulation_steps = 8, warmup_ratio = 0.03
total_step = update_step * accumulation_steps = 160000
warmup_step = int(warmup_ratio * total_step) = 4800
=================================
'''
@dataclass
class Config:
    # common
    eps: float = 1e-6
    is_flash: bool = True
    use_yarn: bool = False
    # text_dim
    vocab_size: int = 8192
    max_seq_len: int = 512
    text_dim: int = 768
    text_n_heads: int = 12
    text_n_kv_heads: int = 6
    text_n_layers: int = 12
    text_dropout_rate: float = 0.0


# Think-Chat:
# =============================================================================================
# =============================================================================================
# @dataclass
# class Config:
#     # common
#     eps: float = 1e-6
#     is_flash: bool = True
#     use_yarn: bool = False
#     # text_dim
#     vocab_size: int = 8192
#     max_seq_len: int = 512
#     text_dim: int = 896
#     text_n_heads: int = 14
#     text_n_kv_heads: int = 7
#     text_n_layers: int = 16
#     text_dropout_rate: float = 0.0