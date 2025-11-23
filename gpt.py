import torch
import torch.nn as nn
from einops import rearrange, repeat

from config import Config


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4, f'x.ndim != 4'
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x2 * cos - x1 * sin
    out = torch.cat((y1, y2), dim = -1)
    out = out.to(dtype = x.dtype)
    return out


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5, device = None):
        super().__init__()
        # base param
        self.dim = dim
        self.eps = eps
        # base module
        self.weight = nn.Parameter(torch.ones(dim, device = device, dtype = torch.float32))

    def forward(self, x):
        # x:[b, n, d]
        assert x.shape[-1] == self.dim, f'x.shape[-1] != self.dim'
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t ** 2, dim = -1, keepdim = True) + self.eps)
        return (t * self.weight).to(dtype)


class Attention(nn.Module):

    def __init__(self, layer_idx, config: Config):
        super().__init__()
        # base param
        self.layer_idx = layer_idx
        self.is_flash = config.is_flash
        self.dim = config.text_dim
        self.n_heads = config.text_n_heads
        self.n_kv_heads = config.text_n_kv_heads
        # assert self.n_heads % self.n_kv_heads == 0, 'n_heads must be divisible by n_kv_heads'
        self.head_dim = self.dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = config.text_dropout_rate
        # base module
        self.to_q = nn.Linear(self.dim, self.n_heads * self.head_dim, bias = False)
        self.to_kv = nn.Linear(self.dim, 2 * self.n_kv_heads * self.head_dim, bias = False)
        self.to_out = nn.Linear(self.n_heads * self.head_dim, self.dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, cos_sin, attn_mask, kv_cache = None):
        # x: [b, n, d]
        # attn_mask: [b, n, n]
        q = self.to_q(x)
        kv = self.to_kv(x).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', d = self.head_dim), (q, kv[0], kv[1]))

        # rotary embedding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # attn
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d'), (q, k, v))

        # kv_cache
        if kv_cache is not None:
            if self.layer_idx not in kv_cache:
                kv_cache[self.layer_idx] = {
                    'k': k,
                    'v': v
                }
            else:
                old_k, old_v = kv_cache[self.layer_idx]['k'], kv_cache[self.layer_idx]['v']
                k = torch.cat((old_k, k), dim = -2)
                v = torch.cat((old_v, v), dim = -2)
                kv_cache[self.layer_idx] = {
                    'k': k,
                    'v': v
                }

        enable_gqa = self.n_heads != self.n_kv_heads
        if self.is_flash:
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(1)
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = attn_mask, is_causal = False,
                                                             dropout_p = self.dropout_rate, enable_gqa = enable_gqa)
        else:
            if enable_gqa:
                k = repeat(k, 'b h n d -> b (r h) n d', r = self.n_heads // self.n_kv_heads)
                v = repeat(v, 'b h n d -> b (r h) n d', r = self.n_heads // self.n_kv_heads)
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(1).logical_not()
                attn = attn + attn_mask.float() * -1e9
            attn = torch.softmax(attn, dim = -1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class FFN(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        # base param
        self.dim = config.text_dim
        self.dropout_rate = config.text_dropout_rate
        # base module
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dim * 4, self.dim),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        # x:[b, n, d]
        out = self.ffn(x)
        return out


class Transformer(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        # base param
        self.dim = config.text_dim
        self.n_layers = config.text_n_layers
        # base module
        self.layers = nn.ModuleList([
            nn.ModuleList([
                RMSNorm(self.dim),
                Attention(i, config),
                RMSNorm(self.dim),
                FFN(config)
            ]) for i in range(self.n_layers)
        ])

    def forward(self, x, cos_sin, attn_mask, kv_cache = None):
        # x:[b, n, d]
        for attn_norm, attn, ffn_norm, ffn in self.layers:
            x = x + attn(attn_norm(x), cos_sin, attn_mask, kv_cache)
            x = x + ffn(ffn_norm(x))
        return x


class ThinkChat(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        # base param
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.dim = config.text_dim
        self.n_heads = config.text_n_heads
        # assert self.dim % self.n_heads == 0, 'dim must be divisible by n_heads'
        self.head_dim = self.dim // self.n_heads
        # base module
        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.cos, self.sin = self._precompute_rotary_embedding(self.max_seq_len * 4, self.head_dim)
        self.transformer = Transformer(config)
        self.to_out = nn.Linear(self.dim, self.vocab_size)

    def _precompute_rotary_embedding(self, seq_len, head_dim, base = 10000, yarn_alpha = 0.4, yarn_m_scale = 0.6):
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype = torch.float32) / head_dim))

        t = torch.arange(seq_len, dtype = torch.float32)
        m = int(seq_len * yarn_m_scale) # 临界点
        t_mapped = t.clone()
        # 长区间部分缩放
        long_pos = t > m
        t_mapped[long_pos] = m + (t[long_pos] - m).pow(yarn_alpha)

        freqs = torch.outer(t_mapped, freqs)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, text, start_pos = 0, attn_mask = None, kv_cache = None):
        # text:[b, n]
        x = self.token_embedding(text)

        n = x.shape[1]
        cos_sin = self.cos[:, start_pos: start_pos + n].to(x.device), self.sin[:, start_pos: start_pos + n].to(x.device)
        if kv_cache is None or n != 1:
            causal_mask = torch.tril(torch.ones(n, n, dtype = torch.bool, device = attn_mask.device))
            causal_mask = causal_mask[None, :, :] # [1, n, n]
            attn_mask = attn_mask[:, None, :].bool() # [b, 1, n]
            attn_mask = attn_mask & attn_mask.transpose(1, 2) # [b, n, n]
            attn_mask = causal_mask & attn_mask
        else:
            attn_mask = None
        x = self.transformer(x, cos_sin, attn_mask, kv_cache)

        out = self.to_out(x)
        return out


if __name__ == "__main__":
    config = Config()
    config.is_flash = False
    model = GPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    # print(model)

    data = torch.randint(0, 12000, (2, 8))
    attn_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 1, 0, 0, 0]])
    print(model(data, start_pos = 0, attn_mask = attn_mask, kv_cache = None).shape)
