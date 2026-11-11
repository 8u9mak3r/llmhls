import torch
from torch import nn
from torch.nn import functional as F
from .custom_ops import rope, gqa


# def _rotate_half(x):
#     x1, x2 = x.chunk(2, dim=-1)
#     return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rope_theta):
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, query, key):
        # positions = torch.arange(
        #     query.shape[-2], device=query.device, dtype=torch.float32
        # )
        # frequencies = torch.outer(positions, self.inv_freq)
        # embedding = torch.cat((frequencies, frequencies), dim=-1)
        # cos = embedding.cos().to(query.dtype)[None, None, :, :]
        # sin = embedding.sin().to(query.dtype)[None, None, :, :]
        # query = query * cos + _rotate_half(query) * sin
        # key = key * cos + _rotate_half(key) * sin
        # return query, key
        return rope(query, key, self.inv_freq)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_attention_heads,
        num_key_value_heads,
        rope_theta,
        attention_bias,
        dtype,
    ):
        super().__init__()
        assert d_model % num_attention_heads == 0
        assert num_attention_heads % num_key_value_heads == 0
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = d_model // num_attention_heads
        self.q_proj = nn.Linear(
            d_model, d_model, bias=attention_bias, dtype=dtype
        )
        self.k_proj = nn.Linear(
            d_model,
            num_key_value_heads * self.head_dim,
            bias=attention_bias,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            d_model,
            num_key_value_heads * self.head_dim,
            bias=attention_bias,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.rotary = RotaryEmbedding(self.head_dim, rope_theta)

    def forward(self, x):
        batch_size, sequence_length, d_model = x.shape
        query = self.q_proj(x).view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )
        key = self.k_proj(x).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        value = self.v_proj(x).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.rotary(query, key)
        # output = F.scaled_dot_product_attention(
        #     query,
        #     key,
        #     value,
        #     dropout_p=0.0,
        #     is_causal=True,
        #     enable_gqa=True,
        # )
        output = gqa(query, key, value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, d_model
        )
        return self.o_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, d_model, intermediate_size, dtype):
        super().__init__()
        self.gate_proj = nn.Linear(
            d_model, intermediate_size, bias=False, dtype=dtype
        )
        self.up_proj = nn.Linear(
            d_model, intermediate_size, bias=False, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_size, d_model, bias=False, dtype=dtype
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        rope_theta,
        rms_norm_eps,
        attention_bias,
        dtype,
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(
            d_model, eps=rms_norm_eps, dtype=dtype
        )
        self.self_attn = GroupedQueryAttention(
            d_model,
            num_attention_heads,
            num_key_value_heads,
            rope_theta,
            attention_bias,
            dtype,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            d_model, eps=rms_norm_eps, dtype=dtype
        )
        self.mlp = SwiGLU(d_model, intermediate_size, dtype)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))
