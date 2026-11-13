import torch
from torch import Tensor
from torch.nn import functional as F


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.library.custom_op("llmhls::rope", mutates_args=())
def rope(
    query: Tensor,
    key: Tensor,
    inv_freq: Tensor,
) -> tuple[Tensor, Tensor]:
    positions = torch.arange(
        query.shape[-2],
        device=query.device,
        dtype=torch.float32,
    )
    frequencies = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().to(query.dtype)[None, None, :, :]
    sin = embedding.sin().to(query.dtype)[None, None, :, :]

    rotated_query = query * cos + _rotate_half(query) * sin
    rotated_key = key * cos + _rotate_half(key) * sin
    return rotated_query, rotated_key


@rope.register_fake
def _rope_fake(query, key, inv_freq):
    return (
        query.new_empty(query.shape),
        key.new_empty(key.shape),
    )


@torch.library.custom_op("llmhls::gqa", mutates_args=())
def gqa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> Tensor:
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
        enable_gqa=True,
    )


@gqa.register_fake
def _gqa_fake(query, key, value):
    output_shape = (*query.shape[:-1], value.shape[-1])
    return query.new_empty(output_shape)
