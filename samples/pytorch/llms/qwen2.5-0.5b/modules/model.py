from torch import nn

from .layers import DecoderLayer


class Qwen25Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        d_model,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        rope_theta,
        rms_norm_eps,
        attention_bias,
        tie_word_embeddings,
        dtype,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    intermediate_size,
                    num_attention_heads,
                    num_key_value_heads,
                    rope_theta,
                    rms_norm_eps,
                    attention_bias,
                    dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model, eps=rms_norm_eps, dtype=dtype)
        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False, dtype=dtype
        )
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(self.norm(hidden_states))
