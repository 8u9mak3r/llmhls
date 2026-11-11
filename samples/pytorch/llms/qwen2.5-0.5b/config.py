import torch


num_layers = 24
d_model = 896
intermediate_size = 4864
num_attention_heads = 14
num_key_value_heads = 2
head_dim = d_model // num_attention_heads
vocab_size = 151936
max_position_embeddings = 32768
rope_theta = 1000000.0
rms_norm_eps = 1e-6
attention_bias = True
tie_word_embeddings = True

batch_size = 1
sequence_length = 9
dtype = torch.float16
