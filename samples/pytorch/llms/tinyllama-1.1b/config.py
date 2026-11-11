import torch


num_layers = 22
d_model = 2048
intermediate_size = 5632
num_attention_heads = 32
num_key_value_heads = 4
head_dim = d_model // num_attention_heads
vocab_size = 32000
max_position_embeddings = 2048
rope_theta = 10000.0
rms_norm_eps = 1e-5
attention_bias = False
tie_word_embeddings = False

batch_size = 1
sequence_length = 9
dtype = torch.float16
