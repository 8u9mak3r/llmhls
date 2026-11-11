import torch


num_layers = 32
d_model = 960
intermediate_size = 2560
num_attention_heads = 15
num_key_value_heads = 5
head_dim = d_model // num_attention_heads
vocab_size = 49152
max_position_embeddings = 8192
rope_theta = 100000.0
rms_norm_eps = 1e-5
attention_bias = False
tie_word_embeddings = True

batch_size = 1
sequence_length = 16
dtype = torch.float16
