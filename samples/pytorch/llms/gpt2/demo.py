import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig

quant_config = Int8WeightOnlyConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2",)
model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="cpu"
).eval()
text = "Who are you?"
encoded_input = tokenizer(text, return_tensors='pt')
output = model.generate(**encoded_input, max_new_tokens=40)

decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded)
