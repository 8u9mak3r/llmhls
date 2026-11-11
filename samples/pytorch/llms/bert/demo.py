import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset

# 1. 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载模型和分词器
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# 3. 加载数据集 (IMDB)
train_dataset = load_dataset("imdb", split="train[:2000]")  # 用2000条做demo

# 4. 数据预处理
def encode_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

encoded_train_dataset = train_dataset.map(encode_batch, batched=True)
encoded_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 5. DataLoader (batch_size=1)
train_loader = DataLoader(encoded_train_dataset, batch_size=1, shuffle=True)

# 6. 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 7. 训练循环
model.train()
epochs = 2  # demo 只跑1个epoch
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["label"].to(device)
        }
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Average Loss: {total_loss/len(train_loader):.4f}")

    
test_dataset = load_dataset("imdb", split="test[:2000]")  # 取100条做测试，避免太大
encoded_test_dataset = test_dataset.map(encode_batch, batched=True)
encoded_test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
test_loader = DataLoader(encoded_test_dataset, batch_size=1)


model = model.eval().half()
correct, total = 0, 0

with torch.no_grad():
    for batch in test_loader:
        # 每个 batch 内只有 1 个样本
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }
        labels = batch["label"].to(device)

        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy on {total} samples: {correct/total:.4f}")


from torchao.quantization import Int8WeightOnlyConfig, Int4WeightOnlyConfig
from torchao.dtypes import Int4CPULayout
import torchao.quantization as aoq
import torch.nn as nn


def filter_fn(module, name):
    blacklist = ["bert.pooler.dense", "classifier"]
    
    if isinstance(module, nn.Linear) and name not in blacklist:
        return True
    
    return False

device = torch.device("cpu")
model = model.to(device)
# quant_config = Int4WeightOnlyConfig(group_size=128, layout=Int4CPULayout())
quantizer = aoq.int4_weight_only(group_size=32, layout=Int4CPULayout())
aoq.quantize_(model.bert, quantizer, filter_fn=filter_fn)

model = model.eval()
correct, total = 0, 0

print(model)

with torch.no_grad():
    for batch in test_loader:
        # 每个 batch 内只有 1 个样本
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }
        labels = batch["label"].to(device)

        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy on {total} samples: {correct/total:.4f}")

