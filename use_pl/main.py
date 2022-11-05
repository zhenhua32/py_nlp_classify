import os
import json
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import AutoTokenizer

from data import load_dataset
from model import BertLinear, PlModel

# 1. 加载数据
train_file = r"D:\code\py_nlp_classify\data\train.csv"
dev_file = r"D:\code\py_nlp_classify\data\dev.csv"
label_file = r"D:\code\py_nlp_classify\data\label.json"
bert_path = "bert-base-chinese"
max_length = 64

train_dataloader, dev_dataloader, label2id, id2label, tokenizer = load_dataset(
    train_file, dev_file, label_file, bert_path, max_length
)

# 2. 定义模型
model = BertLinear(bert_path, len(label2id))
pl_model = PlModel(model)

# 3.训练器
# 先做小批量的验证
limit_batches = 100
trainer = pl.Trainer(accelerator="gpu", devices=-1, max_epochs=3, limit_train_batches=limit_batches)
trainer.fit(pl_model, train_dataloader, dev_dataloader)
