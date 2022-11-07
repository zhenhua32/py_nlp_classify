import os
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AutoTokenizer


def load_dataframe(file: str) -> pd.DataFrame:
    """从文件中加载训练数据
    无需预先处理 label

    Args:
        file (str): 文件路径

    Returns:
        pd.DataFrame: 返回的 df 有两列, 分别是 label 和 title
    """
    df = pd.read_csv(file, header=None, index_col=None, names=["label", "title"], sep="\t")
    print(f"加载数据, 从 {file} 文件中, 原始大小是 {df.shape}")
    df.dropna(axis=0, inplace=True)
    print(f"加载数据, 从 {file} 文件中, 清理后大小是 {df.shape}")
    return df


def load_label(file: str):
    """从文件中加载标签映射

    Args:
        file (str): 文件路径

    Returns:
        tuple: 返回 label2id 和 id2label
    """
    with open(file, "r", encoding="utf-8") as f:
        label2id: dict = json.load(f)
        id2label: dict = dict((v, k) for k, v in label2id.items())
    return label2id, id2label


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2id: dict, tokenizer: BertTokenizer, max_length: int = 64) -> None:
        super().__init__()
        self.df = df
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        title = self.df.iloc[index]["title"]
        label = self.df.iloc[index]["label"]

        encoded_input = self.tokenizer(
            title, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"].squeeze()
        attention_mask = encoded_input["attention_mask"].squeeze()
        # token_type_ids = encoded_input["token_type_ids"].squeeze()

        label_id = self.label2id[label]
        label_id = torch.tensor(label_id)  # 是个标量

        return input_ids, attention_mask, label_id


def load_dataset(train_file, dev_file, label_file, bert_path, max_length=64, batch_size=32):
    """
    获取训练集和验证集, 以及返回 label2id 和 id2label, 还有 tokenizer
    """
    train_file = r"D:\code\py_nlp_classify\data\train.csv"
    dev_file = r"D:\code\py_nlp_classify\data\dev.csv"
    label_file = r"D:\code\py_nlp_classify\data\label.json"
    bert_path = "bert-base-chinese"

    train_df = load_dataframe(train_file)
    dev_df = load_dataframe(dev_file)
    label2id, id2label = load_label(label_file)
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(bert_path)
    train_dataset = CustomDataset(train_df, label2id, tokenizer, max_length=max_length)
    dev_dataset = CustomDataset(dev_df, label2id, tokenizer, max_length=max_length)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=os.cpu_count())
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size, num_workers=os.cpu_count())

    return train_dataloader, dev_dataloader, label2id, id2label, tokenizer
