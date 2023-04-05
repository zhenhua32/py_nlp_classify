"""
来吧, 先从数据集开始
"""

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
        label_dict = json.load(f)
        label2id: dict = {k: v["label_id"] for k, v in label_dict.items()}
        id2label: dict = dict((v, k) for k, v in label2id.items())
    return label2id, id2label


class CustomDataset(Dataset):
    """
    这次使用预先分词好的数据集
    """

    def __init__(self, df: pd.DataFrame, label2id: dict, tokenizer: BertTokenizer, max_length: int = 64) -> None:
        super().__init__()

        self.df = df
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 先 tokenizer
        self.data_dict = self.tokenizer(
            self.df["title"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        label = self.df.iloc[index]["label"]
        label_id = self.label2id[label]

        # 这里直接从 self.data_dict 中取数据
        input_ids = self.data_dict["input_ids"][index]
        attention_mask = self.data_dict["attention_mask"][index]
        token_type_ids = self.data_dict["token_type_ids"][index]

        # 返回的是字典, 所以用 DataLoader 包装后, 得到的也是字典
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": label_id,
        }


if __name__ == "__main__":
    # 测试下数据集, 在当前目录下运行, 而不是根目录下
    train_df = load_dataframe("../data/train.csv")
    label2id, id2label = load_label("../data/label.json")
    tokenizer = AutoTokenizer.from_pretrained(r"D:\code\pretrain_model_dir\bert-base-chinese")
    train_dataset = CustomDataset(train_df, label2id, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    print(len(train_dataset))
    print(train_dataset[0])
    print(len(train_dataloader))
    for batch in train_dataloader:
        print(batch)
        for k, v in batch.items():
            print(k, v.shape, v.dtype, v.device)
        break
