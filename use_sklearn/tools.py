import time
import json

import pandas as pd

"""
工具函数
将不重要的函数都放在这里, 这次主要是学习下 sklearn 中的分类算法
"""


def timeit(func):
    """
    添加一个计时的装饰器
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("time cost: {:.4f}s".format(end - start))

    return wrapper


def load_dataset():
    """
    加载数据集
    df 有三个字段, 分别是 label, text, label_id
    """
    train_df = pd.read_csv("../data/train.csv", header=None, index_col=None, sep="\t", names=["label", "text"])
    dev_df = pd.read_csv("../data/dev.csv", header=None, index_col=None, sep="\t", names=["label", "text"])

    with open("../data/label.json", "r", encoding="utf-8") as f:
        label_dict = json.load(f)
    label2id = dict((k, v["label_id"]) for k, v in label_dict.items())

    train_df["label_id"] = train_df["label"].map(label2id)
    dev_df["label_id"] = dev_df["label"].map(label2id)

    return train_df, dev_df, label2id
