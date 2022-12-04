import os
import time

import torch
import pytorch_lightning as pl
from transformers import BertTokenizer

from model import PlModel, BertLinearMix, AbstractModel
from data import load_label

"""
推理部分的代码
"""


def load_model(model_class: AbstractModel, label_file: str, ckpt_path: str) -> PlModel:
    label2id, id2label = load_label(label_file)
    bert_path = os.path.dirname(ckpt_path)
    model = model_class(bert_path, len(label2id), load_pretrain=False)
    pl_model = PlModel.load_from_checkpoint(ckpt_path, model=model, id2label=id2label)
    pl_model.init_predict(bert_path)

    return pl_model


def main():
    label_file = r"..\data\label.json"
    ckpt_path = "./model_dir/epoch=2-step=30.ckpt"
    pl_model = load_model(BertLinearMix, label_file, ckpt_path)

    query = "加快产城融合 以科技创新引领新城区建设 新城区,城镇化率,中心城区,科技新城,科技创新"

    start = time.time()
    # 推理的时候果然不占显存, 单次 1000 条也是可以的, 用了 6.7 秒多. CPU 用了 49.5 秒多
    result = pl_model.predict_raw([query] * 1000)
    print(time.time() - start)
    print(result[0])


if __name__ == "__main__":
    main()
