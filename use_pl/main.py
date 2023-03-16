import os
import json
from multiprocessing import freeze_support
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from data import load_dataset
from model import BertLinear, PlModel, BertDropout2d, BertLSTM, BertCNN, BertLastHiddenState, BertLinearMix, BertCNN2D, BertDPCNN


def main():

    # 1. 加载数据
    train_file = r"..\data\train.csv"
    dev_file = r"..\data\dev.csv"
    label_file = r"..\data\label.json"
    # bert_path = "bert-base-chinese"
    bert_path = r"D:\code\pretrain_model_dir\structbert-large"
    max_length = 64
    batch_size = 64

    train_dataloader, dev_dataloader, label2id, id2label, tokenizer = load_dataset(
        train_file, dev_file, label_file, bert_path, max_length, batch_size
    )

    # 2. 定义模型
    pl.seed_everything(32)
    model = BertLinear(bert_path, len(label2id))
    # model = BertDropout2d(bert_path, len(label2id))
    # model = BertLSTM(bert_path, len(label2id))
    # model = BertCNN(bert_path, len(label2id))
    # model = BertLastHiddenState(bert_path, len(label2id))
    # model = BertLinearMix(bert_path, len(label2id))
    # model = BertCNN2D(bert_path, len(label2id))
    # model = BertDPCNN(bert_path, len(label2id))
    pl_model = PlModel(model, id2label)

    # 打印下模型的参数量, 以 M 为单位
    print("模型参数量: ", sum(p.numel() for p in pl_model.parameters() if p.requires_grad) / 1000 / 1000, "M")

    # 3.训练器
    # 先做小批量的验证
    limit_batches = 1.0
    ddp = DDPStrategy(process_group_backend="gloo" if os.name == "nt" else "nccl")
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy=ddp,
        precision=16,
        max_epochs=3,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        # profiler="simple",  # 用于分析训练过程中的性能瓶颈
        callbacks=[ModelSummary(max_depth=2)],
    )
    trainer.fit(pl_model, train_dataloader, dev_dataloader)
    trainer.save_checkpoint("best_model.ckpt")


if __name__ == "__main__":
    freeze_support()
    main()
