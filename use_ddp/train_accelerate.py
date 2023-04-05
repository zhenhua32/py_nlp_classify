"""
训练循环
"""

import os
import time

from accelerate import Accelerator, notebook_launcher
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from prettytable import PrettyTable
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_utils import BertTokenizer, CustomDataset, load_dataframe, load_label
from models import BertLinear

# 各种训练参数
train_file = "../data/train.csv"
dev_file = "../data/dev.csv"
label_file = "../data/label.json"
bert_path = r"D:\code\pretrain_model_dir\bert-base-chinese"
max_length = 64
batch_size = 256
num_labels = 15
epochs = 3
learning_rate = 2e-5
log_interval = 10


def get_data_loader():
    """
    加载数据集
    """
    train_df = load_dataframe(train_file)
    dev_df = load_dataframe(dev_file)
    label2id, id2label = load_label(label_file)

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    train_dataset = CustomDataset(train_df, label2id, tokenizer, max_length)
    dev_dataset = CustomDataset(dev_df, label2id, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        # pin_memory=True,
    )
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    return train_loader, dev_loader


def train(
    accelerator: Accelerator,
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    writer: SummaryWriter,
):
    """
    训练一轮
    """
    model.train()
    total_steps = len(train_loader)
    total_examples = len(train_loader.dataset)

    process_bar = tqdm(enumerate(train_loader), total=total_steps, disable=not accelerator.is_main_process)
    for step, batch in process_bar:
        # 1. 准备数据
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)
        # 2. 前向传播
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        # 3. 反向传播
        optimizer.zero_grad()
        # accelerator: 替换反向传播
        accelerator.backward(loss)
        optimizer.step()
        # 4. 记录日志
        if step % log_interval == 0 and accelerator.is_main_process:
            cur_stats = f"Train Epoch: {epoch} [{step * len(input_ids)}/{total_examples} ({100. * step / total_steps:.0f}%)] Loss: {loss.item():.6f}"
            process_bar.set_description(cur_stats)
            writer.add_scalar("train_loss", loss.item(), epoch * total_steps + step)


def test(
    accelerator: Accelerator,
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    writer: SummaryWriter,
):
    """
    评估模型
    """
    model.eval()
    test_loss = 0
    total_examples = len(test_loader.dataset)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, disable=not accelerator.is_main_process):
            # 1. 准备数据
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            # 2. 前向传播
            logits = model(input_ids, attention_mask)
            test_loss += criterion(logits, labels).item()
            # 3. 保存标签和预测结果
            all_labels.extend(labels.tolist())
            all_preds.extend(torch.argmax(logits, dim=1).tolist())

    if accelerator.is_main_process:
        # 打印下前 10 个预测结果
        print(all_labels[:10])
        print(all_preds[:10])
        test_loss /= total_examples
        test_accuracy = accuracy_score(all_labels, all_preds)
        test_recall = recall_score(all_labels, all_preds, average="micro")
        test_precision = precision_score(all_labels, all_preds, average="micro")
        test_f1 = f1_score(all_labels, all_preds, average="micro")
        # 打印下混淆矩阵, 结合 PrettyTable 可以更直观的看到预测结果
        print("混淆矩阵：")
        cm = confusion_matrix(all_labels, all_preds)
        table = PrettyTable([""] + [f"预测{i}" for i in range(num_labels)])
        for i in range(num_labels):
            table.add_row([f"实际{i}"] + [j for j in cm[i]])
        print(table)

        # 打印下分类报告
        print("分类报告：")
        print(classification_report(all_labels, all_preds))

        cur_stats = f"Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}, Precision: {test_precision:.4f}, F1: {test_f1:.4f}"
        tqdm.write(cur_stats)
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("test_accuracy", test_accuracy, epoch)
        writer.add_scalar("test_recall", test_recall, epoch)
        writer.add_scalar("test_precision", test_precision, epoch)
        writer.add_scalar("test_f1", test_f1, epoch)


def train_loop():
    torch.backends.cuda.matmul.allow_tf32 = True
    if os.name == "nt":
        # windows 需要提前初始化, 主要是要设置 backend
        dist.init_process_group(backend="gloo", init_method="tcp://localhost:12345", rank=0, world_size=1)
    # accelerator: 初始化实例
    accelerator = Accelerator(
        mixed_precision="fp16",
    )
    # accelerator: 将 device 替换成 accelerator.device. 另一种选择是移除所有的 to(device) 或者 cuda 操作
    device = accelerator.device

    model = BertLinear(bert_path, num_labels, load_pretrain=True).to(device)
    # linux 上可以用, 没试过
    if os.name != "nt":
        model = torch.compile(model)
    train_loader, test_loader = get_data_loader()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # accelerator: 通过 prepare() 方法转换下
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    os.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, epochs + 1):
        # accelerator: 替换反向传播
        train(accelerator, model, device, train_loader, optimizer, criterion, epoch, writer)
        if accelerator.is_main_process:
            test(accelerator, model, device, test_loader, criterion, epoch, writer)
        accelerator.wait_for_everyone()

    # model_path = os.path.join(os.getcwd(), "model")
    # if not os.path.exists(model_path):
    #     os.mkdir(model_path)
    # model_path = os.path.join(model_path, "model.pt")
    # torch.save(model.state_dict(), model_path)

    writer.close()


if __name__ == "__main__":
    notebook_launcher(train_loop, args=(), num_processes=1)
