"""
使用 transformers 进行文本分类, 使用 pytorch 模型
"""

import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# 定义模型
pretrain_model_name = "bert-base-chinese"
model = AutoModelForSequenceClassification.from_pretrained(pretrain_model_name, num_labels=15)

# 定义分词器
tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name)

# 从本地加载数据
# 标签名要命名为 labels
raw_dataset = load_dataset(
    "csv",
    data_files={"train": "../data/train.csv", "dev": "../data/dev.csv"},
    delimiter="\t",
    column_names=["str_label", "text"],
)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# 分词
tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

# 同样的, 将文本标签转换成整数
label_map = {
    "agriculture": 0,
    "house": 1,
    "game": 2,
    "sports": 3,
    "entertainment": 4,
    "story": 5,
    "car": 6,
    "culture": 7,
    "finance": 8,
    "edu": 9,
    "military": 10,
    "travel": 11,
    "stock": 12,
    "world": 13,
    "tech": 14,
}


def label_function(ex):
    ex["labels"] = label_map[ex["str_label"]]
    return ex


tokenized_datasets = tokenized_datasets.map(label_function, batched=False)

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
dev_dataset = tokenized_datasets["dev"].shuffle(seed=42)


# 加载评估器
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# 定义训练参数
training_args = TrainingArguments("test_trainer_pt", evaluation_strategy="epoch", report_to="none", save_total_limit=2,)

# 定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()
# 评估
trainer.evaluate()
# 保存模型
trainer.save_model("my_model")
