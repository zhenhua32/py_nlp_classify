"""
使用 transformers 进行文本分类, 使用 pytorch 模型
"""
import os

import numpy as np
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

# 定义标签映射
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
id2label = {v: k for k, v in label_map.items()}

# 定义模型
pretrain_model_name = "bert-base-chinese"
# 定义分词器
tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 从本地加载数据
# 标签名要命名为 labels
raw_dataset = load_dataset(
    "csv",
    data_files={"train": "../data/train.csv", "dev": "../data/dev.csv"},
    delimiter="\t",
    column_names=["str_label", "text"],
)


def tokenize_function(examples):
    # 不用在这里使用 padding="max_length", 因为后面会使用 DataCollatorWithPadding
    return tokenizer(examples["text"], truncation=True, max_length=64)


# 分词
tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)


def label_function(ex):
    # 每个标签都有 news_ 开头, 需要去掉
    ex["labels"] = label_map[ex["str_label"].replace("news_", "")]
    return ex


tokenized_datasets = tokenized_datasets.map(label_function, batched=False)
train_dataset = tokenized_datasets["train"].shuffle(seed=42)
dev_dataset = tokenized_datasets["dev"]
# 查看下第一个数据
print("数据集中第一个数据:")
print(train_dataset[0])


# 加载评估器
metric = evaluate.load("f1")
# 坑爹, 不能用组合方式, 因为 f1 的时候需要传入 average 参数, 但在 compute 中传入无效
# metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # 尴尬, 只能分开计算
    result = {}
    result["f1_micro"] = metric.compute(predictions=predictions, references=labels, average="micro")["f1"]
    result["f1_macro"] = metric.compute(predictions=predictions, references=labels, average="macro")["f1"]

    return result


# 定义训练参数
model = AutoModelForSequenceClassification.from_pretrained(
    pretrain_model_name,
    num_labels=15,
    id2label=id2label,
    label2id=label_map,
)

training_args = TrainingArguments(
    "tnews_{}".format(os.path.basename(pretrain_model_name)),
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    save_total_limit=1,
)

# 定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()
# 评估
eval_result = trainer.evaluate()
print("评估结果:", eval_result)
# 保存模型
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)
trainer.save_metrics("all", eval_result)
breakpoint()
