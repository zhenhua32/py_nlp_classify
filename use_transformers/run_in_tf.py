"""
使用 transformers 进行文本分类, 使用 tensorflow 模型
"""

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer


pretrain_model_name = "bert-base-chinese"

# 定义模型
# 这个模型非常大, 我用 kaggle 的 GPU 实例去跑, 直接把 GPU 显存(16GB)占满了
model = TFAutoModelForSequenceClassification.from_pretrained(pretrain_model_name, num_labels=15,)

# 定义分词器
tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name)

# 加载数据集
train_list = []
train_label = []
with open("../data/train.csv", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().split("\t")
        train_list.append(line[1])
        train_label.append(line[0])

train_dataset = tokenizer(train_list, padding="max_length", truncation=True, max_length=32, return_tensors="tf",)

dev_list = []
dev_label = []
with open("../data/dev.csv", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().split("\t")
        dev_list.append(line[1])
        dev_label.append(line[0])

dev_dataset = tokenizer(dev_list, padding=True, truncation=True, max_length=32, return_tensors="tf",)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)


# 将 label 变成数字
# label_map = {label: i for i, label in enumerate(set(train_label))}
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
print(label_map)
train_label = [label_map[label] for label in train_label]
dev_label = [label_map[label] for label in dev_label]

# 使用 tf.data.Dataset 来转换数据
train_features = {x: train_dataset[x] for x in tokenizer.model_input_names}
train_tf_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_label))
train_tf_dataset = train_tf_dataset.shuffle(len(train_list)).batch(8)

eval_features = {x: dev_dataset[x] for x in tokenizer.model_input_names}
eval_tf_dataset = tf.data.Dataset.from_tensor_slices((eval_features, dev_label))
eval_tf_dataset = eval_tf_dataset.batch(8)

# 训练
model.fit(train_tf_dataset, validation_data=eval_tf_dataset, epochs=5)
# 评估
model.evaluate(eval_tf_dataset)
# 保存模型
model.save_pretrained("./model/")
