import json

import tensorflow as tf
from dotenv import load_dotenv

# 在 easytransfer 导入之前载入配置, 有环境变量需要设置
load_dotenv("./config/.env")

from easytransfer import Config, base_model, layers, model_zoo, preprocessors
from easytransfer.datasets import CSVReader, CSVWriter
from easytransfer.evaluators import classification_eval_metrics
from easytransfer.losses import softmax_cross_entropy


class TextClassification(base_model):
    """
    定义文本分类模型
    """

    def __init__(self, **kwargs):
        super(TextClassification, self).__init__(**kwargs)
        self.user_defined_config = kwargs["user_defined_config"]

    def build_logits(self, features, mode=None):
        """构图

        Args:
            features ([type]): [description]
            mode ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        # 负责对原始数据进行预处理，生成模型需要的特征，比如：input_ids, input_mask, segment_ids等
        preprocessor = preprocessors.get_preprocessor(
            self.pretrain_model_name_or_path, user_defined_config=self.user_defined_config
        )
        # 负责构建网络的backbone
        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

        dense = layers.Dense(self.num_labels, kernel_initializer=layers.get_initializer(0.02), name="dense")
        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
        _, pooled_output = model([input_ids, input_mask, segment_ids], mode=mode)
        logits = dense(pooled_output)

        # 用于 continue finetune
        # self.check_and_init_from_checkpoint(mode)
        return logits, label_ids

    def build_loss(self, logits, labels):
        """定义损失函数

        Args:
            logits ([type]): logits returned from build_logits
            labels ([type]): labels returned from build_logits

        Returns:
            [type]: [description]
        """
        return softmax_cross_entropy(labels, self.num_labels, logits)

    def build_eval_metrics(self, logits, labels):
        """定义评估指标

        Args:
            logits ([type]): logits returned from build_logits
            labels ([type]): labels returned from build_logits

        Returns:
            [type]: [description]
        """
        return classification_eval_metrics(logits, labels, self.num_labels)

    def build_predictions(self, output):
        """定义预测输出

        Args:
            output ([type]): returned from build_logits

        Returns:
            [type]: [description]
        """
        logits, _ = output
        predictions = dict()
        predictions["predictions"] = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return predictions


def train(config_json):
    config = Config(mode="train_and_evaluate_on_the_fly", config_json=config_json)
    app = TextClassification(user_defined_config=config)

    train_reader = CSVReader(
        input_glob=app.train_input_fp, is_training=True, input_schema=app.input_schema, batch_size=app.train_batch_size,
    )
    eval_reader = CSVReader(
        input_glob=app.eval_input_fp, is_training=False, input_schema=app.input_schema, batch_size=app.eval_batch_size,
    )

    app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)


def predict(config_json):
    config = Config(mode="predict_on_the_fly", config_json=config_json)
    app = TextClassification(user_defined_config=config)

    pred_reader = CSVReader(
        input_glob=app.predict_input_fp,
        is_training=False,
        input_schema=app.input_schema,
        batch_size=app.predict_batch_size,
    )
    pred_writer = CSVWriter(output_glob=app.predict_output_fp, output_schema=app.output_schema)

    result = app.run_predict(reader=pred_reader, writer=None, checkpoint_path=app.predict_checkpoint_path)
    for row in result:
        print(row)
        break


def evaluate(config_json):
    config = Config(mode="evaluate_on_the_fly", config_json=config_json)
    app = TextClassification(user_defined_config=config)

    eval_reader = CSVReader(
        input_glob=app.eval_input_fp, is_training=False, input_schema=app.input_schema, batch_size=app.eval_batch_size,
    )

    result = app.run_evaluate(reader=eval_reader, checkpoint_path=app.eval_ckpt_path)
    print(result)


if __name__ == "__main__":
    config_json = json.load(open("./model_dir/user_config.json", "r", encoding="utf-8"))
    # tensorboard --logdir model_dir
    # train()
    # predict()
    evaluate()

