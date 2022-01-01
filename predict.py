import json
from dotenv import load_dotenv

# 在 easytransfer 导入之前载入配置, 有环境变量需要设置
load_dotenv("./config/.env", override=True)

from easytransfer import Config, base_model, layers, model_zoo, preprocessors
from easytransfer.datasets import CSVReader, CSVWriter
import tensorflow as tf

from model import TextClassification


class Predictor:
    """
    构建一个预测器
    """
    def __init__(self) -> None:
        config_json = json.load(open("./config/user_config.json"))
        config_json["predict_config"]["predict_checkpoint_path"] = config_json["predict_config"][
            "predict_checkpoint_path"
        ].format(5001)

        config = Config(mode="predict_on_the_fly", config_json=config_json)
        app = TextClassification(user_defined_config=config)

        self.app = app

    def predict(self, text_list: list):
        """
        输入一个文本序列, 进行预测
        """
        result = self.app.estimator.predict(
            input_fn=self.get_input_fn(text_list),
            yield_single_examples=True,
            checkpoint_path=self.app.predict_checkpoint_path,
        )
        result = list(result)
        print("长度", len(result))
        for row in result:
            print(row)

    def get_input_fn(self, text_list):
        """
        输入一个文本序列, 返回一个 input_fn
        """

        def input_fn():
            # label 造假, 用的是固定的, 毕竟是预测, 没人关心这个
            ds = tf.data.Dataset.from_tensor_slices({"content": text_list, "label": ["agriculture"] * len(text_list)})
            ds = ds.batch(2)
            return ds

        return input_fn


if __name__ == "__main__":
    text_list = [
        "出租屋里的年轻人",
        "范冰冰又出新电影了",
        "世界政客的问题在于美国人的政治思想",
    ]
    predictor = Predictor()
    predictor.predict(text_list)

