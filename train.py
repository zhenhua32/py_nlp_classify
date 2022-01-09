import json
from dotenv import load_dotenv

# 在 easytransfer 导入之前载入配置, 有环境变量需要设置
load_dotenv("./config/.env", override=True)

from easytransfer import Config, base_model, layers, model_zoo, preprocessors
from easytransfer.datasets import CSVReader, CSVWriter

from model import TextClassification


def train():
    config_json = json.load(open("./config/user_config.json"))

    config = Config(mode="train_and_evaluate_on_the_fly", config_json=config_json)
    app = TextClassification(user_defined_config=config)

    train_reader = CSVReader(
        input_glob=app.train_input_fp, is_training=True, input_schema=app.input_schema, batch_size=app.train_batch_size,
    )
    eval_reader = CSVReader(
        input_glob=app.eval_input_fp, is_training=False, input_schema=app.input_schema, batch_size=app.eval_batch_size,
    )

    app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)


def predict():
    config_json = json.load(open("./config/user_config.json"))
    config_json["predict_config"]["predict_checkpoint_path"] = config_json["predict_config"]["predict_checkpoint_path"].format(5001)

    config = Config(mode="predict_on_the_fly", config_json=config_json)
    app = TextClassification(user_defined_config=config)

    pred_reader = CSVReader(
        input_glob=app.predict_input_fp,
        is_training=False,
        input_schema=app.input_schema,
        batch_size=app.predict_batch_size,
    )

    pred_writer = CSVWriter(output_glob=app.predict_output_fp, output_schema=app.output_schema)

    app.run_predict(reader=pred_reader, writer=pred_writer, checkpoint_path=app.predict_checkpoint_path)


if __name__ == "__main__":
    # train()
    predict()
