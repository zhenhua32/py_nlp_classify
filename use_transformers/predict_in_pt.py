import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

my_model = AutoModelForSequenceClassification.from_pretrained("./my_model")
my_tokenizer = AutoTokenizer.from_pretrained("./my_model")


def predict(data_list, batch_size=32):
    """
    批次预测
    """
    results = []
    data_len = len(data_list)
    cur = 0
    while cur < data_len:
        batch_data = data_list[cur : cur + batch_size]
        batch_tensor = my_tokenizer(
            batch_data, padding="max_length", truncation=True, max_length=32, return_tensors="pt"
        )
        logits = my_model(**batch_tensor).logits
        predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1).tolist()
        results.extend(predictions)
        cur += batch_size
    assert len(results) == data_len
    return results


test_data_list = []
with open("/kaggle/input/clue-tnews/test.json", "r", encoding="utf-8") as f:
    for line in f:
        sentence = json.loads(line)["sentence"]
        test_data_list.append(sentence)

print(len(test_data_list))

results = predict(test_data_list)


# 更多的处理, 把结果写入文件, 每行是一个预测, json 格式
label_list = []
with open("/kaggle/input/clue-tnews/labels.json", "r", encoding="utf-8") as f:
    for line in f:
        label_list.append(json.loads(line))
id_label_dict = {}
for index, item in enumerate(label_list):
    id_label_dict[index] = item

with open("tnews11_predict.json", "w", encoding="utf-8") as fw:
    with open("/kaggle/input/clue-tnews/test.json", "r", encoding="utf-8") as f:
        index = 0
        for line in f:
            data = json.loads(line)
            predict_id = results[index]
            label = id_label_dict[predict_id]["label"]
            label_desc = id_label_dict[predict_id]["label_desc"]
            one = {"id": data["id"], "label": label, "label_desc": label_desc}
            fw.write(json.dumps(one) + "\n")
            index += 1
