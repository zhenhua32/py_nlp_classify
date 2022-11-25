import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, AutoConfig, BertConfig, BertTokenizer
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_bert_layers(bert_path: str, load_pretrain=True) -> BertModel:
    """
    bert 模型的输出参考这里
    https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
    按元组的顺序来是
    last_hidden_state: (batch_size, sequence_length, hidden_size)
    pooler_output: (batch_size, hidden_size)
    然后是可选的, TODO: 我对后面的输出还不了解
    hidden_states: 是个元组, (batch_size, sequence_length, hidden_size)
    """
    if load_pretrain:
        bert = BertModel.from_pretrained(bert_path)
    else:
        bert_config = BertConfig.from_pretrained(bert_path)
        bert = BertModel(bert_config)
    return bert


class BertLinear(nn.Module):
    """
    定义一个 bert 线性分类网络
    """

    def __init__(self, bert_path: str, num_labels: int, load_pretrain=True) -> None:
        super().__init__()
        self.bert = get_bert_layers(bert_path, load_pretrain)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        # pooled_output: (batch_size, hidden_size)
        logits = self.linear(pooled_output)
        # logits: (batch_size, num_labels)
        return logits


class BertDropout2d(nn.Module):
    """
    定义一个 bert + dropout2d 线性分类网络
    """

    def __init__(self, bert_path: str, num_labels: int, max_token_len=64, load_pretrain=True) -> None:
        super().__init__()
        self.max_token_len = max_token_len

        self.bert = get_bert_layers(bert_path, load_pretrain)
        self.dropout2d = nn.Dropout2d(0.5)
        self.linear = nn.Linear(self.bert.config.hidden_size * max_token_len, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # sequence_output: (batch_size, sequence_length, hidden_size)
        # 这个 permute 操作不理解, 源自 https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/2
        sequence_output = self.dropout2d(sequence_output.permute(0, 2, 1)).permute(0, 2, 1)
        logits = self.linear(sequence_output.view(sequence_output.shape[0], -1))
        return logits


class BertLastHiddenState(nn.Module):
    """
    定义一个 bert 线性分类网络, 使用了 bert 的第一个输出.
    """

    def __init__(self, bert_path: str, num_labels: int, max_token_len=64, load_pretrain=True) -> None:
        super().__init__()
        self.max_token_len = max_token_len

        self.bert = get_bert_layers(bert_path, load_pretrain)
        input_size = self.bert.config.hidden_size * max_token_len
        self.linear = nn.Linear(input_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # sequence_output: (batch_size, sequence_length, hidden_size)
        logits = self.linear(sequence_output.view(sequence_output.shape[0], -1))
        return logits


class BertLinearMix(nn.Module):
    """
    定义一个 bert 线性分类网络, 同时使用 bert 的前两个输出
    """

    def __init__(self, bert_path: str, num_labels: int, max_token_len=64, load_pretrain=True) -> None:
        super().__init__()
        self.max_token_len = max_token_len

        self.bert = get_bert_layers(bert_path, load_pretrain)
        input_size = self.bert.config.hidden_size * max_token_len + self.bert.config.hidden_size
        self.linear = nn.Linear(input_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        # sequence_output: (batch_size, sequence_length, hidden_size)
        # pooled_output: (batch_size, hidden_size)
        logits = self.linear(
            torch.cat(
                [
                    sequence_output.view(sequence_output.shape[0], -1),
                    pooled_output,
                ],
                dim=1,
            )
        )
        return logits


class BertLSTM(nn.Module):
    """
    定义一个 bert + 双向 lstm 的分类网络
    """

    def __init__(
        self, bert_path: str, num_labels: int, max_token_len: int = 64, lstm_hidden_size: int = 128, load_pretrain=True
    ) -> None:
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.max_token_len = max_token_len

        self.bert = get_bert_layers(bert_path, load_pretrain)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.lstm_hidden_size * 2 * max_token_len, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # sequence_output: (batch_size, sequence_length, hidden_size)
        sequence_output, _ = self.lstm(sequence_output)
        # sequence_output: (batch_size, sequence_length, lstm_hidden_size * 2)
        logits = self.linear(sequence_output.reshape(sequence_output.shape[0], -1))
        return logits


class BertCNN(nn.Module):
    """
    定义一个 bert + cnn 的分类网络
    """

    def __init__(self, bert_path: str, num_labels: int, load_pretrain=True):
        super().__init__()
        self.bert = get_bert_layers(bert_path, load_pretrain)
        self.cnn = nn.ModuleList([nn.Conv1d(self.bert.config.hidden_size, 128, k, padding="same") for k in [2, 3, 4]])
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.linear = nn.Linear(128 * 64 * 3, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # sequence_output: (batch_size, sequence_length, hidden_size)
        sequence_output = sequence_output.permute(0, 2, 1)
        # sequence_output: (batch_size, hidden_size, sequence_length)
        cnn_list = [self.relu(conv(sequence_output)) for conv in self.cnn]
        # cnn_list 中每个的 shape: (batch_size, 128, sequence_length)
        pool_list = [self.max_pool(cnn) for cnn in cnn_list]
        # pool_list 中每个的 shape: (batch_size, 128, sequence_length)
        pool_output = torch.cat(pool_list, dim=1).flatten(start_dim=1)
        # pool_output: (batch_size, 128 * 64)
        logits = self.linear(pool_output)
        return logits


class BertCNN2D(nn.Module):
    """
    定义一个 bert + cnn 的分类网络
    """

    def __init__(self, bert_path: str, num_labels: int, load_pretrain=True):
        super().__init__()
        self.bert = get_bert_layers(bert_path, load_pretrain)
        self.cnn = nn.ModuleList(
            [
                nn.Conv2d(1, 64, (k, self.bert.config.hidden_size), padding=0)
                for k in [2, 3, 4]
            ]
        )
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0))
        self.linear = nn.Linear(64 * (61 + 62 + 63), num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # sequence_output: (batch_size, sequence_length, hidden_size)
        sequence_output = sequence_output.unsqueeze(1)
        # sequence_output: (batch_size, 1, sequence_length, hidden_size)
        cnn_list = [self.relu(conv(sequence_output)) for conv in self.cnn]
        # 必须使用 padding=0, 才能把最后一个维度压缩成0, 不能用 padding="same". 但是这种情况下, 第三维度会变成动态的
        # cnn_list 中每个的 shape: (batch_size, 64, sequence_length, 1)
        pool_list = [self.max_pool(cnn) for cnn in cnn_list]
        # pool_list 中每个的 shape: (batch_size, 128, sequence_length, 1)
        pool_output = torch.cat(pool_list, dim=2).flatten(start_dim=1)
        # pool_output: (batch_size, 64 * 64)
        logits = self.linear(pool_output)
        return logits


class PlModel(pl.LightningModule):
    """
    定义一个 pl 模型, 来整合所有的外围操作
    """

    def __init__(self, model: nn.Module, id2label: dict) -> None:
        """
        model: 实际的模型
        id2label: id 到 label 的映射
        """
        super().__init__()
        self.model = model
        self.id2label = id2label
        self.example_input_array = (torch.zeros((4, 64), dtype=torch.long), torch.zeros((4, 64), dtype=torch.long))

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask)
        return logits

    def training_step(self, batch, batch_idx):
        """
        训练模型的单个步骤
        """
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def validation_step(self, batch, batch_idx):
        """
        单个验证步骤
        """
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        self.log("val_loss", loss)
        pred = F.softmax(logits, dim=1).argmax(dim=1)
        return pred, labels

    def validation_epoch_end(self, outputs):
        """
        验证集的整体结果
        """
        preds = []
        labels = []
        for pred, label in outputs:
            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="weighted", zero_division=0)
        recall = recall_score(labels, preds, average="weighted", zero_division=0)
        f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.log("val_precision", precision, prog_bar=True, sync_dist=True)
        self.log("val_recall", recall, prog_bar=True, sync_dist=True)
        self.log("val_f1_micro", f1_micro, prog_bar=True, sync_dist=True)
        self.log("val_f1_macro", f1_macro, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        单个预测步骤, 这个基本是为 dataloader 服务的
        """
        logits = self(batch[0], batch[1])
        probs, pred = F.softmax(logits, dim=1).max(dim=1)
        pred_labels = [self.id2label[p] for p in pred.detach().cpu().numpy()]
        return pred_labels, probs.detach().cpu().numpy()

    def init_predict(self, model_dir: str, max_token_len=64):
        """
        初始化预测, 从模型目录中加载 bert 分词器
        """
        self.max_token_len = max_token_len
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_dir)

    def predict_raw(self, texts: list):
        """
        直接一步到位, 输入文本序列, 对每个文本进行预测
        """
        if not hasattr(self, "tokenizer") or not hasattr(self, "max_token_len"):
            raise RuntimeError("请先调用 init_predict 方法")

        inputs = self.tokenizer(
            texts, padding="max_length", truncation=True, max_length=self.max_token_len, return_tensors="pt"
        )

        # TODO: 加上 GPU 支持
        # TODO: predict_step 会自动调用 GPU 吗? 答案是并不会, 这些功能是 trainer 提供的
        with torch.no_grad():
            pred_labels, probs = self.predict_step((inputs["input_ids"], inputs["attention_mask"]), 0)

        result = []
        for text, pred_label, prob in zip(texts, pred_labels, probs):
            result.append({"text": text, "pred_label": pred_label, "prob": prob})

        return result
