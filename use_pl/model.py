import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, AutoConfig, BertConfig
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

    def __init__(self, bert_path: str, num_labels: int) -> None:
        super().__init__()
        self.bert = get_bert_layers(bert_path)
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

    def __init__(self, bert_path: str, num_labels: int, max_token_len=64) -> None:
        super().__init__()
        self.max_token_len = max_token_len

        self.bert = get_bert_layers(bert_path)
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


class BertLSTM(nn.Module):
    def __init__(self, bert_path: str, num_labels: int, max_token_len: int = 64, lstm_hidden_size: int = 128) -> None:
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.max_token_len = max_token_len

        self.bert = get_bert_layers(bert_path)
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


class PlModel(pl.LightningModule):
    """
    定义一个 pl 模型, 来整合所有的外围操作
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.example_input_array = (torch.zeros((64, 64), dtype=torch.long), torch.zeros((64, 64), dtype=torch.long))

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
