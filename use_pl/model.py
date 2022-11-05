import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, AutoConfig, BertConfig
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_bert_layers(bert_path: str, load_pretrain=True) -> BertModel:
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
        logits = self.linear(pooled_output)
        return logits


class PlModel(pl.LightningModule):
    """
    定义一个 pl 模型, 来整合所有的外围操作
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        """
        训练模型的单个步骤
        """
        input_ids, attention_mask, labels = batch
        logits = self.model(input_ids, attention_mask)
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
        logits = self.model(input_ids, attention_mask)
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
        f1_micro = f1_score(labels, preds, average="micro")
        f1_macro = f1_score(labels, preds, average="macro")
        self.log("val_acc", acc)
        self.log("val_f1_micro", f1_micro)
        self.log("val_f1_macro", f1_macro)
