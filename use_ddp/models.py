"""
用最简单的模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


def get_bert_layers(bert_path: str, load_pretrain=True) -> BertModel:
    """
    bert 模型的输出参考这里
    https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
    按元组的顺序来是
    last_hidden_state: (batch_size, sequence_length, hidden_size)
    pooler_output: (batch_size, hidden_size)
    然后是可选的, TODO: 我对后面的输出还不了解
    hidden_states: 是个元组, 是每一层的输出, (batch_size, sequence_length, hidden_size)
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
