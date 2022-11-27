# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team and Alibaba inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function

import copy
import json
import math

import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    真难, 我看不懂 torch.erf, 不会解这个定积分公式
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`."""

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        emb_size=-1,
        num_hidden_layers=12,
        transformer_type="original",  # support 'original', 'universal' , 'albert' and 'act'
        transition_function="linear",  # support 'linear', 'cnn', 'rnn'
        weighted_transformer=0,  # support 0 or 1
        num_rolled_layers=3,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        attention_type="self",
        rezero=False,
        pre_ln=False,
        squeeze_excitation=False,
        transfer_matrix=False,
        dim_dropout=False,
        roberta_style=False,
        set_mask_zero=False,
        init_scale=False,
        safer_fp16=False,
        grad_checkpoint=False,
    ):  # support 0/1
        """Constructs BertConfig.

        参数介绍不全, 肯定是哪里抄来的, 没有补全. 尤其是这里有个 transformer_type 参数, 支持不同的变种.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.num_hidden_layers = num_hidden_layers
        self.transformer_type = transformer_type
        self.transition_function = transition_function
        self.weighted_transformer = weighted_transformer
        self.num_rolled_layers = num_rolled_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.attention_type = attention_type
        self.rezero = rezero
        self.pre_ln = pre_ln
        self.squeeze_excitation = squeeze_excitation
        self.transfer_matrix = transfer_matrix
        self.dim_dropout = dim_dropout
        self.set_mask_zero = set_mask_zero
        self.roberta_style = roberta_style
        self.init_scale = init_scale
        self.safer_fp16 = safer_fp16
        self.grad_checkpoint = grad_checkpoint

    @classmethod
    def from_dict(cls, json_object: dict):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config: BertConfig, variance_epsilon=1e-12, special_size=None):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(BERTLayerNorm, self).__init__()
        self.config = config
        hidden_size = special_size if special_size is not None else config.hidden_size
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon if not config.roberta_style else 1e-5

    def forward(self, x: torch.Tensor):
        """
        公式可以见 torch.nn.LayerNorm
        https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        """
        previous_type = x.type()
        if self.config.safer_fp16:
            x = x.float()
        # 在最后一个维度上求均值
        u = x.mean(-1, keepdim=True)
        # 标准差
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        if self.config.safer_fp16:
            return (self.gamma * x + self.beta).type(previous_type)
        else:
            return self.gamma * x + self.beta


class BERTEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        hidden_size = config.hidden_size if config.emb_size < 0 else config.emb_size
        # padding_idx 设置了, 会让这些值对梯度的贡献为 0 (不参与梯度更新)
        self.word_embeddings = nn.Embedding(
            config.vocab_size, hidden_size, padding_idx=1 if config.roberta_style else None
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, hidden_size, padding_idx=1 if config.roberta_style else None
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, hidden_size)
        self.config = config
        # 如果自定义而来 emb_size, 需要加一个线性层来转到 hidden_size 的大小
        self.proj = None if config.emb_size < 0 else nn.Linear(config.emb_size, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # 这个 LayerNorm 命名为大写开头的, 是为了兼容 tf checkpoint
        self.LayerNorm = BERTLayerNorm(config, special_size=hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids=None, adv_embedding=None):
        # input_ids: [batch_size, seq_len]
        seq_length = input_ids.size(1)
        if not self.config.roberta_style:
            # 从 0 开始, 一直到 seq_length, 构建位置 id
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        else:
            # roberta_style 中是将 1 作为 padding_id 的. 这里就是找出非 1 的位置
            mask = input_ids.ne(1).int()
            # torch.cumsum 是逐元素累加, 在 dim=1 上, 也就是 seq_len 上
            # * mask 是为了跳过填充部分
            position_ids = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + 1
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 词嵌入
        words_embeddings = self.word_embeddings(input_ids) if adv_embedding is None else adv_embedding
        # words_embeddings: [batch_size, seq_len, hidden_size]
        if self.config.set_mask_zero:
            # 103 是 [MASK] 的 id. 这是 AI 给我推理出来的, 牛!
            words_embeddings[input_ids == 103] = 0.0
        position_embeddings = self.position_embeddings(position_ids)
        # position_embeddings: [batch_size, seq_len, hidden_size]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # token_type_embeddings: [batch_size, seq_len, hidden_size]

        if not self.config.roberta_style:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = words_embeddings + position_embeddings
        # embeddings: [batch_size, seq_len, hidden_size]
        # 用上了前面看到的 BertLayerNorm
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        if self.proj is not None:
            embeddings = self.proj(embeddings)
            embeddings = self.dropout(embeddings)
            # 归一化到最后一个维度都是 config.hidden_size
        else:
            return embeddings, words_embeddings


class BERTFactorizedAttention(nn.Module):
    """
    Factorized 是因式分解
    """
    def __init__(self, config: BertConfig):
        super(BERTFactorizedAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        # 默认情况下 attention_head_size 是 64 = 768 / 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 我还是没看懂, all_head_size 有可能不等于 hidden_size 吗?
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # query 是从 hidden_size => all_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor, *size):
        """
        转置
        """
        # 放弃最后一个维度, 加上两个新维度
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # 交换维度, 比如从 (2, 3, 5) 变换成 (5, 2, 3)
        return x.permute(size)

    def forward(self, hidden_states: torch.Tensor, attention_mask):
        # 应该是这个形状吧, 但看起来有 4 个维度
        # hidden_states: [batch_size, seq_len, hidden_size]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # 然后都进行线性层之后, shape 是 [batch_size, seq_len, all_head_size]

        # transpose_for_scores 第一步会将 shape 转换为 [batch_size, seq_len, num_attention_heads, attention_head_size]
        # 然后第二步就是用 [0, 2, 3, 1] 交换维度, 变成 [batch_size, num_attention_heads, attention_head_size, seq_len]
        query_layer = self.transpose_for_scores(mixed_query_layer, 0, 2, 3, 1)
        # key_layer 和 value_layer 是 [batch_size, num_attention_heads, seq_len, attention_head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer, 0, 2, 1, 3)
        value_layer = self.transpose_for_scores(mixed_value_layer, 0, 2, 1, 3)

        s_attention_scores = query_layer + attention_mask
        s_attention_probs = nn.Softmax(dim=-1)(s_attention_scores)
        s_attention_probs = self.dropout(s_attention_probs)
        # s_attention_probs: [batch_size, num_attention_heads, attention_head_size, seq_len]

        # c_attention_scores: [batch_size, num_attention_heads, seq_len, attention_head_size]
        c_attention_probs = nn.Softmax(dim=-1)(key_layer)
        # 矩阵乘法
        s_context_layer = torch.matmul(s_attention_probs, value_layer)
        # s_context_layer: [batch_size, num_attention_heads, attention_head_size, attention_head_size]
        context_layer = torch.matmul(c_attention_probs, s_context_layer)
        # context_layer: [batch_size, num_attention_heads, seq_len, attention_head_size]

        # contiguous 返回连续张量
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # context_layer: [batch_size, seq_len, num_attention_heads, attention_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # new_context_layer_shape: [batch_size, seq_len, all_head_size]
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer: [batch_size, seq_len, all_head_size]
        return context_layer


def dim_dropout(x: torch.Tensor, p=0, dim=-1, training=False):
    if training is False or p == 0:
        return x
    # 伯努利分布
    # 后面的 (x.data.new(x.size()).zero_() + 1) 操作可以简化为 x.data.new_ones(x.size()), 就是生成 x 形状的值为 1 的张量
    # 然后乘以 (1 - p) 就是存留的概率, p 是 dropout 的概率
    # dropout_mask 是一个二值张量, 0 表示丢弃, 1 表示存留
    dropout_mask = torch.bernoulli((1 - p) * (x.data.new(x.size()).zero_() + 1))
    # dim 是选择的维度, 然后除以这个维度上的总和
    # 这会相应的放大剩下的值, 使得总和不变
    return dropout_mask * (dropout_mask.size(dim) / torch.sum(dropout_mask, dim=dim, keepdim=True)) * x


class BERTSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 还是一直在疑惑, hidden_size 和 all_head_size 是一样的
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.config = config
        # 上面都和 BERTFactorizedAttention 是一样的
        # pre_ln 是预先进行 LayerNorm 的意思
        if config.pre_ln:
            self.LayerNorm = BERTLayerNorm(config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        # 是否预先进行 LayerNorm
        if self.config.pre_ln:
            hidden_states = self.LayerNorm(hidden_states)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 原来还是有点不一样的, 原来的 shape 是 [batch_size, seq_len, num_attention_heads, attention_head_size]
        # x.permute(0, 2, 1, 3) 后是 [batch_size, num_attention_heads, seq_len, attention_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores: [batch_size, num_attention_heads, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if head_mask is not None and not self.training:
            for i, mask in enumerate(head_mask):
                if head_mask[i] == 1:
                    # 这是 head 的掩码, 所以就是让某个 head 的值都变成 0
                    attention_scores[:, i, :, :] = 0.0
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if not self.config.dim_dropout:
            attention_probs = self.dropout(attention_probs)
        else:
            attention_probs = dim_dropout(
                attention_probs, p=self.config.attention_probs_dropout_prob, dim=-1, training=self.training
            )

        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer: [batch_size, num_attention_heads, seq_len, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # context_layer: [batch_size, seq_len, num_attention_heads, attention_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer: [batch_size, seq_len, hidden_size]
        return context_layer


class BERTSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super(BERTSelfOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if not config.pre_ln and not config.rezero:
            self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.rezero:
            # 好像就是一个 0.99, 另一个 1
            self.res_factor = nn.Parameter(torch.Tensor(1).fill_(0.99).to(dtype=next(self.parameters()).dtype))
            self.factor = nn.Parameter(torch.ones(1).to(dtype=next(self.parameters()).dtype))

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 第一种情况, 没有 pre_ln, 也没有 rezero
        if not self.config.rezero and not self.config.pre_ln:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 第二种情况, 有 rezero
        elif self.config.rezero:
            hidden_states = hidden_states + self.factor * input_tensor
        # 第三种情况, 有 pre_ln
        else:
            pass
        # 输出的 shape 是 [batch_size, seq_len, hidden_size]
        return hidden_states


class BERTAttention(nn.Module):
    """
    使用两种可选的注意力机制, 然后加上一个 BERTSelfOutput
    """
    def __init__(self, config: BertConfig):
        super(BERTAttention, self).__init__()
        if config.attention_type.lower() == "self":
            self.self = BERTSelfAttention(config)
        elif config.attention_type.lower() == "factorized":
            self.self = BERTFactorizedAttention(config)
        else:
            raise ValueError("Attention type must in [self, factorized], but got {}".format(config.attention_type))
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_output = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_output, input_tensor)
        # 输出的 shape 是 [batch_size, seq_len, hidden_size]
        return attention_output


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableConv1d, self).__init__()
        # 这里重置了 padding, 所以这个参数没用了
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias
        )
        # (Lin + 2 * padding - dialation * (kernel_size - 1) - 1 ) / stride + 1
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        # kernel_size = 1, stride = 1, padding = 0, dilation = 1, groups = 1
        # (Lin + 2 * 0 - 1 * (1 - 1) - 1 ) / 1 + 1
        # (Lin - 1) / 1 + 1 即 Lin

    def forward(self, x):
        # 注意看下面的例子, hidden_states.transpose(-1, -2), 所以
        # x shape 是 [batch_size, hidden_size, seq_len]
        x = self.depthwise(x)
        # 看下面的使用中, kernel_size = 7, 其他参数都是默认的. in_channels = hidden_size, out_channels = 4 * hidden_size
        # (Lin + 2 * 3 - 1 * (7 - 1) - 1 ) / 1 + 1
        # 即 Lin. 然后 Lin 就是 seq_len.
        # shape 是 [batch_size, hidden_size, seq_len]
        x = self.pointwise(x)
        # shape 是 [batch_size, out_channels, seq_len]
        # 实例中是 [batch_size, 4 * hidden_size, seq_len]
        return x


class BERTIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super(BERTIntermediate, self).__init__()
        self.config = config
        if self.config.pre_ln:
            self.LayerNorm = BERTLayerNorm(config)
        self.intermediate_act_fn = gelu
        if config.transition_function.lower() == "linear":
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        elif config.transition_function.lower() == "cnn":
            self.cnn = DepthwiseSeparableConv1d(config.hidden_size, 4 * config.hidden_size, kernel_size=7)
            # cnn shape: [batch_size, 4 * hidden_size, hidden_size]
        elif config.config.hidden_size.lower() == "rnn":
            raise NotImplementedError("rnn transition function is not implemented yet")
        else:
            raise ValueError("Only support linear/cnn/rnn")

    def forward(self, hidden_states):
        if self.config.pre_ln:
            hidden_states = self.LayerNorm(hidden_states)
        if self.config.transition_function.lower() == "linear":
            hidden_states = self.dense(hidden_states)
            # shape 是 [batch_size, seq_len, intermediate_size]
        elif self.config.transition_function.lower() == "cnn":
            # fuck 我前面算了一遍, 没想到这里调换了维度
            hidden_states = self.cnn(hidden_states.transpose(-1, -2)).transpose(-1, -2)
            # 然后输出的 shape 是 [batch_size, seq_len, hidden_size * 4], 最后一个维度就是默认参数中的 3072
        else:
            pass
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, config):
        super(SqueezeExcitationBlock, self).__init__()
        self.down_sampling = nn.Linear(config.hidden_size, config.hidden_size // 4)
        self.up_sampling = nn.Linear(config.hidden_size // 4, config.hidden_size)

    def forward(self, hidden_states):
        squeeze = torch.mean(hidden_states, 1, keepdim=True)
        excitation = torch.sigmoid(self.up_sampling(gelu(self.down_sampling(squeeze))))
        return hidden_states * excitation


class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.config = config
        if config.transition_function.lower() == "linear":
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        elif config.transition_function.lower() == "cnn":
            self.cnn = DepthwiseSeparableConv1d(4 * config.hidden_size, config.hidden_size, kernel_size=7)
        elif config.config.hidden_size.lower() == "rnn":
            raise NotImplementedError("rnn transition function is not implemented yet")
        else:
            raise ValueError("Only support linear/cnn/rnn")
        if not config.pre_ln and not config.rezero:
            self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.squeeze_excitation:
            self.SEblock = SqueezeExcitationBlock(config)
        if config.rezero:
            self.res_factor = nn.Parameter(torch.Tensor(1).fill_(0.99).to(dtype=next(self.parameters()).dtype))
            self.factor = nn.Parameter(torch.ones(1).to(dtype=next(self.parameters()).dtype))

    def forward(self, hidden_states, input_tensor):
        if self.config.transition_function.lower() == "linear":
            hidden_states = self.dense(hidden_states)
        elif self.config.transition_function.lower() == "cnn":
            hidden_states = self.cnn(hidden_states.transpose(-1, -2)).transpose(-1, -2)
        else:
            pass
        hidden_states = self.dropout(hidden_states)
        if self.config.squeeze_excitation:
            hidden_states = self.SEblock(hidden_states)
        if not self.config.rezero and not self.config.pre_ln:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        elif self.config.rezero:
            hidden_states = hidden_states + self.factor * input_tensor
        else:
            pass
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return attention_output, layer_output


class BERTWeightedLayer(nn.Module):
    def __init__(self, config):
        super(BERTWeightedLayer, self).__init__()
        self.config = config
        self.self = BERTSelfAttention(config)
        self.attention_head_size = self.self.attention_head_size

        # parameter for multi branches
        self.w_o = nn.ModuleList(
            [nn.Linear(self.attention_head_size, config.hidden_size) for _ in range(config.num_attention_heads)]
        )
        self.w_kp = torch.rand(config.num_attention_heads)
        self.w_kp = nn.Parameter(self.w_kp / self.w_kp.sum())
        self.w_a = torch.rand(config.num_attention_heads)
        self.w_a = nn.Parameter(self.w_a / self.w_a.sum())

        # parameter for FFN
        self.intermediate = BERTIntermediate(config)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        self_output = self.self(hidden_states, attention_mask)
        self_outputs = self_output.split(self.self.attention_head_size, dim=-1)
        self_outputs = [self.w_o[i](self_outputs[i]) for i in range(len(self_outputs))]
        self_outputs = [self.dropout(self_outputs[i]) for i in range(len(self_outputs))]
        self_outputs = [kappa * output for kappa, output in zip(self.w_kp, self_outputs)]
        self_outputs = [self.intermediate(self_outputs[i]) for i in range(len(self_outputs))]
        self_outputs = [self.output(self_outputs[i]) for i in range(len(self_outputs))]
        self_outputs = [self.dropout(self_outputs[i]) for i in range(len(self_outputs))]
        self_outputs = [alpha * output for alpha, output in zip(self.w_a, self_outputs)]
        output = sum(self_outputs)
        return self.LayerNorm(hidden_states + output)


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            if config.weighted_transformer:
                self.layer.append(BERTWeightedLayer(config))
            else:
                self.layer.append(BERTLayer(config))
        if config.rezero:
            for index, layer in enumerate(self.layer):
                layer.output.res_factor = nn.Parameter(
                    torch.Tensor(1).fill_(1.0).to(dtype=next(self.parameters()).dtype)
                )
                layer.output.factor = nn.Parameter(torch.Tensor(1).fill_(1).to(dtype=next(self.parameters()).dtype))
                layer.attention.output.res_factor = layer.output.res_factor
                layer.attention.output.factor = layer.output.factor
        self.config = config

    def forward(self, hidden_states, attention_mask, epoch_id=-1, head_masks=None):
        all_encoder_layers = [hidden_states]
        if epoch_id != -1:
            detach_index = int(len(self.layer) / 3) * (2 - epoch_id) - 1
        else:
            detach_index = -1
        for index, layer_module in enumerate(self.layer):
            if head_masks is None:
                if not self.config.grad_checkpoint:
                    self_out, hidden_states = layer_module(hidden_states, attention_mask, None)
                else:
                    self_out, hidden_states = torch.utils.checkpoint.checkpoint(
                        layer_module, hidden_states, attention_mask, None
                    )
            else:
                self_out, hidden_states = layer_module(hidden_states, attention_mask, head_masks[index])
            if detach_index == index:
                hidden_states.detach_()
            all_encoder_layers.append(self_out)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTEncoderRolled(nn.Module):
    def __init__(self, config):
        super(BERTEncoderRolled, self).__init__()
        layer = BERTLayer(config)
        self.config = config
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_rolled_layers)])

    def forward(self, hidden_states, attention_mask, epoch_id=-1, head_masks=None):
        all_encoder_layers = [hidden_states]
        for i in range(self.config.num_hidden_layers):
            if self.config.transformer_type.lower() == "universal":
                hidden_states = self.layer[i % self.config.num_rolled_layers](hidden_states, attention_mask)
            elif self.config.transformer_type.lower() == "albert":
                hidden_states = self.layer[i // (self.config.num_hidden_layers // self.config.num_rolled_layers)](
                    hidden_states, attention_mask
                )
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTEncoderACT(nn.Module):
    def __init__(self, config):
        super(BERTEncoderACT, self).__init__()
        self.layer = BERTLayer(config)
        p = nn.Linear(config.hidden_size, 1)
        self.p = nn.ModuleList([copy.deepcopy(p) for _ in range(config.num_hidden_layers)])
        # Following act paper, set bias init ones
        for module in self.p:
            module.bias.data.fill_(1.0)
        self.config = config
        self.act_max_steps = config.num_hidden_layers
        self.threshold = 0.99

    def should_continue(self, halting_probability, n_updates):
        return (halting_probability.lt(self.threshold).__and__(n_updates.lt(self.act_max_steps))).any()

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = [hidden_states]
        batch_size, seq_len, hdim = hidden_states.size()
        halting_probability = torch.zeros(batch_size, seq_len).cuda()
        remainders = torch.zeros(batch_size, seq_len).cuda()
        n_updates = torch.zeros(batch_size, seq_len).cuda()
        # accumulated_hidden_states = torch.zeros_like(hidden_states)
        for i in range(self.act_max_steps):
            p = torch.sigmoid(self.p[i](hidden_states).squeeze(2))
            still_running = halting_probability.lt(1.0).float()
            new_halted = (halting_probability + p * still_running).gt(self.threshold).float() * still_running
            still_running = (halting_probability + p * still_running).le(self.threshold).float() * still_running
            halting_probability = halting_probability + p * still_running
            remainders = remainders + new_halted * (1 - halting_probability)
            halting_probability = halting_probability + new_halted * remainders
            n_updates = n_updates + still_running + new_halted
            update_weights = (p * still_running + new_halted * remainders).unsqueeze(2)
            transformed_states = self.layer(hidden_states, attention_mask)
            # accumulated_hidden_states = (transformed_states * update_weights) + accumulated_hidden_states
            hidden_states = transformed_states * update_weights + hidden_states * (1 - update_weights)
            all_encoder_layers.append(hidden_states)
            if not self.should_continue(halting_probability, n_updates):
                # print(len(all_encoder_layers))
                break
        return all_encoder_layers, torch.mean(n_updates + remainders)


class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.config = config
        self.embeddings = BERTEmbeddings(config)
        if config.transformer_type.lower() == "original":
            self.encoder = BERTEncoder(config)
        elif config.transformer_type.lower() == "universal":
            self.encoder = BERTEncoderRolled(config)
        elif config.transformer_type.lower() == "albert":
            self.encoder = BERTEncoderRolled(config)
        elif config.transformer_type.lower() == "act":
            self.encoder = BERTEncoderACT(config)
        elif config.transformer_type.lower() == "textnas":
            from textnas_final import input_dict, op_dict, skip_dict

            self.encoder = TextNASEncoder(config, op_dict, input_dict, skip_dict)
        else:
            raise ValueError("Not support transformer type: {}".format(config.transformer_type.lower()))
        self.pooler = BERTPooler(config)

    def forward(
        self, input_ids, token_type_ids=None, attention_mask=None, epoch_id=-1, head_masks=None, adv_embedding=None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output, word_embeddings = self.embeddings(
            input_ids, token_type_ids, adv_embedding
        )  # if adv_embedding is None else (adv_embedding, None)
        if self.config.transformer_type.lower() == "act":
            all_encoder_layers, act_loss = self.encoder(embedding_output, extended_attention_mask)
        elif self.config.transformer_type.lower() == "reformer":
            sequence_output = self.encoder(embedding_output)
            all_encoder_layers = [sequence_output, sequence_output]
        else:
            all_encoder_layers = self.encoder(embedding_output, extended_attention_mask, epoch_id, head_masks)
        all_encoder_layers.insert(0, word_embeddings)
        sequence_output = all_encoder_layers[-1]
        if not self.config.safer_fp16:
            pooled_output = self.pooler(sequence_output)
        else:
            pooled_output = sequence_output[:, 0]
        return all_encoder_layers, pooled_output


class BertForSequenceClassificationMultiTask(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, label_list, core_encoder):
        super(BertForSequenceClassificationMultiTask, self).__init__()
        if core_encoder.lower() == "bert":
            self.bert = BertModel(config)
        elif core_encoder.lower() == "lstm":
            self.bert = LSTMModel(config)
        else:
            raise ValueError("Only support lstm or bert, but got {}".format(core_encoder))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleList()
        for label in label_list:
            self.classifier.append(nn.Linear(config.hidden_size, len(label)))
        self.label_list = label_list

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels=None,
        labels_index=None,
        epoch_id=-1,
        head_masks=None,
        adv_embedding=None,
        return_embedding=False,
        loss_weight=None,
    ):
        all_encoder_layers, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, epoch_id, head_masks, adv_embedding
        )
        pooled_output = self.dropout(pooled_output)
        logits = [classifier(pooled_output) for classifier in self.classifier]
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            regression_loss_fct = nn.MSELoss(reduction="none")
            labels_lst = torch.unbind(labels, 1)
            max_index = len(labels_lst)
            loss_lst = []
            for index, (label, logit) in enumerate(zip(labels_lst, logits)):
                if len(self.label_list[index]) != 1:
                    loss = loss_fct(logit, label.long())
                else:
                    loss = regression_loss_fct(logit.squeeze(-1), label)
                labels_mask = (labels_index == index).to(dtype=next(self.parameters()).dtype)
                if loss_weight is not None:
                    loss = loss * loss_weight[index]
                loss = torch.mean(loss * labels_mask)
                loss_lst.append(loss)
            if not return_embedding:
                return sum(loss_lst), logits
            else:
                return sum(loss_lst), logits, all_encoder_layers[0]
        else:
            return logits
