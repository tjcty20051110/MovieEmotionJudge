# model.py
import torch
import torch.nn as nn
from config import (
    EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
    N_LAYERS, BIDIRECTIONAL, DROPOUT, DEVICE
)


def count_parameters(model):
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_matrix):
        super().__init__()

        # 嵌入层（使用预训练GloVe向量初始化）
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            padding_idx=embedding_matrix.shape[0] - 1,  # PAD_TOKEN的索引
            freeze=False  # 允许微调嵌入层
        )

        # LSTM层
        self.rnn = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT,
            batch_first=False
        )

        # 全连接层（双向LSTM需×2）
        self.fc = nn.Linear(HIDDEN_DIM * 2, OUTPUT_DIM)

        # Dropout层
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, text, text_lengths):
        # text: [句子长度, batch_size]
        embedded = self.dropout(self.embedding(text))  # [句子长度, batch_size, 嵌入维度]

        # 打包变长序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(),  # 长度需在CPU上
            batch_first=False,
            enforce_sorted=False  # 已在collate_fn中排序
        )

        # LSTM输出
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # 拼接最后一层的双向隐藏状态
        hidden = self.dropout(torch.cat(
            (hidden[-2, :, :], hidden[-1, :, :]),  # 最后两层（双向）
            dim=1
        ))  # [batch_size, 2*隐藏维度]

        return self.fc(hidden)  # [batch_size, 1]