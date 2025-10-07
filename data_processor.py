# data_processor.py
# 数据处理模块，负责数据集加载、词汇表构建和数据迭代器创建
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset  # HuggingFace datasets库
import spacy
from spacy.tokenizer import Tokenizer
import os
from config import SEED,MAX_VOCAB_SIZE,BATCH_SIZE,VECTOR_PATH,PAD_TOKEN,UNK_TOKEN,DEVICE,EMBEDDING_DIM
# 配置参数
# 设置随机种子确保可重复性
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# 初始化分词器
try:
    # 尝试加载spaCy模型
    nlp = spacy.load("en_core_web_sm")
    tokenizer = Tokenizer(nlp.vocab)
except:
    # 安装基础英文模型（如果需要）
    print("Installing spaCy English model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    tokenizer = Tokenizer(nlp.vocab)

    # 或者使用简单的空格分词作为备选
    def simple_tokenize(text):
        return text.lower().split()
    tokenizer = simple_tokenize


class TextDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, data, vocab=None, is_train=True):
        self.data = data
        self.vocab = vocab
        self.is_train = is_train
        # 提取文本和标签
        self.texts = [item['text'] for item in self.data]
        self.labels = [1 if item['label'] == 'positive' else 0 for item in self.data]

        # 如果是训练集且没有词汇表，则构建词汇表
        if is_train and self.vocab is None:
            self.vocab = self.build_vocab()

    def build_vocab(self):
        """构建词汇表"""
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        word_counts = {}

        # 统计词频
        for text in self.texts:
            tokens = tokenizer(text) if callable(tokenizer) else tokenizer.tokenize(text)
            for token in tokens:
                if token not in word_counts:
                    word_counts[token] = 1
                else:
                    word_counts[token] += 1

        # 按词频排序并添加到词汇表（限制最大词汇量）
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:MAX_VOCAB_SIZE-2]:  # 预留两个特殊符号的位置
            vocab[word] = len(vocab)

        return vocab

    def text_to_indices(self, text):
        """将文本转换为索引序列"""
        tokens = tokenizer(text) if callable(tokenizer) else tokenizer.tokenize(text)
        return [self.vocab.get(token, self.vocab[UNK_TOKEN]) for token in tokens]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = self.text_to_indices(text)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def collate_batch(batch):
    """自定义批处理函数"""
    text_list, label_list, length_list = [], [], []
    for text_indices, label in batch:
        label_list.append(label)
        text_list.append(text_indices)
        length_list.append(torch.tensor(len(text_indices), dtype=torch.long))

    # 填充序列
    text_padded = pad_sequence(text_list, padding_value=0, batch_first=False)  # 0是PAD_TOKEN的索引
    return (
        torch.stack(label_list).to(DEVICE),
        text_padded.to(DEVICE),
        torch.stack(length_list).to(DEVICE)
    )


def get_imdb_data():
    """加载IMDB数据集"""
    dataset = load_dataset("imdb")
    train_data = dataset["train"]
    test_data = dataset["test"]
    return train_data, test_data


# 加载数据集
train_data, test_data = get_imdb_data()

# 创建训练数据集并构建词汇表
train_dataset = TextDataset(train_data, is_train=True)
vocab = train_dataset.vocab

# 创建验证集和测试集
train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = random_split(
    train_dataset, [train_size, valid_size],
    generator=torch.Generator().manual_seed(SEED)
)

# 确保验证集和测试集使用相同的词汇表
valid_dataset = TextDataset(train_data.select(range(valid_size)), vocab=vocab, is_train=False)
test_dataset = TextDataset(test_data, vocab=vocab, is_train=False)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
)


def load_glove_vectors(glove_path):
    """加载GloVe词向量并构建嵌入矩阵"""
    word_to_vec = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < EMBEDDING_DIM + 1:
                continue
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:1+EMBEDDING_DIM]], dtype=np.float32)
            word_to_vec[word] = vec

    # 构建嵌入矩阵
    embedding_matrix = torch.zeros(len(vocab), EMBEDDING_DIM)
    for word, idx in vocab.items():
        if word in word_to_vec:
            embedding_matrix[idx] = torch.from_numpy(word_to_vec[word])
    return embedding_matrix


# 加载词向量
embedding_matrix = load_glove_vectors(VECTOR_PATH)

# 暴露给外部的变量
__all__ = [
    "vocab", "embedding_matrix", "train_loader",
    "valid_loader", "test_loader", "tokenizer", "DEVICE"
]

