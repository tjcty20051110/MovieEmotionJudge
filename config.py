# config.py
#集中管理所有超参数和路径，便于后续直接修改
import torch

# 随机种子（保证实验可复现）
SEED = 1234

# 数据处理参数
MAX_VOCAB_SIZE = 25_000  # 词汇表最大规模
BATCH_SIZE = 64          # 批处理大小
VECTOR_PATH = "glove.6B.100d.txt"  # 本地GloVe词向量路径
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# 模型参数
EMBEDDING_DIM = 100      # 词嵌入维度（需与GloVe一致）
HIDDEN_DIM = 256         # LSTM隐藏层维度
OUTPUT_DIM = 1           # 输出维度（二分类）
N_LAYERS = 2             # LSTM层数
BIDIRECTIONAL = True     # 是否双向LSTM
DROPOUT = 0.5            # Dropout比例

# 训练参数
N_EPOCHS = 8            # 训练轮数
MODEL_SAVE_PATH = "tut2-model.pt"  # 模型保存路径

# 设备配置（自动选择GPU/CPU）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定随机种子
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True