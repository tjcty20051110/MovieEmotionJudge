%%1 关于项目环境的说明
本项目具体运行的环境为windows11,nvidia-drive=538.20,cuda=12.1,python=3.10.10
请根据自己的电脑驱动与cuda版本在pytorch官网选择合适的pytorch版本进行下载
例如本项目：pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
检查是否支持cuda,可以通过下列指令：
import torch
print(torch.cuda.is_available())
若结果为true,则当前版本支持cuda，否则不支持
不支持的话可以改为使用cpu进行计算，不过会很慢
%%2 关于项目结构的说明
# MovieEmotionJudge 项目结构说明
`MovieEmotionJudge` 是一个基于循环神经网络（RNN）的电影评论情感分析项
目，利用IMDB数据集训练模型，实现对电影评论“正面/负面”情感的自动判断。
以下是项目的文件与目录结构说明：
## 目录与文件结构
```
MovieEmotionJudge/
├── .venv/                  # Python虚拟环境（隔离项目依赖）
├── aclImdb_v1/             # IMDB电影评论数据集目录
│   ├── aclImdb/            # 数据集核心目录
│   │   ├── test/           # 测试集（含“正面/负面”评论子目录）
│   │   ├── train/          # 训练集（含“正面/负面”评论子目录）
│   │   ├── imdb.vocab      # IMDB数据集词汇表（记录出现的单词）
│   │   ├── imdbEr.txt      # 数据集辅助说明文件
│   │   └── README          # 数据集官方说明
│   └── aclImdb_v1.tar.gz   # 数据集原始压缩包（备份用）
├── config.py               # 项目全局配置（设备、训练参数、模型路径等）
├── data_processor.py       # 数据预处理（加载、分词、构建数据加载器）
├── demo.py                 # 功能演示脚本（快速验证核心逻辑）
├── glove.6B.100d.txt       # 预训练GloVe词向量（100维，优化嵌入层）
├── main.py                 # 主程序入口（训练、评估、可视化全流程）
├── model.py                # 模型定义（RNN/LSTM/GRU等情感分析模型）
├── predictor.py            # 预测模块（加载模型，对新评论做情感预测）
├── readme.md               # 项目说明文档（当前文件）
├── requirement.txt         # 依赖清单（项目所需Python库）
├── trainer.py              # 训练逻辑（训练、验证函数 + 模型保存）
└── tut2-model.pt           # 训练好的模型权重文件（最佳参数备份）
```
## 核心文件/目录说明
### 1. 虚拟环境：`.venv/`，也可采用conda
- 用于隔离项目依赖，避免与系统Python环境冲突。
- 激活后可通过 `pip install -r requirement.txt` 一键安装所有依赖。
### 2. 数据集：`aclImdb_v1/` + `glove.6B.100d.txt`
- `aclImdb_v1/aclImdb/`：IMDB电影评论数据集，包含**训练集**（`train/`）和**测试集**（`test/`），每条评论已标注“正面/负面”情感。
- `glove.6B.100d.txt`：预训练的**GloVe词向量**（100维），用于初始化模型的“词嵌入层”，提升情感分析的准确性。
### 3. 配置与数据处理
- `config.py`：存储项目**全局配置**，如：
  - `DEVICE`：指定计算设备（CPU/GPU）；
  - `N_EPOCHS`：训练轮数；
  - `MODEL_SAVE_PATH`：模型权重保存路径。
- `data_processor.py`：负责**数据预处理**，包括：
  - 加载IMDB原始文本；
  - 分词、构建词表；
  - 生成PyTorch数据加载器（`train_loader`、`valid_loader`等）。
### 4. 模型与训练
- `model.py`：定义**情感分析模型结构**，通常基于RNN（如LSTM/GRU），包含“词嵌入层→循环层→分类输出层”。
- `trainer.py`：实现**训练逻辑**，包括：
  - `train`：单轮训练函数；
  - `evaluate`：验证/评估函数；
  - `run_training`：完整训练流程（多轮训练、模型保存、指标记录）。
- `tut2-model.pt`：训练完成后保存的**最佳模型权重**，可直接用于预测或部署。
### 5. 主程序与预测
- `main.py`：项目**主入口**，整合“训练→评估→结果可视化”全流程（如绘制损失曲线、准确率曲线）。
- `predictor.py`：**预测模块**，加载训练好的模型，对新的电影评论文本做“正面/负面”情感预测。
### 6. 辅助文件
- `requirement.txt`：列出项目所有**Python依赖库**（如`torch`、`matplotlib`、`spacy`等），执行 `pip install -r requirement.txt` 可一键安装。
- `demo.png`：事例训练损失图。
项目通过以上结构，实现了“数据预处理→模型定义→训练验证→预测部署”的完整机器学习工作流，便于维护与功能扩展。