# predictor.py
import torch
from config import MODEL_SAVE_PATH, DEVICE, EMBEDDING_DIM
from data_processor import vocab, embedding_matrix, tokenizer
from model import RNN


def load_model():
    """加载训练好的模型"""
    model = RNN(
        vocab_size=len(vocab),
        embedding_matrix=embedding_matrix
    )
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()  # 切换到评估模式
    return model


def predict_sentiment(model, sentence):
    """预测单条文本的情感（返回0-1之间的分数，越接近1越正面）"""
    with torch.no_grad():
        # 文本预处理：分词→转索引
        tokenized = tokenizer(sentence)
        indexed = [vocab[token] for token in tokenized]
        length = torch.tensor([len(indexed)], dtype=torch.long).to(DEVICE)

        # 转换为张量并添加批次维度（[句子长度, 1]）
        tensor = torch.LongTensor(indexed).to(DEVICE).unsqueeze(1)

        # 预测并计算概率
        prediction = torch.sigmoid(model(tensor, length))
        return prediction.item()


if __name__ == "__main__":
    # 加载模型
    model = load_model()

    # 预测示例
    print("情感预测示例（值越接近1表示越正面）：")
    test_sentences = [
        "This film is terrible. The acting was poor and the plot was confusing.",
        "This film is great! I loved the characters and the story was amazing.",
        "An okay movie. Not the best, but not the worst either."
    ]

    for sentence in test_sentences:
        score = predict_sentiment(model, sentence)
        print(f"句子: {sentence}")
        print(f"预测分数: {score:.4f}\n")