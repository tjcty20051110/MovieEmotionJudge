# main.py
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from config import MODEL_SAVE_PATH, DEVICE
from data_processor import (
    vocab, embedding_matrix, train_loader,
    valid_loader, test_loader
)
from model import RNN, count_parameters
from trainer import run_training, evaluate

def main():
    # 初始化模型
    model = RNN(
        vocab_size=len(vocab),
        embedding_matrix=embedding_matrix
    )
    print(f'模型可训练参数: {count_parameters(model):,}')

    # 训练模型
    train_losses, train_accs, valid_losses, valid_accs = run_training(model)

    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'r-', label='train_loss')
    plt.plot(valid_losses, 'b-', label='val_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'r-', label='train_acc')
    plt.plot(valid_accs, 'b-', label='val_acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

    # 测试集评估
    #添加weights_only=True这个参数，提升模型安全性，避免报错
    model.load_state_dict(torch.load(MODEL_SAVE_PATH,weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, nn.BCEWithLogitsLoss().to(DEVICE))
    print(f'\ntest_result:loss={test_loss:.3f} | accuracy={test_acc*100:.2f}%')

if __name__ == "__main__":
    main()