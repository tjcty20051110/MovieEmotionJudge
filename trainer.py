# trainer.py
import torch
import time
import torch.nn as nn
import torch.optim as optim
from config import DEVICE, N_EPOCHS, MODEL_SAVE_PATH
from data_processor import train_loader, valid_loader

def binary_accuracy(preds, y):
    """计算二分类准确率"""
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)


def epoch_time(start_time, end_time):
    """计算每轮训练耗时"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion):
    """训练模型"""
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        labels, text, text_lengths = batch
        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    """评估模型"""
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            labels, text, text_lengths = batch
            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def run_training(model):
    """执行完整训练流程"""
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    model = model.to(DEVICE)

    best_valid_loss = float('inf')
    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # 打印结果
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Acc: {train_acc * 100:.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Acc: {valid_acc * 100:.2f}%')

    return train_losses, train_accs, valid_losses, valid_accs