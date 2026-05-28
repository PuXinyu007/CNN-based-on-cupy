import argparse
import os

import cupy as cp

from config import config, lr_decay
from model import CNN
from optimizer import Adam
from utils import load_emnist_csv
from visualize import plot_curves, visualize_all_channels


def parse_args():
    parser = argparse.ArgumentParser(description="CNN 训练 EMNIST")
    parser.add_argument('--epochs', type=int, default=config['num_epochs'], help='训练轮数')
    parser.add_argument('--lr', type=float, default=config['lr'], help='学习率')
    parser.add_argument('--batch-size', type=int, default=config['batch_size'], help='批次大小')
    parser.add_argument('--visualize', action='store_true', help='是否保存可视化图表')
    parser.add_argument('--data-path', type=str, default=config['data_path'], help='数据集路径')
    return parser.parse_args()


def train(args):
    X, T = load_emnist_csv(args.data_path, num_classes=26)
    ratio = 0.95
    split = int(ratio * X.shape[0])
    idx = cp.random.permutation(X.shape[0])
    X = X[idx]
    T = T[idx]
    X_train, X_test = X[0:split], X[split:]
    T_train, T_test = T[0:split], T[split:]

    model = CNN()
    optimizer = Adam(model, lr=args.lr, weight_decay=config['weight_decay'])

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    num_samples = X_train.shape[0]
    milestones = config['milestones']

    print(f"开始训练: epochs={args.epochs}  lr={args.lr}  batch_size={args.batch_size} \n")

    for epoch in range(args.epochs):
        shuffled_indices = cp.random.permutation(num_samples)
        X_train_s = X_train[shuffled_indices]
        T_train_s = T_train[shuffled_indices]

        epoch_loss = 0
        num_batches = (num_samples + args.batch_size - 1) // args.batch_size
        
        model.train()

        for i in range(num_batches):
            start_idx = i * args.batch_size
            end_idx = min(start_idx + args.batch_size, num_samples)

            x_batch = X_train_s[start_idx:end_idx]
            t_batch = T_train_s[start_idx:end_idx]

            loss = model.loss(x_batch, t_batch)
            epoch_loss += float(loss) * (end_idx - start_idx)

            model.backward()
            optimizer.update()

        if epoch + 1 in milestones:
            optimizer.lr *= lr_decay

        epoch_loss /= num_samples

        model.eval()
        _, train_acc = model.evaluate(X_train[:2000], T_train[:2000])
        train_acc = float(train_acc)

        test_loss, test_acc = model.evaluate(X_test, T_test)
        test_loss = float(test_loss)
        test_acc = float(test_acc)

        train_losses.append(epoch_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {epoch_loss:.4f} - Test Acc: {test_acc*100:.2f}%")


    os.makedirs('results', exist_ok=True)
    plot_curves(
        train_losses, test_losses, train_accs, test_accs,
        num_epochs=args.epochs,
        save_path='results/loss_acc.png'
    )

    sample = X_test[0].reshape(1, 1, 28, 28)
    model.eval()
    visualize_all_channels(model, sample, save_path='results/feature_maps.png')

    return model, test_accs[-1]


if __name__ == '__main__':
    args = parse_args()
    train(args)
