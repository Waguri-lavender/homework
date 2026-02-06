#!/usr/bin/env python3
"""
第三项作业：用 NumPy 实现两层神经网络，在 MNIST 上达到 85% 以上准确率
脚本模式运行: python mnist_nn.py [参数]
"""

import argparse
import gzip
import os
import struct

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============ 激活函数 ============

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    x = np.clip(x, -500, 500)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ============ 两层神经网络 ============

class TwoLayerNet:
    """两层神经网络: 784 -> hidden -> 10"""

    def __init__(self, hidden_size=256, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(784, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 10) * 0.01
        self.b2 = np.zeros((1, 10))

    def forward(self, x):
        """前向传播"""
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.y_pred = softmax(self.z2)
        return self.y_pred

    def backward(self, y_true, lr=0.1):
        """
        反向传播
        交叉熵 + Softmax: dL/dz2 = y_pred - y_true (one-hot)
        """
        batch_size = y_true.shape[0]
        dL_dz2 = (self.y_pred - y_true) / batch_size

        dL_dW2 = self.a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        dL_da1 = dL_dz2 @ self.W2.T

        dL_dz1 = dL_da1 * relu_derivative(self.z1)
        dL_dW1 = self.x.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        self.W2 -= lr * dL_dW2
        self.b2 -= lr * dL_db2
        self.W1 -= lr * dL_dW1
        self.b1 -= lr * dL_db1


# ============ 训练与评估 ============

def _read_mnist_images(filepath):
    """读取 MNIST 图像文件"""
    with gzip.open(filepath, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)

def _read_mnist_labels(filepath):
    """读取 MNIST 标签文件"""
    with gzip.open(filepath, "rb") as f:
        struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def load_mnist(data_dir="mnist_data"):
    """
    加载 MNIST 数据集（从本地目录）
    data_dir: MNIST 数据目录，应包含 4 个 .gz 文件
    """
    print("加载 MNIST...")
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    
    paths = [os.path.join(data_dir, f) for f in files]
    
    # 检查文件是否存在
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}\n请确保 MNIST 数据集在 {data_dir} 目录中")
    
    X_train = _read_mnist_images(paths[0]) / 255.0
    y_train = _read_mnist_labels(paths[1])
    X_test = _read_mnist_images(paths[2]) / 255.0
    y_test = _read_mnist_labels(paths[3])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test


def one_hot(y, num_classes=10):
    n = len(y)
    oh = np.zeros((n, num_classes))
    oh[np.arange(n), y] = 1
    return oh


def cross_entropy_loss(y_pred, y_true_onehot):
    """交叉熵损失"""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))


def train(model, X_train, y_train, epochs=50, batch_size=128, lr=0.1):
    """训练，返回 (train_losses, train_accs) 用于可视化"""
    n = len(X_train)
    y_oh = one_hot(y_train)
    train_losses, train_accs = [], []

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x_batch = X_train[idx]
            y_batch = y_oh[idx]

            model.forward(x_batch)
            model.backward(y_batch, lr=lr)

        pred = model.forward(X_train)
        loss = cross_entropy_loss(pred, y_oh)
        acc = np.mean(np.argmax(pred, axis=1) == y_train)
        train_losses.append(loss)
        train_accs.append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss:.4f}, 训练准确率: {acc:.4f}")

    return train_losses, train_accs


def plot_training_curves(train_losses, train_accs, save_path=None):
    """绘制训练曲线：损失 + 准确率"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, color="#2E86AB", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("训练损失")
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, color="#E94F37", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("训练准确率")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"训练曲线已保存: {save_path}")
    plt.show()


def evaluate(model, X_test, y_test):
    """评估"""
    pred = model.forward(X_test)
    acc = np.mean(np.argmax(pred, axis=1) == y_test)
    return acc


# ============ 主程序（脚本模式） ============

def main():
    parser = argparse.ArgumentParser(description="两层神经网络 MNIST 分类（目标准确率 ≥85%）")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=0.15, help="学习率")
    parser.add_argument("--hidden", type=int, default=512, help="隐藏层神经元数")
    parser.add_argument("--data-dir", type=str, default="mnist_data", help="MNIST 数据目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save-plot", type=str, default=None, help="保存训练曲线到文件")
    parser.add_argument("--no-plot", action="store_true", help="不显示训练曲线")
    args = parser.parse_args()

    print("=" * 50)
    print("第三项作业：两层神经网络 MNIST")
    print("=" * 50)
    print(f"参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, hidden={args.hidden}")

    X_train, y_train, X_test, y_test = load_mnist(data_dir=args.data_dir)
    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    model = TwoLayerNet(hidden_size=args.hidden, seed=args.seed)
    print("\n训练中...")
    train_losses, train_accs = train(
        model, X_train, y_train,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )

    acc = evaluate(model, X_test, y_test)
    print(f"\n测试准确率: {acc:.4f} ({acc*100:.2f}%)")
    print("达标!" if acc >= 0.85 else "未达 85%")

    if not args.no_plot:
        plot_training_curves(
            train_losses, train_accs,
            save_path=args.save_plot
        )

    return acc


if __name__ == "__main__":
    main()
