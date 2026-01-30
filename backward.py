"""
反向传播 (Backpropagation) - 仅实现反向传播
用 NumPy 实现，不依赖 PyTorch
"""

import numpy as np


# ============ 单层：线性 + Sigmoid 的反向传播 ============

def linear_sigmoid_backward(dL_da, x, z, W, b, lr=0.1):
    """
    单层反向传播

    前向（已由外部完成）: x -> z = xW + b -> a = sigmoid(z)
    输入:
        dL_da: 损失对 a 的梯度 (batch, out)
        x, z: 前向缓存的输入和线性输出
        W, b: 当前层参数
    输出: 原地更新 W, b；返回 dL_dx 供上游使用
    """
    a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    da_dz = a * (1 - a)
    dL_dz = dL_da * da_dz

    dL_dW = x.T @ dL_dz
    dL_db = np.sum(dL_dz, axis=0, keepdims=True)

    W -= lr * dL_dW
    b -= lr * dL_db

    return dL_dz @ W.T


# ============ 多层：逐层反向传播 ============

def mlp_backward(dL_dy_pred, activations, z_values, weights, biases, lr=0.1):
    """
    多层反向传播

    前向（已由外部完成）: x -> z0 -> a0 -> z1 -> a1 -> ... -> y_pred
    输入:
        dL_dy_pred: 损失对最后一层输出 y_pred 的梯度
        activations: [x, a0, a1, ..., y_pred]，前向各层输出
        z_values: [z0, z1, ...]，前向各层线性输出
        weights, biases: 各层参数
    输出: 原地更新 weights, biases
    """
    dL_da = dL_dy_pred

    for i in range(len(weights) - 1, -1, -1):
        z = z_values[i]
        a_prev = activations[i]

        if i == len(weights) - 1:
            a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            da_dz = a * (1 - a)
        else:
            da_dz = (z > 0).astype(float)

        dL_dz = dL_da * da_dz
        dL_dW = a_prev.T @ dL_dz
        dL_db = np.sum(dL_dz, axis=0, keepdims=True)

        weights[i] -= lr * dL_dW
        biases[i] -= lr * dL_db

        dL_da = dL_dz @ weights[i].T
