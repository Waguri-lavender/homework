import numpy as np

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
    a = 1 / (1 + np.exp(-z))
    da_dz = a * (1 - a)
    dL_dz = dL_da * da_dz

    dL_dW = x.T @ dL_dz
    dL_db = np.sum(dL_dz, axis=0, keepdims=True)

    W -= lr * dL_dW
    b -= lr * dL_db

    return dL_dz @ W.T
