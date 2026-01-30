# 深度学习作业项目

本项目包含三项作业：激活函数介绍、反向传播实现、两层神经网络 MNIST 分类。

---

## 项目结构

```
work_1/
├── README.md           # 本说明文档
├── requirements.txt    # 依赖列表
├── 激活函数介绍.md     # 作业一：ReLU 变体及其他激活函数
├── backward.py         # 作业二：反向传播实现
├── mnist_nn.py        # 作业三：两层神经网络 + MNIST
└── mnist_data/        # MNIST 数据集缓存（自动下载）
```

---

## 作业概览

| 作业 | 内容 | 对应文件 |
|------|------|----------|
| 一 | ReLU 变体、其他激活函数及编程实现 | `激活函数介绍.md` |
| 二 | 反向传播（单层/多层）NumPy 实现 | `backward.py` |
| 三 | 两层神经网络，MNIST 分类，目标准确率 ≥85% | `mnist_nn.py` |

---

## 环境准备

### 1. 安装依赖

```bash
cd /Users/han/work_1
pip install -r requirements.txt
```

使用国内镜像（推荐）：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 依赖说明

- `numpy`：数值计算
- `scikit-learn`：数据预处理、MNIST 备选加载方式

---

## 执行流程

### 作业一：激活函数

阅读文档即可：

```bash
# 打开查看
open 激活函数介绍.md
```

内容包括：ReLU、Leaky ReLU、PReLU、ELU、SELU、GELU、Swish、Mish、Sigmoid、Tanh、Softmax、Maxout 等，以及各自的 NumPy 实现。

---

### 作业二：反向传播

```bash
python backward.py
```

**功能**：
- `linear_sigmoid_backward()`：单层线性 + Sigmoid 的反向传播
- `mlp_backward()`：多层网络的反向传播

**输入**：上游梯度 `dL_da`，前向缓存的 `x`, `z`, `W`, `b` 等  
**输出**：更新后的参数，以及传给上一层的梯度

---

### 作业三：MNIST 两层神经网络

```bash
# 默认参数运行
python mnist_nn.py

# 脚本模式：自定义参数
python mnist_nn.py --epochs 50 --hidden 256 --lr 0.1
python mnist_nn.py --help   # 查看所有参数
```

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 80 | 训练轮数 |
| `--batch-size` | 128 | 批大小 |
| `--lr` | 0.15 | 学习率 |
| `--hidden` | 512 | 隐藏层神经元数 |
| `--data-dir` | mnist_data | MNIST 数据目录 |
| `--no-mirror` | - | 使用 sklearn 加载（不用镜像） |
| `--seed` | 42 | 随机种子 |
| `--save-plot` | - | 保存训练曲线到文件（如 `train_curves.png`） |
| `--no-plot` | - | 不显示训练曲线（适用于无图形界面环境） |

**训练可视化**：训练结束后自动显示损失曲线和准确率曲线。使用 `--save-plot train_curves.png` 可保存图片。

**流程**：

1. **加载 MNIST**
   - 优先从镜像下载（AWS / Google / 官方）
   - 缓存到 `mnist_data/`，下次直接使用
   - 若镜像不可用，可手动下载到 `mnist_data/`（见下方）

2. **训练**
   - 网络：784 → 512(ReLU) → 10(Softmax)
   - 交叉熵损失 + 反向传播
   - 约 80 个 epoch，batch_size=128

3. **评估**
   - 在测试集上计算准确率
   - 目标：≥ 85%

**输出示例**：

```
加载 MNIST...
训练集: 60000, 测试集: 10000

训练中...
  Epoch 10, 训练准确率: 0.xxxx
  ...
  Epoch 80, 训练准确率: 0.xxxx

测试准确率: 0.xxxx (xx.xx%)
达标!
```

---

## MNIST 数据说明

### 自动下载（默认）

首次运行会从以下镜像依次尝试下载：

1. AWS：`https://ossci-datasets.s3.amazonaws.com/mnist/`
2. Google：`https://storage.googleapis.com/cvdf-datasets/mnist/`
3. 官方：`http://yann.lecun.com/exdb/mnist/`

### 手动下载

若自动下载失败，可从 [阿里云天池](https://tianchi.aliyun.com/dataset/206667) 下载 MNIST，将以下 4 个 `.gz` 文件放入 `mnist_data/` 目录：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

### 使用 sklearn 加载

若希望使用 `sklearn.datasets.fetch_openml` 加载：

```python
X_train, y_train, X_test, y_test = load_mnist(use_mirror=False)
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 运行反向传播示例
python backward.py

# 3. 运行 MNIST 训练（首次会下载数据）
python mnist_nn.py
```
