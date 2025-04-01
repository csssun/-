import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=10):
        self.lr = learning_rate  # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        self.weights = None  # 特征权重
        self.bias = 0.0  # 偏置项
        self.mu = None  # 特征均值
        self.sigma = None  # 特征标准差

    def _standardize(self, X):
        """Z-score 标准化"""
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)
        return (X - self.mu) / self.sigma

    def _sigmoid(self, z):
        """Sigmoid 函数"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # 数据标准化
        X_std = self._standardize(X)

        # 参数初始化
        n_features = X.shape[1]
        self.weights = np.array([0.5, -0.3, -0.8, 1.2])  # 初始权重
        self.bias = 0.1  # 初始偏置

        # 梯度下降迭代
        for iter in range(self.max_iter):
            # 前向传播
            linear = np.dot(X_std, self.weights) + self.bias
            y_pred = self._sigmoid(linear)

            # 计算梯度
            dw = np.dot(X_std.T, (y_pred - y))  # 特征权重梯度
            db = np.sum(y_pred - y)  # 偏置项梯度

            # 参数更新
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # 标准化新数据
        X_std = (X - self.mu) / self.sigma
        # 计算概率
        z = np.dot(X_std, self.weights) + self.bias
        return self._sigmoid(z)


# 训练数据
X_train = np.array([
    [28, 1.2, 3, 45],
    [35, 2.8, 5, 32],
    [42, 0.9, 2, 68],
    [53, 3.5, 10, 25],
    [31, 1.5, 4, 55]
])
y_train = np.array([0, 0, 1, 0, 1])

# 模型训练
model = LogisticRegression()
model.fit(X_train, y_train)

# 新样本预测（客户006）
X_new = np.array([[45, 2.0, 6, 40]])
prob = model.predict(X_new)[0]
pred = 1 if prob >= 0.5 else 0

# 输出结果
print(f"训练后参数权重： {model.weights.round(3)}")
print(f"训练后偏置项： {model.bias:.3f}")
print(f"新样本预测概率： {prob:.3f}")
print(f"分类结果： {pred}")
