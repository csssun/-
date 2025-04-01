# 导入必要的库
import numpy as np
from collections import Counter

# 定义 KNN 算法类
class KNN:
    def __init__(self, k=3):
        self.k = k  # 设置 K 值，默认为 3

    def fit(self, X_train, y_train):
        self.X_train = X_train  # 训练集特征
        self.y_train = y_train  # 训练集标签

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]  # 对每个测试样本进行预测
        return np.array(predictions)

    def _predict(self, x):
        # 计算测试样本与所有训练样本的距离
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        # 获取距离最近的 K 个样本的索引
        k_indices = np.argsort(distances)[:self.k]
        # 获取 K 个最近邻样本的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 使用多数投票法确定预测标签
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        # 计算欧氏距离
        return np.sqrt(np.sum((x1 - x2) ** 2))


# 示例数据
X_train = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2]
])
y_train = np.array(['Setosa', 'Setosa', 'Versicolor', 'Setosa', 'Versicolor'])
X_test = np.array([[5.1, 3.5, 1.4, 0.2]])  # 待分类样本

# 标准化处理
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# 创建 KNN 模型并训练
knn = KNN(k=3)
knn.fit(X_train, y_train)

# 进行预测
predictions = knn.predict(X_test)
print("预测结果：", predictions)
