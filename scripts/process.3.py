import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

# 1. 加载真实数据集（Diabetes）
# 这个数据集包含 10 个特征，我们将只选择其中一个
diabetes = load_diabetes()

# 选择一个特征作为自变量 X (例如：bmi, 身体质量指数，索引为 2)
# X 是一个 (442, 1) 的二维数组
X = diabetes.data[:, np.newaxis, 2] 

# y 是因变量/标签 (疾病进展的定量指标)
# y 是一个 (442,) 的一维数组
y = diabetes.target

# 将 y 转换为 (442, 1) 的二维数组，以便于与 X 堆叠
y = y.reshape(-1, 1) 

# 2. 检查数据形状
print(f"原始特征 X 的形状: {X.shape}")
print(f"原始标签 y 的形状: {y.shape}")

# 3. 将标签列 (y) 提升到第一列
# 最终的数据集结构为: [标签 y | 特征 X]
dataset_with_y_first = np.hstack((y, X))

# 4. 检查最终数据集的形状和内容
print(f"\n最终数据集的形状: {dataset_with_y_first.shape}")
print(f"数据格式: [标签, 特征]")
print(f"前5行数据:\n{dataset_with_y_first[:5]}")

# 5. 保存为 .npy 文件
file_name = 'real_world_single_regression_diabetes_bmi.npy'
np.save(file_name, dataset_with_y_first)

print(f"\n真实数据集已成功保存到文件: {file_name}")