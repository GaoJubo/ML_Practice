import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def preprocess_iris_to_binary(data, target, feature_names):
    """
    将鸢尾花数据集（或类似数据集）的特征和标签都转化为 0/1 二元值，
    并将标签列移动到数据框的第一列。
    ... (函数内容与之前保持一致)
    """
    
    # 1. 标签二值化处理
    binary_target = np.where(target == 0, 0, 1)

    # 2. 特征二值化处理
    df_features = pd.DataFrame(data, columns=feature_names)
    df_binary_features = df_features.copy()
    
    # 使用中位数作为二值化阈值
    for col in df_binary_features.columns:
        median_val = df_binary_features[col].median()
        df_binary_features[col] = (df_binary_features[col] > median_val).astype(int)

    # 3. 组合特征和标签，并将标签提前到第一列
    df_binary_features.insert(0, 'binary_label', binary_target)
    
    # 重命名列名
    new_columns = ['binary_label'] + [f'{name}_binary' for name in feature_names]
    df_binary_features.columns = new_columns

    return df_binary_features

# --- 执行脚本和保存 ---

# 1. 加载和处理数据
iris = load_iris()
binary_iris_df = preprocess_iris_to_binary(iris.data, iris.target, iris.feature_names)

# 2. 将 DataFrame 转换为 NumPy 数组
# .values 属性会将 DataFrame 转换为底层的 NumPy 数组
binary_iris_array = binary_iris_df.values

# 3. 定义保存的文件名
filename = 'binary_iris_dataset.npy'

# 4. 使用 numpy.save() 保存数组到 .npy 文件
try:
    np.save(filename, binary_iris_array)
    print(f"\n数据集已成功保存为 {filename} 文件。")
    print(f"保存的数组形状: {binary_iris_array.shape}")
except Exception as e:
    print(f"保存文件时发生错误: {e}")

# 可选：加载并验证保存的文件
print("\n--- 验证文件加载 ---")
loaded_array = np.load(filename)
print(f"成功加载文件 {filename}。")
print(f"加载后的数组形状: {loaded_array.shape}")
print("加载后的数据（前 5 行）：")
print(loaded_array[:5])