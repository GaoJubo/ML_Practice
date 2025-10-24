import pandas as pd
import numpy as np
# 泰坦尼克号数据集通常作为教学数据，可从 seaborn 中获取
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder

def preprocess_titanic_to_binary(df):
    """
    对泰坦尼克号数据集进行清洗，将所有特征和标签转化为 0/1 二元值，
    并将标签列移动到数据框的第一列。

    Args:
        df (pd.DataFrame): 原始泰坦尼克号数据集。

    Returns:
        np.ndarray: 处理后的包含 0/1 值的 NumPy 数组，标签在第一列。
    """
    
    # --- 1. 特征选择 ---
    # 选择关键特征。像 'deck', 'alone', 'class', 'who', 'adult_male' 等是衍生或不完整特征，我们不使用。
    selected_features = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    df_clean = df[selected_features].copy()

    # --- 2. 缺失值处理 ---
    # 填充 'age'（年龄）：使用中位数填充
    df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
    
    # 填充 'fare'（票价）：使用中位数填充
    df_clean['fare'].fillna(df_clean['fare'].median(), inplace=True)

    # 填充 'embarked'（登船港口）：使用众数（最常见的那个）填充
    # 因为 'embarked' 是类别特征，使用众数填充更合理
    df_clean['embarked'].fillna(df_clean['embarked'].mode()[0], inplace=True)
    
    print(f"处理缺失值后，数据集形状: {df_clean.shape}")

    # --- 3. 数值特征二值化（使用中位数作为阈值）---
    numerical_cols = ['age', 'fare', 'sibsp', 'parch']
    print("\n数值特征二值化阈值 (中位数):")
    for col in numerical_cols:
        median_val = df_clean[col].median()
        print(f"  {col}: {median_val:.2f}")
        # 如果特征值 > 中位数则为 1，否则为 0
        df_clean[col] = (df_clean[col] > median_val).astype(int)
        df_clean.rename(columns={col: f'{col}_binary_median'}, inplace=True)

    # --- 4. 类别特征二值化（独热编码 One-Hot Encoding）---
    # 'sex', 'pclass', 'embarked'
    
    # 对于 'sex'（性别），只有 male/female 两个值，LabelEncoder 也可以，但 One-Hot 更通用
    df_clean = pd.get_dummies(df_clean, columns=['sex', 'embarked', 'pclass'], drop_first=True, dtype=int)
    
    # 'drop_first=True' 的目的是避免多重共线性，这使得每个类别特征都转化为 0/1 的二元特征。

    # --- 5. 准备最终数据集和重新排列 ---
    
    # 提取标签 (survived)
    label = df_clean['survived']
    
    # 删除标签列
    df_features_final = df_clean.drop('survived', axis=1)
    
    # 插入标签到第一列
    df_features_final.insert(0, 'binary_label', label)
    
    # 转换为 NumPy 数组
    binary_titanic_array = df_features_final.values
    
    print("\n处理后的特征列名（不含标签）：")
    print(df_features_final.columns[1:].tolist())

    return binary_titanic_array

# --- 执行脚本和保存 ---

# 1. 加载泰坦尼克号数据集
titanic_df = sns.load_dataset('titanic')

# 2. 调用函数进行处理
binary_titanic_array = preprocess_titanic_to_binary(titanic_df)

# 3. 定义保存的文件名
filename = 'binary_titanic_dataset.npy'

# 4. 使用 numpy.save() 保存数组到 .npy 文件
try:
    np.save(filename, binary_titanic_array)
    print(f"\n数据集已成功保存为 {filename} 文件。")
    print(f"保存的数组形状: {binary_titanic_array.shape}")
except Exception as e:
    print(f"保存文件时发生错误: {e}")

# 可选：加载并验证保存的文件
print("\n--- 验证文件加载 ---")
loaded_array = np.load(filename)
print(f"成功加载文件 {filename}。")
print(f"加载后的数组形状: {loaded_array.shape}")
print("加载后的数据（前 5 行）：")
print(loaded_array[:5])