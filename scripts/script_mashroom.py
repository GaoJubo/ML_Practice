import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. 定义数据文件的路径和列名 ---

COLUMN_NAMES = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat'
]

FILE_PATH = r'E:\someShy\ML_Practice\scripts\agaricus-lepiota.csv' 
OUTPUT_CSV_FILE = 'mushroom_encoded_data.csv'
OUTPUT_NPY_FILE = 'mushroom_encoded_data.npy'

# --- 2. 数据加载和缺失值处理 ---

def load_and_clean_data(file_path, column_names):
    """加载数据，并处理缺失值。"""
    try:
        df = pd.read_csv(file_path, header=None, names=column_names)
    except FileNotFoundError:
        print(f"错误：文件未找到在路径 {file_path}。请检查文件路径。")
        return None

    print(f"原始数据集形状: {df.shape}")

    # 缺失值处理：在 'stalk-root' 列中，缺失值被编码为 '?'
    MISSING_VALUE = '?'
    
    # 用众数 (mode) 替换缺失值
    mode_value = df['stalk-root'].mode()[0]
    df['stalk-root'] = df['stalk-root'].replace(MISSING_VALUE, mode_value)
    
    print(f"处理后，'stalk-root' 缺失值 ('?') 填充为: {mode_value}")
    
    return df

# --- 3. 特征工程 (编码) ---

def feature_engineering(df):
    """对分类特征进行编码。"""
    
    # 目标变量处理：将 'class' 编码为数字 (0 和 1)
    # 'e' (edible) -> 0, 'p' (poisonous) -> 1
    df['class'] = df['class'].map({'e': 0, 'p': 1})
    
    # 分离特征 X 和目标变量 y
    X = df.drop('class', axis=1)
    y = df['class']
    
    # 独热编码 (One-Hot Encoding)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    print(f"独热编码后的特征数量: {X_encoded.shape[1]}")
    
    # 将目标变量 y 重新加入到特征集中，形成最终的 DataFrame
    # 目标变量放在第一列
    final_df = pd.concat([y.rename('target'), X_encoded], axis=1)
    
    return final_df

# --- 4. 数据保存函数 ---

def save_data(final_df):
    """保存数据到 CSV 和 NumPy 文件。"""
    
    # --- 保存为 CSV 文件 ---
    final_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\n数据已保存到 CSV 文件: {OUTPUT_CSV_FILE}")
    
    # --- 保存为 NPY 文件 ---
    # 转换为 NumPy 数组 (只包含数值，不包含列名)
    final_array = final_df.values.astype(np.float32)
    np.save(OUTPUT_NPY_FILE, final_array)
    print(f"数据已保存到 NumPy 文件: {OUTPUT_NPY_FILE}")
    print(f"NumPy 数组形状: {final_array.shape}, 数据类型: {final_array.dtype}")
    
    return final_array

# --- 5. 执行流程 ---

def run_preprocessing_and_save():
    """运行整个数据预处理流程并保存文件。"""
    
    df = load_and_clean_data(FILE_PATH, COLUMN_NAMES)
    if df is None:
        return

    # 进行特征工程，并组合成最终的 DataFrame
    final_df = feature_engineering(df)
    
    # 保存数据
    save_data(final_df)
    
    print("\n预处理、特征工程和保存完成。")

if __name__ == "__main__":
    run_preprocessing_and_save()