import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

class DataProcessor:
    """数据处理类，负责数据加载、预处理和分割"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """根据数据集名称加载数据"""
        if dataset_name == "iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            X, y = data.data, data.target
        elif dataset_name == "mushroom":
            # 模拟蘑菇数据集，实际应用中应该从文件加载
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            X, y = data.data, data.target
        elif dataset_name == "titanic":
            # 模拟泰坦尼克号数据集，实际应用中应该从文件加载
            from sklearn.datasets import load_iris
            data = load_iris()
            X, y = data.data, data.target
        elif dataset_name == "auto_mpg":
            # 模拟auto_mpg数据集，实际应用中应该从文件加载
            from sklearn.datasets import load_boston
            data = load_boston()
            X, y = data.data, data.target
        elif dataset_name == "california_housing":
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            X, y = data.data, data.target
        elif dataset_name == "diabetes":
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            X, y = data.data, data.target
        else:
            raise ValueError(f"未知数据集: {dataset_name}")
        
        return X, y
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray, task_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """数据预处理，包括标准化和标签编码"""
        # 特征标准化
        X = self.scaler.fit_transform(X)
        
        # 如果是分类任务且标签是字符串类型，进行标签编码
        if task_type == "classification" and y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
            
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, split_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """数据分割为训练集和测试集"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-split_ratio, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    def apply_pca(self, X_train: np.ndarray, X_test: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """应用PCA降维"""
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_test_pca

class ModelTrainer:
    """模型训练类，负责模型训练和评估"""
    
    def __init__(self):
        self.model = None
        self.training_time = 0
        self.train_loss_history = []
        
    def get_model(self, model_name: str, model_params: Dict[str, Any]):
        """根据模型名称和参数获取模型实例"""
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC, SVR
        from sklearn.cluster import KMeans
        
        # 处理参数类型转换
        processed_params = {}
        for key, value in model_params.items():
            if isinstance(value, str):
                if value.lower() == "true":
                    processed_params[key] = True
                elif value.lower() == "false":
                    processed_params[key] = False
                elif value.lower() == "null":
                    processed_params[key] = None
                elif value.replace(".", "", 1).isdigit():
                    processed_params[key] = float(value) if "." in value else int(value)
                else:
                    processed_params[key] = value
            else:
                processed_params[key] = value
        
        # 根据模型名称实例化模型
        if model_name == "DecisionTreeClassifier":
            self.model = DecisionTreeClassifier(**processed_params)
        elif model_name == "DecisionTreeRegressor":
            self.model = DecisionTreeRegressor(**processed_params)
        elif model_name == "RandomForestClassifier":
            self.model = RandomForestClassifier(**processed_params)
        elif model_name == "RandomForestRegressor":
            self.model = RandomForestRegressor(**processed_params)
        elif model_name == "GBM_C":
            self.model = GradientBoostingClassifier(**processed_params)
        elif model_name == "GBM_R":
            self.model = GradientBoostingRegressor(**processed_params)
        elif model_name == "KnnClassifier":
            self.model = KNeighborsClassifier(**processed_params)
        elif model_name == "KnnRegressor":
            self.model = KNeighborsRegressor(**processed_params)
        elif model_name == "LogisticRegression":
            self.model = LogisticRegression(**processed_params)
        elif model_name == "LinearRegression":
            self.model = LinearRegression(**processed_params)
        elif model_name == "NaiveBayes":
            self.model = GaussianNB(**processed_params)
        elif model_name == "SVM":
            self.model = SVC(**processed_params)
        elif model_name == "KMeans":
            self.model = KMeans(**processed_params)
        else:
            raise ValueError(f"未知模型: {model_name}")
            
        return self.model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练模型"""
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        # 如果模型有损失历史，记录下来
        if hasattr(self.model, 'loss_curve_'):
            self.train_loss_history = self.model.loss_curve_.tolist()
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, task_type: str) -> Dict[str, float]:
        """评估模型性能"""
        y_pred = self.model.predict(X_test)
        
        metrics = {}
        if task_type == "classification":
            metrics["Accuracy"] = accuracy_score(y_test, y_pred)
            metrics["Precision"] = precision_score(y_test, y_pred, average='weighted')
            metrics["Recall"] = recall_score(y_test, y_pred, average='weighted')
            metrics["F1 Score"] = f1_score(y_test, y_pred, average='weighted')
        else:  # regression
            metrics["MSE"] = mean_squared_error(y_test, y_pred)
            metrics["RMSE"] = np.sqrt(metrics["MSE"])
            metrics["MAE"] = mean_absolute_error(y_test, y_pred)
            metrics["R2 Score"] = r2_score(y_test, y_pred)
            
        return metrics
    
    def get_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """获取模型预测结果"""
        return self.model.predict(X_test)

def train_and_evaluate_model(
    task_type: str,
    dataset_name: str,
    model_name: str,
    split_ratio: float = 0.7,
    use_pca: bool = False,
    pca_params: Dict[str, Any] = {},
    model_params: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """完整的模型训练和评估流程"""
    # 初始化处理器
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    
    # 加载数据
    X, y = data_processor.load_dataset(dataset_name)
    
    # 数据预处理
    X, y = data_processor.preprocess_data(X, y, task_type)
    
    # 数据分割
    X_train, X_test, y_train, y_test = data_processor.split_data(X, y, split_ratio)
    
    # PCA降维
    if use_pca:
        n_components = pca_params.get("n_components", 2)
        X_train, X_test = data_processor.apply_pca(X_train, X_test, n_components)
    
    # 获取并训练模型
    model = model_trainer.get_model(model_name, model_params)
    model_trainer.train_model(X_train, y_train)
    
    # 评估模型
    metrics = model_trainer.evaluate_model(X_test, y_test, task_type)
    
    # 获取预测结果
    y_pred = model_trainer.get_predictions(X_test)
    
    # 构建结果字典
    result = {
        "task_type": task_type,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "metrics": metrics,
        "test_predictions": y_pred.tolist(),
        "test_targets": y_test.tolist(),
        "train_loss_history": model_trainer.train_loss_history,
        "training_time": model_trainer.training_time,
        "timestamp": datetime.now().isoformat()
    }
    
    return result