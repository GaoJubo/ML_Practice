from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import numpy as np
import time
import json
from datetime import datetime
import os
import sys
import importlib
import inspect
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.datasets import (
    load_iris, load_breast_cancer, load_wine, load_digits,
    load_boston, load_diabetes, fetch_california_housing
)
import pandas as pd

# 导入配置
from config import (
    TASK_MODELS, TASK_DATASETS, CLASSIFICATION_METRICS, 
    REGRESSION_METRICS, PCA_PARAMS, BASELINE_TIMES
)

# 创建FastAPI应用
app = FastAPI(
    title="机器学习算法展示平台 API",
    description="提供机器学习模型训练和评估的API接口",
    version="1.0.0"
)

# 添加CORS中间件，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求和响应模型
class TrainRequest(BaseModel):
    task_type: str  # "classification" 或 "regression"
    dataset_name: str
    model_name: str
    split_ratio: float = 0.7
    use_pca: bool = False
    pca_params: Dict[str, Any] = {}
    model_params: Dict[str, Any] = {}

class TrainResult(BaseModel):
    task_type: str
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    test_predictions: List[Any]
    test_targets: List[Any]
    train_loss_history: Optional[List[float]] = None
    timestamp: str

# 数据集加载函数
def load_dataset(dataset_name: str):
    """根据数据集名称加载数据"""
    if dataset_name == "iris":
        data = load_iris()
    elif dataset_name == "mushroom":
        # 模拟蘑菇数据集，实际应用中应该从文件加载
        data = load_breast_cancer()
    elif dataset_name == "titanic":
        # 模拟泰坦尼克号数据集，实际应用中应该从文件加载
        data = load_iris()
    elif dataset_name == "auto_mpg":
        # 模拟auto_mpg数据集，实际应用中应该从文件加载
        data = load_boston()
    elif dataset_name == "california_housing":
        data = fetch_california_housing()
    elif dataset_name == "diabetes":
        data = load_diabetes()
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    return data

# 模型训练和评估函数
def train_and_evaluate(request: TrainRequest):
    """训练模型并评估性能"""
    start_time = time.time()
    
    # 加载数据集
    data = load_dataset(request.dataset_name)
    X, y = data.data, data.target
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-request.split_ratio, random_state=42
    )
    
    # PCA降维
    if request.use_pca:
        n_components = request.pca_params.get("n_components", PCA_PARAMS["n_components"])
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    
    # 获取模型类
    model_class = None
    if request.model_name == "DecisionTreeClassifier":
        model_class = DecisionTreeClassifier
    elif request.model_name == "DecisionTreeRegressor":
        model_class = DecisionTreeRegressor
    elif request.model_name == "RandomForestClassifier":
        model_class = RandomForestClassifier
    elif request.model_name == "RandomForestRegressor":
        model_class = RandomForestRegressor
    elif request.model_name == "GBM_C":
        model_class = GradientBoostingClassifier
    elif request.model_name == "GBM_R":
        model_class = GradientBoostingRegressor
    elif request.model_name == "KnnClassifier":
        model_class = KNeighborsClassifier
    elif request.model_name == "KnnRegressor":
        model_class = KNeighborsRegressor
    elif request.model_name == "LogisticRegression":
        model_class = LogisticRegression
    elif request.model_name == "LinearRegression":
        model_class = LinearRegression
    elif request.model_name == "NaiveBayes":
        model_class = GaussianNB
    elif request.model_name == "SVM":
        model_class = SVC
    elif request.model_name == "KMeans":
        model_class = KMeans
    else:
        raise ValueError(f"未知模型: {request.model_name}")
    
    # 实例化模型
    model_params = request.model_params.copy()
    # 处理字符串类型的参数值
    for key, value in model_params.items():
        if isinstance(value, str):
            if value.lower() == "true":
                model_params[key] = True
            elif value.lower() == "false":
                model_params[key] = False
            elif value.lower() == "null":
                model_params[key] = None
            elif value.replace(".", "", 1).isdigit():
                model_params[key] = float(value) if "." in value else int(value)
    
    model = model_class(**model_params)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    metrics = {}
    if request.task_type == "classification":
        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
        metrics["Precision"] = precision_score(y_test, y_pred, average='weighted')
        metrics["Recall"] = recall_score(y_test, y_pred, average='weighted')
        metrics["F1 Score"] = f1_score(y_test, y_pred, average='weighted')
    else:  # regression
        metrics["MSE"] = mean_squared_error(y_test, y_pred)
        metrics["RMSE"] = np.sqrt(metrics["MSE"])
        metrics["MAE"] = mean_absolute_error(y_test, y_pred)
        metrics["R2 Score"] = r2_score(y_test, y_pred)
    
    # 训练损失历史（对于某些模型）
    train_loss_history = None
    if hasattr(model, 'loss_curve_'):
        train_loss_history = model.loss_curve_.tolist()
    
    # 计算训练时间
    training_time = time.time() - start_time
    
    # 返回结果
    result = TrainResult(
        task_type=request.task_type,
        model_name=request.model_name,
        dataset_name=request.dataset_name,
        metrics=metrics,
        test_predictions=y_pred.tolist(),
        test_targets=y_test.tolist(),
        train_loss_history=train_loss_history,
        timestamp=datetime.now().isoformat()
    )
    
    return result

# API端点
@app.get("/api/v1/config")
async def get_config():
    """获取前端配置数据"""
    return {
        "TASK_MODELS": TASK_MODELS,
        "TASK_DATASETS": TASK_DATASETS,
        "CLASSIFICATION_METRICS": CLASSIFICATION_METRICS,
        "REGRESSION_METRICS": REGRESSION_METRICS,
        "PCA_PARAMS": PCA_PARAMS,
        "BASELINE_TIMES": BASELINE_TIMES
    }

@app.post("/api/v1/train", response_model=TrainResult)
async def train_model(request: TrainRequest):
    """训练模型并返回结果"""
    try:
        result = train_and_evaluate(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "机器学习算法展示平台 API",
        "version": "1.0.0",
        "docs": "/docs"
    }

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)