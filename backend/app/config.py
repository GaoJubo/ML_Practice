# --- 任务定义与模型映射 ---
TASK_MODELS = {
    "classification": [
        "DecisionTreeClassifier", "GBM_C", "KMeans", "KnnClassifier", 
        "LogisticRegression", "NaiveBayes", "RandomForestClassifier", "SVM"
    ],
    "regression": [
        "GBM_R", "DecisionTreeRegressor", "KnnRegressor", 
        "LinearRegression", "RandomForestRegressor"
    ]
}

# --- 数据集定义 ---
TASK_DATASETS = {
    "classification": ["iris", "mushroom", "titanic"],
    "regression": ["auto_mpg", "california_housing", "diabetes"]
}

# --- 统一性能指标定义 ---
CLASSIFICATION_METRICS = ["Accuracy", "Precision", "Recall", "F1 Score"]
REGRESSION_METRICS = ["MSE", "RMSE", "MAE", "R2 Score"]

# --- PCA 参数配置 ---
PCA_PARAMS = {
    "n_components": 2,  # 降维后的目标维度
}

# --- 模型超参数配置 ---
MODEL_PARAMS = {
    "DecisionTreeClassifier": {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    "DecisionTreeRegressor": {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    "RandomForestClassifier": {
        "n_estimators": 100,
        "max_features": "sqrt",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    "RandomForestRegressor": {
        "n_estimators": 100,
        "max_features": "sqrt",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    "GBM_C": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3
    },
    "GBM_R": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3
    },
    "KnnClassifier": {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto"
    },
    "KnnRegressor": {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto"
    },
    "LogisticRegression": {
        "C": 1.0,
        "fit_intercept": True,
        "max_iter": 100
    },
    "LinearRegression": {
        "fit_intercept": True
    },
    "NaiveBayes": {
        "var_smoothing": 1e-9
    },
    "SVM": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale"
    },
    "KMeans": {
        "n_clusters": 3,
        "init": "k-means++",
        "n_init": 10
    }
}

# --- 模型基准训练时间 (在完整数据集上测试所得，单位：秒) ---
# 前端将根据此基准时间进行估算
BASELINE_TIMES = {
    "DecisionTreeClassifier_iris": 0.05,
    "SVM_california_housing": 2.5,
    "LinearRegression_diabetes": 0.03,
    "RandomForestClassifier_mushroom": 0.8,
    "DecisionTreeRegressor_auto_mpg": 0.06,
    "KnnClassifier_iris": 0.02,
    "KnnRegressor_diabetes": 0.03,
    "LogisticRegression_titanic": 0.1,
    "NaiveBayes_iris": 0.01,
    "GBM_C_iris": 0.2,
    "GBM_R_california_housing": 1.5,
    "RandomForestRegressor_auto_mpg": 0.5,
    "KMeans_iris": 0.1
}