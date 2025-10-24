# **机器学习算法展示平台开发文档**

项目名称： 机器学习算法展示平台  
技术栈： 后端：FastAPI (Python)；前端：Vue.js 3  
作者： Gemini (AI 协作)  
日期： 2025 年 10 月

## **目录**

1. 项目概述与目标  
2. 架构设计与技术选型  
3. 后端 (FastAPI) 设计  
   3.1. 文件结构  
   3.2. 核心 API 规范  
   3.3. 统一参数配置  
   3.4. 模型训练流程  
4. 前端 (Vue.js) 设计与实现  
   4.1. 页面布局与区域功能  
   4.2. 核心状态管理（任务分离）  
   4.3. 交互与可视化实现  
5. 安装与部署指南  
6. 未来优化方向

## **1\. 项目概述与目标**

本项目旨在构建一个高交互性、实时反馈的 Web 平台，用于展示和比较主流机器学习算法（分类与回归）的性能。用户通过界面配置实验参数（数据集、模型、超参数、PCA），触发后端训练，并以雷达图和趋势图的形式可视化结果。

**核心目标：**

1. 实现分类和回归任务的清晰分离。  
2. 统一后端模型接口，简化训练调用。  
3. 前端采用清晰的四分格布局，提供优秀的用户体验。  
4. 实时记录并展示最新五个模型的训练结果，且回归和分类结果独立存储。  
5. **前端本地估算训练时间**，驱动进度条，简化后端任务管理。

## **2\. 架构设计与技术选型**

### **2.1 整体架构**

采用经典**前后端分离**架构，通过标准的 RESTful API 进行数据交换。训练流程采用单次同步（阻塞式）请求，前端本地展示进度条。

| 模块 | 技术 | 职责 |
| :---- | :---- | :---- |
| **后端服务** | Python / FastAPI | 接收请求、数据加载、数据预处理（PCA）、模型训练、指标计算、结果序列化。 |
| **前端应用** | Vue.js (3.x) | 用户界面渲染、状态管理、API 调用、数据缓存（最新 5 个模型）、高性能可视化（ECharts）。 |
| **持久化** | 本地文件 (npy) | 存储数据集。 |

### **2.2 文件结构**

核心开发将集中在 backend/app 和 frontend 文件夹。

E:.  
...  
├─backend  
│  │  \_\_init\_\_.py  
│  │    
│  ├─app                  \<- FastAPI 应用主体、路由、业务逻辑  
│  │  │  main.py          \<- 启动文件与主路由  
│  │  │  config.py        \<- 模型与超参数配置字典  
│  │  │  processing.py    \<- 数据加载、分割、PCA 逻辑  
│  │  │  schemas.py       \<- API 请求/响应模型 (Pydantic)  
│  │    
│  ├─dataset              \<- 数据集  
│  └─models               \<- 模型实现  
│      │  ...  
│      │  PCA.py  
│      │  utils.py         
│        
├─frontend                \<- Vue.js 应用  
│  │  src/  
│  │  ├─components/  
│  │  │  ├─Layout.vue         \<- 整体四分格布局  
│  │  │  ├─TaskSelector.vue   \<- 任务选择 (左上 A)  
│  │  │  ├─ControlPanel.vue   \<- 配置/控制 (右上 B)  
│  │  │  ├─HyperParams.vue    \<- 超参数配置 (左下 C)  
│  │  │  └─ResultDisplay.vue  \<- 结果展示 (右下 D)  
│  │  ├─store/  
│  │  │  └─experiment.js    \<- Pinia/Vuex 状态管理模块  
│  │  └─views/  
│  │      └─Home.vue

## **3\. 后端 (FastAPI) 设计**

### **3.1 统一参数配置 (backend/app/config.py)**

用于定义所有可用模型、数据集、指标、超参数和**基线训练时间**。

\# \--- 任务定义与模型映射 \---  
TASK\_MODELS \= {  
    "classification": \[  
        "DecisionTreeClassifier", "GBM\_C", "KMeans", "KnnClassifier",   
        "LogisticRegression", "NaiveBayes", "RandomForestClassifier", "SVM"  
    \],  
    "regression": \[  
        "GBM\_R", "DecisionTreeRegressor", "KnnRegressor",   
        "LinearRegression", "RandomForestRegressor"  
    \]  
}

\# \--- 数据集定义 \---  
TASK\_DATASETS \= {  
    "classification": \["iris", "mushroom", "titanic"\],  
    "regression": \["auto\_mpg", "california\_housing", "diabetes"\]  
}

\# \--- 统一性能指标定义 \---  
CLASSIFICATION\_METRICS \= \["Accuracy", "Precision", "Recall", "F1 Score"\]  
REGRESSION\_METRICS \= \["MSE", "RMSE", "MAE", "R2 Score"\]

\# \--- PCA 参数配置 \---  
PCA\_PARAMS \= {  
    "n\_components": 2, \# 降维后的目标维度  
}

\# \--- 模型超参数配置 \---  
\# ... (此处应包含所有模型及超参数的详细配置)

\# \--- 模型基线训练时间 (在完整数据集上测试所得，单位：秒) \---  
\# 前端将根据此基准时间进行估算  
BASELINE\_TIMES \= {  
    "DecisionTreeClassifier\_iris": 0.05,  
    "SVM\_california\_housing": 2.5,  
    "LinearRegression\_diabetes": 0.03,  
    "RandomForestClassifier\_mushroom": 0.8,  
    \# ... 完整数据集和模型组合的基准时间表  
}

### **3.2 核心 API 规范 (backend/app/main.py)**

训练流程采用单次阻塞式请求/响应模式。

#### **API 1: 获取配置信息 (GET /api/v1/config)**

此接口返回前端渲染所需的全部静态配置数据，包括 TASK\_MODELS, TASK\_DATASETS, MODEL\_PARAMS, PCA\_PARAMS, 以及 **BASELINE\_TIMES**。

#### **API 2: 模型训练与评估 (POST /api/v1/train)**

接收配置参数，执行模型训练和评估，并在完成后返回结果。前端将阻塞等待此响应。

**请求模型 (schemas.py \- TrainRequest)**

| 字段 | 类型 | 描述 |
| :---- | :---- | :---- |
| task\_type | str | "classification" 或 "regression"。 |
| dataset\_name | str | 选定的数据集名称。 |
| model\_name | str | 选定的模型名称。 |
| split\_ratio | float | 训练集分割比例 (0.5 \- 0.9)。 |
| use\_pca | bool | 是否进行 PCA 降维。 |
| pca\_params | dict | PCA 参数 (e.g., {"n\_components": 2})。 |
| model\_params | dict | 选定模型的超参数字典。 |

**响应模型 (schemas.py \- TrainResult)**

此结构即为 API 2 的最终响应体。

| 字段 | 类型 | 描述 |
| :---- | :---- | :---- |
| task\_type | str | 任务类型。 |
| model\_name | str | 训练使用的模型名称。 |
| metrics | dict | **测试集**性能指标。 |
| test\_predictions | list | 测试集上的预测结果。 |
| test\_targets | list | 测试集上的真实目标值。 |
| train\_loss\_history | list | 训练过程中的损失/指标历史（如果有）。 |

### **3.3 模型训练流程 (backend/app/processing.py)**

1. **加载数据**：根据 dataset\_name 加载数据。  
2. **数据分割**：根据 split\_ratio 分割为训练集/测试集。  
3. **PCA 处理 (Conditional)**：如果 use\_pca 为 True，应用 PCA 降维。  
4. **模型训练**：实例化模型，传入 model\_params，调用 model.fit()。  
5. **模型评估**：调用 model.predict()，计算性能指标和图表所需的数据。  
6. **返回结果**：将 TrainResult 对象序列化为 JSON 返回。

## **4\. 前端 (Vue.js) 设计与实现**

### **4.2 核心状态管理 (experiment.js)**

需要严格分离历史记录，并管理进度条状态。

// Pinia/Vuex State 示例  
const state \= {  
    currentTask: 'regression',   
    currentConfig: {   
        dataset\_name: null,   
        model\_name: null,   
        split\_ratio: 0.7,   
        use\_pca: false,   
        pca\_params: PCA\_PARAMS,   
        model\_params: {}   
    },

    // 历史记录 (严格分离，最大长度均为 5\)  
    regressionHistory: \[\],     
    classificationHistory: \[\], 

    // 时间估算所需数据  
    baselineTimes: {}, // 存储从 API 1 获取的 BASELINE\_TIMES  
    estimatedTime: 0,  // 估算的总时间（秒）  
    elapsedTime: 0,    // 已流逝时间（秒）  
      
    // 加载状态  
    isLoading: false,  
};

### **4.3 交互与可视化实现 (ResultDisplay.vue)**

#### **初始加载逻辑**

1. 在应用启动时，调用 GET /api/v1/config 获取配置数据，并将 **BASELINE\_TIMES** 存储到 baselineTimes 状态中。

#### **运行按钮 (B区) 交互逻辑：**

1. 用户点击运行按钮。  
2. **【前端估算时间】** 调用前端函数 calculateEstimatedTime() 计算本次训练的预估时间 estimatedTime。  
   * **估算公式（示例）：**  
     // T\_base \= BASELINE\_TIMES\[modelName \+ '\_' \+ datasetName\];  
     // 估算时间 \= T\_base \* currentConfig.split\_ratio \* PCA\_Factor \* HyperParam\_Factor;  
     // 注意：这是前端的预估时间，用于驱动进度条，不用于后端实际计时。

3. 设置 isLoading \= true，初始化 elapsedTime \= 0。  
4. **启动本地进度条计时器**：设置一个 setInterval，每 100 毫秒递增 elapsedTime，并更新进度条。  
   * 进度条百分比：Math.min(99, (state.elapsedTime / state.estimatedTime) \* 100)。  
   * **时间耗尽处理**：如果 elapsedTime 超过 estimatedTime，进度条**停在 99%**，等待后端响应。  
5. **发送阻塞请求**：发送 POST /api/v1/train 请求。  
6. **接收结果处理**：  
   * **成功响应 (200)**：  
     * 清除本地计时器。  
     * 进度条设置为 100%。  
     * 设置 isLoading \= false。  
     * 将 response.result (即 TrainResult) 添加到相应的 \*History 数组中。  
     * 更新雷达图。  
   * **失败响应 (Error)**：  
     * 清除本地计时器。  
     * 设置 isLoading \= false。  
     * 向用户展示错误信息。

#### **4.3.1 雷达图 (左侧 D)**

* **数据源**：根据 currentTask 状态动态选择 regressionHistory 或 classificationHistory。  
* **交互**：点击雷达图上的指标轴标签时，更新 selectedMetric 状态，触发右侧详细图表的渲染。

#### **4.3.2 详细图表 (右侧 D)**

* **数据聚合**：图表聚合当前历史记录（最新的 5 个模型）在 selectedMetric 上的表现。  
* **对比展示**：**必须**展示训练集与测试集的结果。

## **5\. 安装与部署指南**

### **5.1 后端环境配置 (FastAPI)**

1. **环境准备**：  
   git clone \[repository\_url\]  
   cd project/backend  
   python \-m venv venv  
   source venv/bin/activate

2. **依赖安装**：  
   pip install \-r requirements.txt 

3. **启动服务**：  
   uvicorn app.main:app \--reload \--host 0.0.0.0 \--port 8000

### **5.2 前端环境配置 (Vue.js)**

1. **环境准备**：  
   cd project/frontend

2. **依赖安装**：  
   npm install

3. **启动应用**：  
   npm run serve

## **6\. 未来优化方向**

1. **性能监控**：在 TrainResult 中增加训练耗时、预测耗时等指标，用于优化前端的时间估算公式。  
2. **数据类型提示**：在前端展示数据集中各个特征的数据类型，辅助用户选择合适的模型和超参数。  
3. **模型热加载**：在后端实现模型文件的热加载机制，允许添加新模型而无需重启 FastAPI 服务。