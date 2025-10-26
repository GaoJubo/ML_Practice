import numpy as np
from collections import Counter
from utils import normalize_data
from utils import check_Feature_Label_Alignment

class KMeans:
    '''
    K-Means 聚类算法实现.
    但是我对其进行了一个简单的改动,使其可以在二分类有监督任务上使用
    就是看它所归属的簇的标签的众数作为该样本的预测标签
    '''
    def __init__(self, max_iters=300, tol=1e-4, random_state=313, normalize=True):
        '''
        为了适应有监督的二分类任务,固定质心个数为2,并且添加了一个属性y用来存储训练集的真实标签
        '''
        self._init(n_clusters=2, max_iters=max_iters, tol=tol, random_state=random_state, normalize=normalize)
        self.y=None

    def _init(self, n_clusters=2, max_iters=300, tol=1e-4, random_state=313, normalize=True):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None#一个向量列表
        self.labels_ = None
        self.random_state = random_state
        self.normalize = normalize

        self._X_mean = None
        self._X_std = None
        self.clusters_labels=[]

    def _calculate_distance(self, X, centroids):
        """
        计算每个样本到每个质心的距离

        Parameters:
        X (np.ndarray): 特征数据集，形状为 (n_samples, n_features)。
        centroids (np.ndarray): 质心数据集，形状为 (n_clusters, n_features)。

        Returns:
        np.ndarray: 距离矩阵，形状为 (n_samples, n_clusters)。
        """
        #仅仅计算欧氏距离的平方以提高效率
        distances = np.sum(X,axis=1,keepdims=True)-2 * X@centroids.T + np.sum(centroids**2, axis=1).T

        return np.maximum(distances, 0)  # 确保距离非负

    def fit(self, X, y):
        '''
        训练K-Means模型

        Parameters:
        X (np.ndarray): 特征数据集，形状为 (n_samples, n_features)。
        y (np.ndarray): 真实标签，形状为 (n_samples, )。
        '''
        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(y,dtype=np.int64)
        X,y=check_Feature_Label_Alignment(X,y)

        self.y = y
        self._fit(X)

        predict_labels=self.labels_
        cluster_labels=np.zeros(self.n_clusters)
        for k in range(self.n_clusters):
            mask = (predict_labels == k)
            most_common = Counter(np.squeeze(self.y[mask])).most_common(1)
            cluster_labels[k] = most_common[0][0]
            


        self.clusters_labels=cluster_labels

        # print(f'训练结束{X.shape} {y.shape}')

    def _fit(self, X):

        #数据预处理
        X=np.asarray(X,dtype=np.float64)

        if self.normalize:
            X, self._X_mean, self._X_std = normalize_data(X)
        else:
            self._X_mean = np.zeros(X.shape[1])
            self._X_std = np.ones(X.shape[1])

        #初始化质心
        np.random.seed(self.random_state)
        centroids_incides = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[centroids_incides]

        #迭代优化
        for i in range(self.max_iters):
            #计算距离矩阵,并得到簇标签列表
            disances = self._calculate_distance(X, self.centroids)
            self.labels_ = np.argmin(disances, axis=1)

            #更新每个质心
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask=(self.labels_==k)
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)

            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids

    
    def _predict(self, X):
        '''
        预测函数

        Parameters:
        X (np.ndarray): 特征数据集，形状为 (n_samples, n_features)。

        Returns:
        np.ndarray: 预测标签，形状为 (n_samples, )。
        '''
        X=np.asarray(X,dtype=np.float64)

        #数据预处理
        X=(X - self._X_mean) / self._X_std

        distances = self._calculate_distance(X, self.centroids)
        predict_labels = np.argmin(distances, axis=1)

        return predict_labels
    
    def predict(self, X):
        X=np.asarray(X,dtype=np.float64)
        predict_labels = self._predict(X)
        predictions=np.empty(X.shape[0],dtype=np.int64)
        for k in range(self.n_clusters):
            mask = (predict_labels == k)
            predictions[mask] = self.clusters_labels[k]

        return predictions
    
if __name__ == "__main__":
    X=[
        [1.0,2.0],
        [1.5,1.8],
        [5.0,8.0],
        [8.0,8.0],
        [1.0,0.6],
        [9.0,11.0],
    ]

    y=[0,0,1,1,0,1]

    kmeans=KMeans(max_iters=100, tol=1, random_state=313, normalize=False)
    kmeans.fit(X,y) 
    preds=kmeans.predict(X)
    print("预测结果:",preds)