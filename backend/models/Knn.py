import numpy as np
from collections import Counter
from utils import check_Feature_Label_Alignment
from utils import normalize_data

class KnnClassifier:
    '''
    k近邻分类器,只支持二分类任务,超参敏感,默认归一化(数据尺度敏感)
    '''

    def __init__(self, k=3, normalize=True):
        self.k = k
        self.normalize = normalize
        self.X_train = None
        self.y_train = None

        self._X_dim= None
        self._X_mean= None
        self._X_std= None

    def fit(self, X, y):
        '''
        由于knn是惰性学习,所以只需要进行数据校验和数据预处理
        '''

        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(y,dtype=np.float64)
        #数据校验
        X,y=check_Feature_Label_Alignment(X,y)
        self._X_dim=X.shape[1]

        #数据预处理
        if self.normalize:
            X,self._X_mean,self._X_std=normalize_data(X)
        else:
            self._X_mean=np.zeros(self._X_dim)
            self._X_std=np.ones(self._X_dim)

        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        '''
        批量预测函数
        '''

        X=np.asarray(X,dtype=np.float64)
        #数据校验
        if X.shape[1]!=self._X_dim:
            raise ValueError(f'输入特征维度错误,期望维度为{self._X_dim},但收到的维度为{X.shape[1]}')
        
        #数据预处理
        X_test=(X - self._X_mean) / self._X_std
        y_train=self.y_train.reshape(-1)

        #推理
        X_train=self.X_train
        y_train=self.y_train.reshape(-1)
  
        
        #利用广播计算距离矩阵,避免循环,提升批处理效率
        #由于目的是比较大小,所以不需要开根号
        distances_matrix= np.sum(X_test**2,axis=1,keepdims=True) + \
                            np.sum(X_train**2,axis=1,keepdims=True).T - 2 * X_test @ X_train.T
        distances_matrix=np.maximum(distances_matrix,0.0)  #数值稳定性处理

        incidences=np.argsort(distances_matrix,axis=1)[:,:self.k]  #取前k个最近邻的索引
        y_labels=y_train[incidences]  #取前k个最近邻的标签

        predicts=np.ndarray(shape=(X_test.shape[0],),dtype=y_train.dtype)
        for i in range(X.shape[0]):
            most_common=Counter(y_labels[i,:]).most_common(1)
            predicts[i]=most_common[0][0]

        return predicts
        
        


class KnnRegressor:
    '''
    k近邻回归器,只多维度输出,超参敏感,默认归一化(数据尺度敏感)
    '''

    def __init__(self, k=3, normalize=True):
        self.k = k
        self.normalize = normalize
        self.X_train = None
        self.y_train = None

        self._X_dim= None
        self._X_mean= None
        self._X_std= None


    def fit(self, X, y):
        '''
        由于knn是惰性学习,所以只需要进行数据校验和数据预处理
        '''

        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(y,dtype=np.float64)
        #数据校验
        X,y=check_Feature_Label_Alignment(X,y)
        self._X_dim=X.shape[1]
        self._y_dim=y.shape[1] if len(y.shape)>1 else 1

        #数据预处理
        if self.normalize:
            X,self._X_mean,self._X_std=normalize_data(X)
        else:
            self._X_mean=np.zeros(self._X_dim)
            self._X_std=np.ones(self._X_dim)

        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        '''
        批量预测函数
        '''

        X=np.asarray(X,dtype=np.float64)
        #数据校验
        if X.shape[1]!=self._X_dim:
            raise ValueError(f'输入特征维度错误,期望维度为{self._X_dim},但收到的维度为{X.shape[1]}')
        
        #数据预处理
        X_test=(X - self._X_mean) / self._X_std

        #推理
        X_train=self.X_train
        y_train=self.y_train
        
        #利用广播计算距离矩阵,避免循环,提升批处理效率
        #由于目的是比较大小,所以不需要开根号
        distances_matrix= np.sum(X_test**2,axis=1,keepdims=True) + \
                            np.sum(X_train**2,axis=1,keepdims=True).T - 2 * X_test @ X_train.T
        distances_matrix=np.maximum(distances_matrix,0.0)  #数值稳定性处理

        incidences=np.argsort(distances_matrix,axis=1)[:,:self.k]  #取前k个最近邻的索引
        y_values=y_train[incidences]  #取前k个最近邻的标签

        predicts=np.ndarray(shape=(X_test.shape[0],y_train.shape[1]),dtype=y_train.dtype)
        for i in range(X.shape[0]):
            predicts[i,:]=np.mean(y_values[i,:,:],axis=0)

        return predicts
        
if __name__ == "__main__":

    X_train = np.array([
        [1, 1], [1, 2], [2, 1], # 0 类
        [6, 6], [7, 6], [6, 7]  # 1 类
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # 多个测试样本
    X_test = np.array([
        [3, 3],  # 预期靠近 0 类
        [8, 8],  # 预期靠近 1 类
        [0, 0]   # 预期靠近 0 类
    ])

    knnC = KnnClassifier(k=3, normalize=True)
    knnC.fit(X_train, y_train)
    predictions = knnC.predict(X_test)
    print("KNN Classifier Predictions:", predictions)  # 输出预测结果


    X_train = np.array([
        [1, 1], [1, 2], [2, 1], # 0 类
        [6, 6], [7, 6], [6, 7]  # 1 类
    ])
    y_train = np.array([[0,1], 
                        [2,0], 
                        [4,6], 
                        [1,2], 
                        [4,5], 
                        [0,0]])

    # 多个测试样本
    X_test = np.array([
        [3, 3],  # 预期靠近 0 类
        [8, 8],  # 预期靠近 1 类
        [0, 0]   # 预期靠近 0 类
    ])

    knnR= KnnRegressor(k=3, normalize=True)
    knnR.fit(X_train, y_train)
    predictions = knnR.predict(X_test)
    print("KNN Regressor Predictions:", predictions)  # 输出预测结果

