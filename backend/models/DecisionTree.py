import numpy as np
from abc import ABC, abstractmethod
from utils import check_Feature_Label_Alignment

class Node():
    '''
    决策树节点
    '''
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,value=None):
        self.feature_index=feature_index
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

class DecisionTree(ABC):
    '''
    决策树基类,可以适用于连续和离散的特征,多维回归和多值单标签分类
    '''

    def __init__(self,min_samples_split=2, max_depth=100,random_state=313):
        self.root=None
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.random_state=random_state

        self.n_features_=None

    def fit(self, X, y):
        '''
        训练决策树模型
        '''
        X,y=np.array(X,dtype=np.float64),np.array(y,dtype=np.float64)
        check_Feature_Label_Alignment(X, y)
        y=y.reshape(-1,1)#确保y是二维的,列向量
    
        self.n_features_=X.shape[1]
        self.root=self._grow_tree(X,y)
        
    @abstractmethod
    def _calculate_loss(self,y:np.ndarray)->float:
        '''
        计算度量指标
        '''
        pass

    def _loss_split(self,y_left,y_right):
        '''
        计算一种划分的度量指标
        '''
        total_len=len(y_left)+len(y_right)

        return (len(y_left)*self._calculate_loss(y_left)+len(y_right)*self._calculate_loss(y_right))/total_len

    def _best_split(self,X,y):
        '''
        迭代最佳划分
        '''
        m,n_features=X.shape

        if m<=1:
            #就一个样本,无法划分
            return None,None,float('inf')

        best_loss=float('inf')
        best_feature_index=None
        best_threshold=None

        for feature_index in range(n_features):
            uniques_values=np.sort(np.unique(X[:,feature_index]).reshape(-1))

            if len(uniques_values)==1:
                #该特征所有值相同,无法划分
                continue

            for i in range(len(uniques_values)-1):
                threshold=(uniques_values[i]+uniques_values[i+1])/2

                left_y=y[X[:,feature_index]<=threshold]
                right_y=y[X[:,feature_index]>threshold]
                current_loss=self._loss_split(left_y,right_y)

                if current_loss<best_loss:
                    best_loss=current_loss
                    best_feature_index=feature_index
                    best_threshold=threshold
        
        return best_feature_index,best_threshold,best_loss
    
    @abstractmethod
    def _calculate_leaf_value(self,y:np.ndarray):
        '''
        计算叶节点的值
        '''
        pass

    def _grow_tree(self,X,y,depth=0):
        '''
        递归生成决策树
        '''
        if any([
            depth>=self.max_depth,
            X.shape[0]<self.min_samples_split,
            self._calculate_loss(y)==0
        ]) :
            return Node(value=self._calculate_leaf_value(y))
        
        feature_index,threshold,_=self._best_split(X,y)
        if feature_index is None:
            return Node(value=self._calculate_leaf_value(y))
        
        left_indices=X[:,feature_index]<=threshold
        right_indices=X[:,feature_index]>threshold
        left_node=self._grow_tree(X[left_indices],y[left_indices],depth+1)
        right_node=self._grow_tree(X[right_indices],y[right_indices],depth+1)

        return Node(feature_index=feature_index,threshold=threshold,left=left_node,right=right_node)
    
    def _predict_sample(self,x):
        '''
        预测单个样本
        '''
        node=self.root
        while node.value is None:
            if x[node.feature_index]<=node.threshold:
                node=node.left
            else:
                node=node.right
        
        return node.value
    
    def predict(self,X):
        '''
        批量预测样本
        '''
        X=np.array(X,dtype=np.float64)
        #数据校验
        if X.shape[1]!=self.n_features_:
            raise ValueError(f"特征数量不匹配,训练时特征数量为{self.n_features_},预测时特征数量为{X.shape[1]}")
        X=np.array(X,dtype=np.float64)

        predictions=[self._predict_sample(x) for x in X]
        return np.array(predictions)




class DecisionTreeClassifier(DecisionTree):
    '''
    决策树分类器
    '''

    def _calculate_loss(self,y:np.ndarray)->float:
        '''
        计算基尼指数
        '''
        m=len(y)
        if m==0:
            return 0.0
        
        classes,counts=np.unique(y,return_counts=True)
        prob_squared_sum=sum((count/m)**2 for count in counts)

        gini_index=1-prob_squared_sum
        return gini_index

    def _calculate_leaf_value(self,y:np.ndarray):
        '''
        计算叶节点的类别标签
        '''
        classes,counts=np.unique(y,return_counts=True)
        majority_class=classes[np.argmax(counts)]
        return majority_class
    
class DecisionTreeRegressor(DecisionTree):
    '''
    决策树回归器
    '''

    def _calculate_loss(self,y:np.ndarray)->float:
        '''
        计算均方误差
        '''
        if len(y)==0:
            return 0.0
        
        mean_value=np.mean(y,axis=0)
        mse=np.mean((y - mean_value)**2)
        return mse

    def _calculate_leaf_value(self,y:np.ndarray):
        '''
        计算叶节点的预测值
        '''
        mean_value=np.mean(y,axis=0)
        return mean_value
    
if __name__=="__main__":
    X_1=[[2.5, 1.5],
       [1.0, 3.5], 
       [3.5, 2.0],
       [4.0, 3.0],
       [3.0, 4.5],
       [5.0, 1.0]]
    
    X_2=[
        [1,0,1,0],
        [0,1,0,1],
        [1,1,0,0],
        [0,0,1,1],
        [1,0,0,1],
        [0,1,1,0]
    ]
    
    y_classification=[0, 1, 0, 1, 3, 0]
    y_regression=[2.0, 3.5, 2.5, 4.0, 3.0, 5.0]

    model_c_continuous=DecisionTreeClassifier(max_depth=3)
    model_c_continuous.fit(X_1,y_classification)
    predictions_c_continuous=model_c_continuous.predict(X_1)
    print("分类器(连续特征)预测结果:",predictions_c_continuous)  # 输出分类结果

    model_c_discrete=DecisionTreeClassifier(max_depth=3)
    model_c_discrete.fit(X_2,y_classification)
    predictions_c_discrete=model_c_discrete.predict(X_2)
    print("分类器(离散特征)预测结果:",predictions_c_discrete)  # 输出分类结果


    model_r_continuous=DecisionTreeRegressor(max_depth=3)
    model_r_continuous.fit(X_1,y_regression)
    predictions_r_continuous=model_r_continuous.predict(X_1)
    print("回归器(连续特征)预测结果:",predictions_r_continuous)  # 输出回归结果

    model_r_discrete=DecisionTreeRegressor(max_depth=3)
    model_r_discrete.fit(X_2,y_regression)
    predictions_r_discrete=model_r_discrete.predict(X_2)
    print("回归器(离散特征)预测结果:",predictions_r_discrete)  # 输出回归结果           




