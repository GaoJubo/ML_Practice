import sys
import os

target_path = os.path.join(os.getcwd(), 'models')#main.py将在backend目录下启动
sys.path.insert(0, target_path)
# print(sys.path)

import numpy as np
from DecisionTree import DecisionTreeRegressor,Node
from abc import ABC, abstractmethod
from utils import check_Feature_Label_Alignment
from joblib import Parallel, delayed



class BasicRegressor(DecisionTreeRegressor):
    '''
    基础回归树,用于GBM,该树和原始树完全一样,这里重新集成一下仅仅是为了对称美观,和名称统一
    '''

class BasicClassifier(DecisionTreeRegressor):
    '''
    基础分类树,用于GBM分类
    '''
    def _calculate_leaf_value(self, r:np.ndarray, p:np.ndarray)->np.float64:
        '''
        实现最优叶子节点值的计算
        param r: 当前轮迭代的残差向量
        param p: 前一轮迭代后的预测概率值向量
        '''
        numerator = np.sum(r)

        Hessian = p * (1 - p)
        denominator = np.sum(Hessian)

        epsilon = 1e-15
        gamma = numerator / (denominator + epsilon)
        #这段不懂数学上怎么推导的,先记着吧

        return gamma
    
    def _grow_tree(self,X,y,p,depth=0):
        '''
        递归生成决策树
        param p: 前一轮迭代后的预测概率值向量
        !!! 注意这里的y是概率伪残差向量 !!!
        '''
        if any([
            depth>=self.max_depth,
            X.shape[0]<self.min_samples_split,
            self._calculate_loss(y)==0
        ]) :
            return Node(value=self._calculate_leaf_value(y,p))
        
        feature_index,threshold,_=self._best_split(X,y)
        if feature_index is None:
            return Node(value=self._calculate_leaf_value(y,p))
        
        left_indices=X[:,feature_index]<=threshold
        right_indices=X[:,feature_index]>threshold
        left_node=self._grow_tree(X[left_indices],y[left_indices],p[left_indices],depth+1)
        right_node=self._grow_tree(X[right_indices],y[right_indices],p[right_indices],depth+1)

        return Node(feature_index=feature_index,threshold=threshold,left=left_node,right=right_node)
    
    def fit(self, X, y, p):
        '''
        训练
        '''
        X,y=np.array(X,dtype=np.float64),np.array(y,dtype=np.float64)
        check_Feature_Label_Alignment(X, y)
        y=y.reshape(-1,1)#确保y是二维的,列向量
    
        self.n_features_=X.shape[1]
        self.root=self._grow_tree(X,y,p)

class GBM(ABC):
    def __init__(self,n_estimators=100,learning_rate=0.1,max_depth=3,n_jobs=-1,verbose=False):
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.maxth_depth=max_depth
        self.n_jobs=-1
        self.verbose=verbose

        self._X_dim=None
        self._trees=[]
        self.F0=None
    
    @abstractmethod
    def _intialF0(y):
        pass

class GBM_R(GBM):

    def _intialF0(self,y):
        return np.mean(y,axis=0)

    def fit(self,X,y):
        #数据预处理
        X,y=np.asarray(X,np.float64),np.asarray(y,np.float64)
        X,y=check_Feature_Label_Alignment(X,y)
        self._X_dim=X.shape[1]

        self._trees=[]

        self.F0=self._intialF0(y)
        current_F=np.full(y.shape,self.F0)
        for i in range(self.n_estimators):
            res=y-current_F
            meta_r=BasicRegressor(max_depth=self.maxth_depth)
            meta_r.fit(X,res)
            f_i=meta_r.predict(X)

            self._trees.append(meta_r)
            current_F+=self.learning_rate*f_i
            if self.verbose: print(f'构建第{i}棵基础树')

        return self
    
    def predict(self,X):
        X=np.asarray(X,dtype=np.float64)
        if X.shape[1]!=self._X_dim:
            raise ValueError('预测数据的特征数量与训练数据不符合')
        
        def meta_predict(tree,X):
            return tree.predict(X)
        
        results_list=Parallel(n_jobs=self.n_jobs)(
            delayed(meta_predict)(tree,X)
            for tree in self._trees
        )

        results_list=np.array(results_list)
        result=np.sum(results_list,axis=0)*self.learning_rate
        result+=self.F0

        return self.F0
    
class GBM_C(GBM):
    def _sigmoid(self,z):
        return 1.0/(1.0+np.exp(-np.clip(z,-700,700)))
    
    def _intialF0(self,y):
        p_init=np.mean(y)
        return np.log(p_init/(1-p_init))
    

    def fit(self,X,y):
        X,y=np.asarray(X,np.float64),np.asarray(y,np.float64)
        X,y=check_Feature_Label_Alignment(X,y)
        self._X_dim=X.shape[1]

        self._trees=[]

        self.F0=self._intialF0(y)
        print(f'y.shape:{y.shape}')
        F_current=np.full(y.shape,self.F0)
        print(f'F_current{F_current.shape}')
        print(f'{self.F0}')

        for i in range(self.n_estimators):
            p_current=self._sigmoid(F_current)

            res=y-p_current
            tree=BasicClassifier(max_depth=self.maxth_depth)
            tree.fit(X,res,p_current)

            F_i=tree.predict(X)
            self._trees.append(tree)

            # print(F_current.shape)
            # print((self.learning_rate*F_i).shape)
            delta=self.learning_rate*F_i
            F_current=F_current+delta.reshape(-1,1)
            if self.verbose: print(f'构建第{i}棵基础树')

    
        return self
    
    def _predict_prob(self,X):
        X=np.asarray(X,np.float64)
        if X.shape[1]!=self._X_dim:
            raise ValueError('预测数据的特征数与训练数据不符')
        
        def meta_predict(tree,X):
            return tree.predict(X)
        
        results_list=Parallel(n_jobs=self.n_jobs)(
            delayed(meta_predict)(tree,X)
            for tree in self._trees
        )

        results_list=np.array(results_list)
        result=np.sum(results_list,axis=0)*self.learning_rate
        result+=self.F0

        return self._sigmoid(result)
    
    def predict(self,X,threshold=0.5):
        X=np.asarray(X,np.float64)
        if X.shape[1]!=self._X_dim:
            raise ValueError('预测数据的特征数与训练数据不符')
        proba=self._predict_prob(X)
        return (proba>=threshold).astype(int)


        

        
        







if __name__=='__main__':
    # X_r=np.random.rand(1000,20)
    # y_r=np.random.rand(1000,1)

    # model_r=GBM_R(n_estimators=5,verbose=True)
    # model_r.fit(X_r,y_r)
    # y_hat=model_r.predict(X_r)

    # print(f'回归GBM的mse:{np.mean((y_hat-y_r)**2)}')

    data=np.load(r'E:\someShy\ML_Practice\backend\dataset\mushroom.npy')
    X=data[:,1:]
    y=data[:,0]

    model_c=GBM_C(n_estimators=5,verbose=True)
    model_c.fit(X,y)
    y_hat=model_c.predict(X)
    print(f'二分类GBM的准确率为{np.mean((y_hat==y).astype(int))}')

    











        
        
