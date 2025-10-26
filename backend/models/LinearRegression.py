import numpy as np
from scipy import linalg
import scipy as sp
from utils import check_Feature_Label_Alignment
from utils import normalize_data

class LinearRegression():
    '''
    使用三种求解方法分别对应两种正则化项和无正则化项
    '''
    def __init__(self,normalize=False,n_jobs=-1,
                penalty='none',alpha=1.0,
                max_iter=10000,tol=1e-3,learning_rate=0.01,random_seed=None):
        
        self.normalize:bool = normalize
        self.n_jobs:int = n_jobs
        self.penalty:str = penalty
        self.alpha:float = alpha
        self.max_iter:int = max_iter
        self.tol:float = tol
        self.learning_rate:float = learning_rate
        self.random_seed:int = random_seed

        self.coef_=None
        self._X_dim=None
        self._X_mean=None
        self._X_std=None
        self._y_mean=None
        self._y_std=None
        
    
    def _fit_normal(self,X_processed,y_processed):
        X_with_bias=np.c_[np.ones((X_processed.shape[0],1)),X_processed]
        W_full, _ , _ , _ = linalg.lstsq(X_with_bias, y_processed)
        return W_full
    
    def _fit_ridge(self,X_processed,y_processed):
        X_with_bias=np.c_[np.ones((X_processed.shape[0],1)),X_processed]
        n_features=X_with_bias.shape[1]
        I=np.eye(n_features)
        I[0,0]=0
        W_full=sp.linalg.solve(X_with_bias.T@X_with_bias+self.alpha*I,X_with_bias.T@y_processed,assume_a='pos')
        return W_full

    def _fit_l1(self,X_processed,y_processed):
        N,D=X_processed.shape
        X_full=np.c_[np.ones((N,1)),X_processed]
        W=np.zeros((D+1,y_processed.shape[1]))

        for i in range(self.max_iter):
            y_pred=X_full@W
            gradient=2*X_full.T@(y_pred-y_processed)/N

            W_new=W-self.learning_rate*gradient
            W_new[1:,:]=np.sign(W_new[1:,:])*np.maximum(0,np.abs(W_new[1:,:])-self.learning_rate*self.alpha)
            if np.sum(np.abs(W_new-W))<self.tol:
                W=W_new
                break
            W=W_new

            # if i%10==0:
            #     print(f'迭代{i},MSE={np.mean((y_pred-y_processed)**2)}')

        return W

    def fit(self,X,y):
        #数据校验
        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(y,dtype=np.float64)
        # print(f'原始输入 {X.shape} {y.shape}')
        X,y=check_Feature_Label_Alignment(X,y)
        self._X_dim=X.shape[1]
        # print(f'维度检查 {X.shape} {y.shape}')


        #数据标准化
        X,self._X_mean, self._X_std, y , self._y_mean, self._y_std= \
        (*normalize_data(X),*normalize_data(y)) if self.normalize else (X,np.zeros(X.shape[1]),np.ones(X.shape[1]),y,np.zeros(y.shape[1]),np.ones(y.shape[1]))
        # print(f'标准化 {X.shape} {y.shape}')

        #根据不同的正则化选项进行训练
        if self.penalty=='none':
            W_full=self._fit_normal(X,y)
            self.coef_=W_full.T
        elif self.penalty=='l2':
            if self.alpha<=0:
                raise ValueError('岭回归的正则化参数alpha必须大于0')
            if self.learning_rate<=0:
                raise ValueError('学习率必须大于0')
            W_full=self._fit_ridge(X,y)
            self.coef_=W_full.T
        elif self.penalty=='l1':
            if self.alpha<=0:
                raise ValueError('Lasso回归的正则化参数alpha必须大于0')
            if self.learning_rate<=0:
                raise ValueError('学习率必须大于0')
            W_full=self._fit_l1(X,y)
            self.coef_=W_full.T
            
         

    def predict(self,X):
        X=np.asarray(X,dtype=np.float64)
        if X.shape[1]!=self._X_dim:
            raise ValueError(f'预测与训练的数据维度不匹配,训练数据维度{self._X_dim},预测数据维度{X.shape[1]}')
        X=(X-self._X_mean)/self._X_std

        X_with_bias=np.c_[np.ones((X.shape[0],1)),X]
        y=X_with_bias@self.coef_.T
        y=self._y_std*y+self._y_mean
        return y
    

if __name__=='__main__':
    X_train=np.load(r'E:\someShy\ML_Practice\backend\dataset\regression\auto_mpg.npy')[:,1:]
    Y_train=np.load(r'E:\someShy\ML_Practice\backend\dataset\regression\auto_mpg.npy')[:,0]
    

    for penalty in ['none','l2','l1']:
        model=LinearRegression(normalize=True,penalty=penalty,alpha=0.1,learning_rate=0.01)
        model.fit(X_train,Y_train)
        print(f'penalty={penalty},MSE={np.mean((model.predict(X_train)-Y_train)**2)}')
