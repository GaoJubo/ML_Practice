import numpy as np
from utils import check_Feature_Label_Alignment
from utils import normalize_data

class LogisticRegression():
    '''
    利用逻辑回归进行二分类,实现了弹性网络正则化,提供三种求解器
    '''
    def __init__(self,normalize=False,learning_rate=0.01,max_iter=5000,tol=1e-3,
                lmbda=0.5,l1_ratio=0.5,
                solver='pgd'):
        '''
        :param normalize: 是否对数据进行标准化
        :param learning_rate: 学习率
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        :param lmbda: 正则化强度
        :param l1_ratio: 弹性网络中L1正则化的比例
        :param solver: 优化方法,支持'pgd'
        '''

        self.normalize:bool = normalize
        self.learning_rate:float = learning_rate
        self.max_iter:int = max_iter
        self.tol:float = tol
        self.lmbda:float = lmbda
        self.l1_ratio:float = l1_ratio
        self.solver:str = solver

        self.weights_=None
        self.bias_=None
        self._X_mean=None
        self._X_std=None
        self._X_dim=None

    def _sigmoid(self,z):
        z=np.clip(z,-700,700)
        return 1/(1+np.exp(-z))
    
    def compute_linear_output(self,X):
        return X @ self.weights_ + self.bias_
    
    def _soft_thresholding(self, v, tau):
        """软阈值算子：sgn(v) * max(0, |v| - tau)"""
        # tau 必须是一个标量，代表阈值
        return np.sign(v) * np.maximum(0, np.abs(v) - tau)

    def _fit_pgd(self,X,y):
        n_samples,n_features=X.shape
        lr=self.learning_rate
        l2_factor=self.lmbda*(1 - self.l1_ratio)

        #软阈值
        tau=lr*self.lmbda*self.l1_ratio/n_samples

        for _ in range(self.max_iter):
            w_old = np.copy(self.weights_)
            #1.前向传播
            linear_output=self.compute_linear_output(X)
            y_pred=self._sigmoid(linear_output)
            error=y_pred - y

            #2.计算光滑项梯度
            dW_loss = (1 / n_samples) * (X.T @ error)
            db_loss = (1 / n_samples) * np.sum(error)

            dW_l2_reg = (l2_factor / n_samples) * self.weights_

            dW_f = dW_loss + dW_l2_reg
            db_f = db_loss

            w_tilde = self.weights_ - lr * dW_f

            #3.软阈值近端步
            self.weights_ = self._soft_thresholding(w_tilde, tau)
            self.bias_ -= lr * db_f

            #早停检查
            weight_change = np.linalg.norm(self.weights_ - w_old)
            if weight_change < self.tol:
                print(f"早停：PGD 在第 {_+1} 次迭代收敛，权重变化量 {weight_change:.6f} < tol={self.tol}")
                break
        
    def _fit_cd(self,X,y):
        n_samples,n_features=X.shape
        l1_coeff=self.lmbda*self.l1_ratio/n_samples
        l2_coeff=self.lmbda*(1 - self.l1_ratio)/n_samples

        for _ in range(self.max_iter):
            w_old = np.copy(self.weights_)

            for j in range(self.weights_.shape[0]):
                #单变量梯度步
                error=y-self._sigmoid(self.compute_linear_output(X))

                grad_j = - (1 / n_samples) * (X[:, j].reshape(-1, 1).T @ error)
                v_j = self.weights_[j, 0] - self.learning_rate * grad_j

                #加入l1,l2惩罚
                tau_j = self.learning_rate * l1_coeff
                w_j_new = self._soft_thresholding(v_j, tau_j)
                w_j_new /= (1 + self.learning_rate * l2_coeff)

                self.weights_[j, 0] = w_j_new.item(0)
            #更新偏置
            error = y - self._sigmoid(self.compute_linear_output(X))
            db = - (1 / n_samples) * np.sum(error)
            self.bias_ -= self.learning_rate * db

            #早停检查
            weight_change = np.linalg.norm(self.weights_ - w_old)
            if weight_change < self.tol:
                print(f"早停：CD 在第 {_+1} 次迭代收敛，权重变化量 {weight_change:.6f} < tol={self.tol}")
                break

    def fit(self,X,y):
        '''
        :param X: 训练数据,shape=(n_samples,n_features)
        :param y: 训练标签,shape=(n_samples,)
        '''

        #数据校验
        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(y,dtype=np.float64)

        X,y=check_Feature_Label_Alignment(X,y)
        self._X_dim=X.shape[1]

        #数据标准化
        if self.normalize:
            X,self._X_mean,self._X_std=normalize_data(X)
        else:
            self._X_mean=np.zeros(self._X_dim)
            self._X_std=np.ones(self._X_dim)
        
        #参数初始化
        self.weights_=np.zeros((self._X_dim,1))
        self.bias_=0.0

        #选择并运行求解器
        if self.solver=='pgd':
            self._fit_pgd(X,y)
        elif self.solver=='cd':
            self._fit_cd(X,y)
        else:
            raise ValueError(f'不支持的优化方法: {self.solver}')
        
    def predict_proba(self,X):
        if self.normalize:
            X = (X - self._X_mean) / self._X_std
        
        return self._sigmoid(self.compute_linear_output(X))
    
    def predict(self,X,threshold=0.5):
        probabilities=self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

        
if __name__=='__main__':
    np.random.seed(42)
    #生成简单数据集
    X=np.random.randn(1000,125)
    X[:,99]=X[:,1]*100
    Z=X@np.random.randn(125).reshape(-1,1)+np.random.randn(1000,1)*0.5
    P=1/(1+np.exp(-Z))
    y=(P>0.5).astype(int)

    #测试
    solvers=['pgd','cd']
    paramters=[
                {'l1_ratio':0.5},
                {'l1_ratio':0},
                {'l1_ratio':1}
                ]

    for solver in solvers:
        for paramater in paramters:
            model=LogisticRegression(**paramater,solver=solver)
            model.fit(X,y)
            acc=np.mean((model.predict(X)==y).astype(int))
            print(f'求解器-{solver},参数:{paramater},--平均分类准确率--:{acc:.8f}\n')
            
        


    
