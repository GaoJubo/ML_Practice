import numpy as np
from utils import check_Feature_Label_Alignment

class SVM:
    """
    一个使用梯度下降实现的简单线性支持向量机（Soft Margin）。
    仅使用 NumPy 进行二分类。
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=100):
        # 学习率
        self.lr = learning_rate
        # 正则化参数（对应于 1/C，此处为了方便，直接用 lambda_param 代替 C 的倒数，
        # 在更新公式中实现 C * Hinge Loss 的效果，通常 C 的值越大，容忍的错误越少。
        # 这里的 lambda_param 相当于目标函数中的 1/C，但在梯度更新中体现为 L2 正则化的系数。
        # 严格来说，这里的 lambda_param 对应于 Scikit-learn 中的 C 的倒数，即：
        # C = 1 / lambda_param
        self.lambda_param = lambda_param 
        # 迭代次数
        self.n_iters = n_iters
        # 权重 w 和 偏差 b
        self.w = None
        self.b = None

        self._X_dim=None

    def fit(self, X, y):
        """
        训练 SVM 模型。
        X: 训练数据 (n_samples, n_features)
        y: 标签 (n_samples,); 必须是 -1 或 1
        """
        X,y=np.asarray(X,np.float64),np.asarray(y,np.float64)
        X,y=check_Feature_Label_Alignment(X,y)
        n_samples, n_features = X.shape
        self._X_dim=n_features


        y = np.where(y <= 0, -1, 1)
        if not all(val in [-1, 1] for val in np.unique(y)):
            raise ValueError("标签 y 必须是 -1 或 1")

        
        # 初始化权重 w 和 偏差 b
        # 使用小的随机值或零初始化
        self.w = np.zeros(n_features)
        self.b = 0

        # 梯度下降训练
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # 计算 y_i * (w * x_i + b)
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # 样本落在边界或正确一侧 (满足 margin 要求)
                    # 此时 Hinge Loss 梯度为 0
                    # 仅更新正则化项的梯度
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # 样本是误分类点或位于 Margin 内部 (违反 margin 要求)
                    # 此时 Hinge Loss 梯度不为 0
                    # 梯度 = 正则化项梯度 + Hinge Loss 梯度
                    dw = 2 * self.lambda_param * self.w - y[idx] * x_i
                    db = -y[idx]

                # 更新 w 和 b
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        """
        进行预测。
        返回：-1 或 1
        """
        X=np.asarray(X,np.float64)
        if self._X_dim!=X.shape[1]:raise ValueError('预测集特征数与训练集上不符合')

        # 计算决策函数 w * x + b
        approx = np.dot(X, self.w) + self.b
        # 返回符号函数 (signum function)
        return np.where(np.sign(approx)==-1,0,1)
    

if __name__=='__main__':

    data=np.load(r'E:\someShy\ML_Practice\backend\dataset\mushroom.npy')
    X=data[:,1:]
    y=data[:,0]

    # print(np.unique(y))

    model_c=SVM()
    model_c.fit(X,y)
    y_hat=model_c.predict(X)

    print(f'二分类SVM的准确率为{np.mean((y_hat==y).astype(int))}')