try:
    import cupy as cp
    xp=cp
    is_cupy=True
    print("使用cupy加速计算")
except ImportError:
    import numpy as np
    xp=np
    is_cupy=False
    print("未安装cupy,使用numpy进行计算")



import numpy as np

class PCA:
    def __init__(self,n_components):

        self.n_components = n_components

        self.mean_ = None
        self.components_ = None

        self.is_cupy=is_cupy

    def fit(self,X):
        X=xp.asarray(X)    


        self.mean_=xp.mean(X,axis=0)
        X_centered = X - self.mean_

        cov_marix = xp.cov(X_centered.T)

        eigenvalues, eigenvectors = xp.linalg.eigh(cov_marix)

        index=xp.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:,index]

        if self.n_components==None:
            self.components_=eigenvectors
        else:
            self.components_ = eigenvectors[:,:self.n_components]

    def transform(self,X):
        X=xp.asanyarray(X)
        X_centered = X - self.mean_

        X_projected=X_centered@self.components_

        if self.is_cupy:
            return X_projected.get()
        else:
            return X_projected
    
if __name__ == "__main__":
    import numpy as np
    import cupy as cp
    import time

    # X_range=[np.load(r'E:\someShy\ML_Practice\backend\dataset\mushroom.npy')]
    X_range=[np.random.rand(100000,3000)]
    
    for xp in [cp,np]:
        is_cupy=(xp==cp)
        for X in X_range:
            start=time.time()
            pca=PCA(n_components=int(X.shape[1]/2))
            pca.fit(X)
            X_pca=pca.transform(X)
            end=time.time()
            print(f"使用{xp.__name__},数据形状{X.shape},降维后形状{X_pca.shape},耗时{end-start:.4f}秒")


#根据我的测试,np和cp在这段程序上的性能有四个大区间
#1.数据量较小(如100*100),cp明显慢于np,因为数据传输和初始化开销占比大
#2.数据量中等,cp快于np
#3.数据量较大(10000,5000),cp略快于np,但性能提升不明显,大概百分之20
#4.数据量非常大(如100000*3000),cp远快于np,性能提升非常明显,大概在2倍
#但是之后显存会爆