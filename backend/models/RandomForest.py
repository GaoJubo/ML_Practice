import numpy as np
from abc import ABC, abstractmethod
import sys
print(sys.path)
from backend.models.DecisionTree import Node,DecisionTreeClassifier,DecisionTreeRegressor

class RandomSplitMixin:
    '''
    随机划分特征的混入类
    '''
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

        np.random.seed(self.random_state)

        #为防止单次随机的特征项值唯一,进行多次随机选择
        feature_index=None
        uniques_values=None
        for _ in range(n_features):
            feature_index=np.random.choice(n_features)
            uniques_values=np.sort(np.unique(X[:,feature_index]).reshape(-1))  #随机选择一个特征进行划分
            if len(uniques_values)!=1:
                 break
        else:
            return None,None,float('inf')
             

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
    

class ClassifierTree(RandomSplitMixin,DecisionTreeClassifier):
    '''
    构建随机森林的基本分类树,只需要将特征划分的选择由遍历改为随机选择即可
    '''

class RegressorTree(RandomSplitMixin,DecisionTreeRegressor):
    '''
    构建随机森林的基本回归树,只需要将特征划分的选择由遍历改为随机选择即可
    '''

    

class RandomForest(ABC):
    '''
    随机森林基类
    '''

    def __init__(self,n_estimators:int=100,
                 max_depth:int=10,
                 min_samples_split:int=2,
                 random_state:int=313):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.random_state=random_state
        self.trees=[]

    @abstractmethod
    def _built_tree(self,X:np.ndarray,y:np.ndarray):
        '''
        构建随机森林,返回值应该是一个需要的树
        '''
        pass
    
    def fit(self,X:np.ndarray,y:np.ndarray):
        '''
        训练随机森林模型
        '''
        X,y=np.array(X,dtype=np.float64),np.array(y,dtype=np.float64)

        for i in range(self.n_estimators):

            bagging_indices=np.random.choice(len(X),size=len(X),replace=True)
            X_sample=X[bagging_indices]
            y_sample=y[bagging_indices]

            self.trees.append(self._built_tree(X_sample,y_sample))
    
    @abstractmethod
    def predict(self,X:np.ndarray)->np.ndarray:
        '''
        随机森林预测
        '''
        pass

class RandomForestClassifier(RandomForest):
    '''
    随机森林分类器
    '''

    def _built_tree(self,X:np.ndarray,y:np.ndarray):
        tree=ClassifierTree(min_samples_split=self.min_samples_split,
                            max_depth=self.max_depth,)
        tree.fit(X,y)
        return tree

    def predict(self,X:np.ndarray)->np.ndarray:
        '''
        随机森林分类预测
        '''
        X=np.array(X,dtype=np.float64)
        predictions=[]

        for tree in self.trees:
            predictions.append(tree.predict(X).reshape(-1,1))
        
        predictions=np.concatenate(predictions,axis=1)

        #多数表决
        final_predictions=[]
        for i in range(predictions.shape[0]):
            counts=np.bincount(predictions[i].astype(int))
            final_predictions.append(np.argmax(counts))
        
        return np.array(final_predictions)

class RandomForestRegressor(RandomForest):
    '''
    随机森林回归器
    '''

    def _built_tree(self,X:np.ndarray,y:np.ndarray):
        tree=RegressorTree(min_samples_split=self.min_samples_split,
                           max_depth=self.max_depth,)
        tree.fit(X,y)
        return tree

    def predict(self,X:np.ndarray)->np.ndarray:
        '''
        随机森林回归预测
        '''
        X=np.array(X,dtype=np.float64)
        predictions=[]

        for tree in self.trees:
            predictions.append(tree.predict(X).reshape(-1,1))
        
        predictions=np.concatenate(predictions,axis=1)

        #取平均值
        final_predictions=np.mean(predictions,axis=1)

        return final_predictions
    

if __name__=="__main__":

    data=np.load(r'E:\someShy\ML_Practice\backend\dataset\mushroom.npy')

    train_indices=np.random.choice(len(data),int(len(data)*0.8),replace=False)
    all_indices = np.arange(len(data))
    test_indices = np.setdiff1d(all_indices, train_indices)

    X_train=data[train_indices,1:]
    y_train=data[train_indices,0]
    X_test=data[test_indices,1:]
    y_test=data[test_indices,0]

    rfc=RandomForestClassifier(n_estimators=10,max_depth=10,min_samples_split=2,random_state=42)
    rfc.fit(X_train,y_train)
    predictions=rfc.predict(X_test)
    accuracy=np.sum(predictions==y_test)/len(y_test)    
    print(f"随机森林分类器准确率: {accuracy*100:.2f}%")

    rfr=RandomForestRegressor(n_estimators=10,max_depth=10,min_samples_split=2,random_state=42)
    rfr.fit(X_train,y_train)
    predictions=rfr.predict(X_test)
    mse=np.mean((predictions - y_test)**2)
    print(f"随机森林回归器均方误差: {mse:.4f}")
