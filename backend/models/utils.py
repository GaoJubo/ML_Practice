import numpy as py 

def check_Feature_Label_Alignment(X, y):
    """
    Check if the number of samples in features matches the number of labels.

    Parameters:
    features (np.ndarray): The feature dataset.
    labels (np.ndarray): The label dataset.

    Return:
    np.ndarray, np.ndarray: The validated feature and label datasets.
    """
    if X.ndim>2 or y.ndim>2:
        raise ValueError('训练数据的维度超过二维,这是不应该的')
    if y.ndim==1:
        y=y.reshape(-1,1)
    if X.shape[0]!=y.shape[0]:
        if X.shape[0]==y.shape[1]:
                y=y.T
        else:
            raise ValueError(f'X的样本数与y的样本数不匹配！')
        
    return X, y

def normalize_data(X):
    """
    Normalize the feature and label datasets.

    Parameters:
    features (np.ndarray): The feature dataset.

    Return:
    np.ndarray, np.ndarray, np.ndarray: The normalized feature and label datasets along with their means and standard deviations.
    """
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1e-8  # 防止除以零
    X_normalized = (X - X_mean) / X_std

    return X_normalized, X_mean, X_std