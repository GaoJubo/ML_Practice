import numpy as np

class NaiveBayes:
    """
    伯努利朴素贝叶斯二分类器 (仅支持二分类)
    适用于 0/1 二值化特征。
    """
    def __init__(self, alpha=1.0):
        # 拉普拉斯平滑参数 (通常为 1.0)
        self.alpha = alpha
        
        # 存储类别标签 (假设为 0 和 1)
        self.classes_ = None
        # 存储先验概率 log P(C_k)
        self.class_log_prior_ = None
        # 存储参数 log P(x_i=1 | C_k)
        self.feature_log_prob_ = None
        # 存储参数 log P(x_i=0 | C_k)
        self.neg_feature_log_prob_ = None


    def fit(self, X, y):
        """
        训练模型: 计算先验概率和特征条件概率 (使用拉普拉斯平滑)
        X: 训练数据 (N_samples, N_features)，要求为 0 或 1
        y: 标签 (N_samples,)，要求为 0 或 1
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        if n_classes != 2:
            raise ValueError("此实现仅支持二分类问题。")

        # 形状: (N_classes, )
        class_count = np.zeros(n_classes)
        # 形状: (N_classes, N_features)
        feature_count = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            # 筛选出属于当前类别 c 的样本
            X_c = X[y == c]
            
            # 1. 统计类别样本总数 N_k
            class_count[idx] = X_c.shape[0]
            
            # 2. 统计特征 i 在类别 c 下出现次数 N_ik (即 X_c 中值为 1 的总和)
            # np.sum(..., axis=0) 沿着行求和，得到每列(特征)为 1 的计数
            feature_count[idx, :] = np.sum(X_c, axis=0)

        # === 1. 计算 Log 先验概率 P(C_k) ===
        # P(C_k) = (N_k + alpha - 1) / (N_total + alpha * N_classes - N_classes) (拉普拉斯平滑的变体)
        # 简化版 (常用): log P(C_k) = log(N_k / N_total)
        # 考虑到 alpha=1 的情况，P(C_k) = (N_k + 1) / (N_total + 2)
        n_total = X.shape[0]
        # P(C_k) = (N_k + alpha) / (N_total + alpha * N_classes)
        prior = (class_count + self.alpha) / (n_total + self.alpha * n_classes)
        self.class_log_prior_ = np.log(prior)


        # === 2. 计算 Log 特征条件概率 P(x_i | C_k) ===

        # 伯努利分布的参数 theta_ik = P(x_i=1 | C_k)
        # theta_ik = (N_ik + alpha) / (N_k + alpha * 2)
        # N_k 是 class_count，N_ik 是 feature_count
        theta = (feature_count + self.alpha) / (class_count[:, np.newaxis] + self.alpha * 2)

        # 存储 Log 概率
        self.feature_log_prob_ = np.log(theta)         # log P(x_i=1 | C_k)
        self.neg_feature_log_prob_ = np.log(1 - theta) # log P(x_i=0 | C_k)

        return self


    def predict_log_proba(self, X):
        """
        计算 Log 后验概率: log P(C_k) + sum(log P(x_i | C_k))
        X: 待预测数据 (N_samples, N_features)，要求为 0 或 1
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Log 后验概率矩阵，形状: (N_samples, N_classes)
        log_proba = np.zeros((n_samples, n_classes))

        # X_i * log P(x_i=1 | C_k) + (1 - X_i) * log P(x_i=0 | C_k)
        # X 形状: (N_samples, N_features)
        # self.feature_log_prob_ 形状: (N_classes, N_features)
        
        for idx in range(n_classes):
            # 计算 log P(x|C_k) = sum_i log P(x_i|C_k)
            # log P(x_i|C_k) = X * log P(x_i=1|C_k) + (1-X) * log P(x_i=0|C_k)

            # X * log P(x_i=1 | C_k): 仅在 x_i=1 时起作用
            log_lik_1 = X * self.feature_log_prob_[idx, :]
            
            # (1 - X) * log P(x_i=0 | C_k): 仅在 x_i=0 时起作用
            log_lik_0 = (1 - X) * self.neg_feature_log_prob_[idx, :]
            
            # log P(x|C_k) 是它们的总和 (对特征求和)
            # log_likelihood 形状: (N_samples,)
            log_likelihood = np.sum(log_lik_1 + log_lik_0, axis=1)

            # Log 后验概率: log P(C_k) + log P(x|C_k)
            log_proba[:, idx] = self.class_log_prior_[idx] + log_likelihood
            
        return log_proba
        
    def predict(self, X):
        """
        预测类别标签
        """
        log_proba = self.predict_log_proba(X)
        
        # 找出 log_proba 最大的索引，即为预测类别
        # np.argmax(..., axis=1) 返回每行最大值的索引
        predictions = self.classes_[np.argmax(log_proba, axis=1)]
        
        return predictions
    

if __name__ == "__main__":
    # 简单测试
    X_train = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 1, 1],
                        [0, 0, 0],
                        [1, 0, 0]])
    y_train = np.array([1, 0, 1, 0, 1])

    model = NaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)

    X_test_1 = np.array([[1, 0, 1],
                       [0, 1, 0],
                       [0, 0, 1]])
    
    X_test_2 = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 1, 1],
                        [0, 0, 0],
                        [1, 0, 0]])
    
    predictions1 = model.predict(X_test_1)
    print("Predictions_1:", predictions1)

    predictions2 = model.predict(X_test_2)
    print("Predictions_2:", predictions2)