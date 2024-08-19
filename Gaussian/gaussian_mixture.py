from sklearn.mixture import GaussianMixture
import numpy as np

# 示例数据
data = np.random.rand(100, 2)  # 生成一些随机数据

# 初始化GMM模型
gmm = GaussianMixture(n_components=2, random_state=0)  # 假设有两个类簇

# 拟合模型
gmm.fit(data)

# 预测每个数据点的类簇
labels = gmm.predict(data)

# 打印均值和协方差
print("均值：", gmm.means_)
print("协方差：", gmm.covariances_)

# 打印混合系数
print("混合系数：", gmm.weights_)
