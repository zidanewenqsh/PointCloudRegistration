import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 生成数据
np.random.seed(0)
data1 = np.random.normal(loc=[2, 2], scale=0.5, size=(100, 2))
data2 = np.random.normal(loc=[6, 6], scale=0.5, size=(100, 2))
data = np.vstack((data1, data2))

# 手动实现 GMM
def initialize_parameters(data, n_components):
    idx = np.random.choice(len(data), n_components, replace=False)
    means = data[idx]
    covariances = [np.cov(data, rowvar=False)] * n_components
    weights = np.ones(n_components) / n_components
    return means, covariances, weights

def e_step(data, means, covariances, weights):
    n, d = data.shape
    k = len(weights)
    responsibilities = np.zeros((n, k))
    for j in range(k):
        dist = data - means[j]
        inv_cov = np.linalg.inv(covariances[j])
        exponent = np.diag(-0.5 * np.dot(np.dot(dist, inv_cov), dist.T))
        coeff = 1 / np.sqrt((2 * np.pi)**d * np.linalg.det(covariances[j]))
        responsibilities[:, j] = weights[j] * coeff * np.exp(exponent)
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

def m_step(data, responsibilities):
    n, d = data.shape
    k = responsibilities.shape[1]
    n_k = responsibilities.sum(axis=0)
    weights = n_k / n
    means = np.dot(responsibilities.T, data) / n_k[:, None]
    covariances = []
    for j in range(k):
        dist = data - means[j]
        cov = np.dot(responsibilities[:, j] * dist.T, dist) / n_k[j]
        covariances.append(cov)
    return means, covariances, weights

def gmm_fit(data, n_components, n_iter):
    means, covariances, weights = initialize_parameters(data, n_components)
    for _ in range(n_iter):
        responsibilities = e_step(data, means, covariances, weights)
        means, covariances, weights = m_step(data, responsibilities)
    return means, covariances, weights, responsibilities

# 手动 GMM 聚类
means, covs, weights, responsibilities = gmm_fit(data, 2, 10)
manual_labels = responsibilities.argmax(axis=1)

# sklearn GMM 聚类
model = GaussianMixture(n_components=2, random_state=0)
model.fit(data)
sklearn_labels = model.predict(data)

# 可视化
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=manual_labels, cmap='viridis', marker='o', label='Manual GMM')
plt.title('Manual GMM Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=sklearn_labels, cmap='viridis', marker='o', label='Sklearn GMM')
plt.title('Sklearn GMM Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
