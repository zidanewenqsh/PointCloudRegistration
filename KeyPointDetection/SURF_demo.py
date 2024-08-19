import numpy as np
import cv2
from scipy import ndimage

def compute_hessian_det(image, sigma):
    """ 计算Hessian矩阵行列式 """
    # 计算x和y的二阶导数
    Ixx = ndimage.gaussian_filter(image, sigma=sigma, order=(2, 0))
    Iyy = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 2))
    Ixy = ndimage.gaussian_filter(image, sigma=sigma, order=(1, 1))
    
    # 计算行列式
    return Ixx * Iyy - Ixy**2

def non_max_suppression(det, threshold):
    """ 非极大值抑制 """
    local_max = ndimage.maximum_filter(det, size=3)
    max_mask = (det == local_max)
    det_thresholded = det > threshold
    max_mask[det_thresholded == 0] = 0
    return np.nonzero(max_mask)

def detect_keypoints(image, sigma=1.6, threshold=0.1):
    """ 检测关键点 """
    det = compute_hessian_det(image, sigma)
    keypoints = non_max_suppression(det, threshold)
    return keypoints

# 加载图像并转换为灰度
image = cv2.imread(r'D:\MyProjects\PointCloudRegistration\KeyPointDetection\datas\1.jpg', cv2.IMREAD_GRAYSCALE)

# 检测关键点
keypoints = detect_keypoints(image)
# print(keypoints)
# keypoints 是一个元组，其中包含两个数组：一个为行索引，一个为列索引
rows, cols = keypoints

# # 打印每个关键点的坐标
# print("Keypoints (x, y) coordinates:")
# for y, x in zip(rows, cols):
#     print(f"({x}, {y})")

# # 可视化关键点
import matplotlib.pyplot as plt

# plt.imshow(image, cmap='gray')
# # plt.scatter([p[1] for p in keypoints], [p[0] for p in keypoints], c='r', s=2)
# plt.scatter(cols, rows, c='r', s=1)  # 在图上标记关键点
# plt.show()
# 创建图像显示布局
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1行2列的子图

# 显示原始图像
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')  # 不显示坐标轴

# 显示标记关键点的图像
axes[1].imshow(image, cmap='gray')
axes[1].scatter(cols, rows, c='r', s=1)  # 在图上以红色标记关键点
axes[1].set_title('Image with Keypoints')
axes[1].axis('off')  # 不显示坐标轴

# 显示整个图形
plt.show()
