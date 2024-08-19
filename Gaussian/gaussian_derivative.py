import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# 生成高斯衍射核
def gaussian_derivative(sigma):
    size = int(np.ceil(sigma * 3)) * 2 + 1
    x, y = np.meshgrid(np.arange(-size//2 + 1, size//2 + 1), np.arange(-size//2 + 1, size//2 + 1))
    Gx = -x * np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4)
    Gy = -y * np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4)
    return Gx, Gy

# 使用高斯衍射核进行边缘检测
def apply_gaussian_derivative(image, sigma=1.0):
    Gx, Gy = gaussian_derivative(sigma)
    grad_x = cv2.filter2D(image, -1, Gx)
    grad_y = cv2.filter2D(image, -1, Gy)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude

# 加载图像并转换为灰度
image = cv2.imread(r'D:\MyProjects\PointCloudRegistration\KeyPointDetection\datas\cat01.jpg', cv2.IMREAD_GRAYSCALE)

# 应用高斯衍射算法
result = apply_gaussian_derivative(image, sigma=1.5)

# 标准化到0-255范围并转换为uint8
result_float32 = np.float32(result)
result_normalized = cv2.normalize(result_float32, None, 0, 255, cv2.NORM_MINMAX)
result_uint8 = result_normalized.astype(np.uint8)

# 显示结果
cv2.imshow('Gaussian Derivatives', result_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()
