import numpy as np
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(data, bins=20, title='Histogram', xlabel='Values', ylabel='Frequency'):
    """
    绘制一维Numpy数据的直方图。
    
    参数:
    - data: 一维Numpy数组
    - bins: 直方图的分组数量，默认为20
    - title: 直方图的标题，默认为'Histogram'
    - xlabel: X轴的标签，默认为'Values'
    - ylabel: Y轴的标签，默认为'Frequency'
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, alpha=0.75, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# # 示例用法
# data = np.random.normal(loc=0, scale=1, size=1000)  # 生成1000个正态分布的随机数据
# plot_histogram(data, bins=30, title='Normal Distribution', xlabel='Data Points', ylabel='Count')

def compute_normals(points, k=6):
    """计算点云的法线。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    return np.asarray(pcd.normals)

def select_key_points(normals, threshold=0.02):
    """选择关键点。"""
    key_points = []
    # x = np.linalg.norm(normals[i])
    for i in range(len(normals)):
        # x = np.linalg.norm(normals[i])
        # print(x)
        if np.linalg.norm(normals[i]) > threshold:
            key_points.append(i)
    return key_points

def build_feature_descriptor(points, key_points, radius=0.1, sectors=4):
    """构建特征描述符。"""
    descriptors = []
    for idx in key_points:
        descriptor = np.zeros(sectors)
        key_point = points[idx]
        for point in points:
            if np.linalg.norm(point - key_point) <= radius:
                angle = np.arctan2(point[1] - key_point[1], point[0] - key_point[0])
                sector_idx = int((angle + np.pi) / (2 * np.pi / sectors))
                descriptor[sector_idx] += 1
        descriptors.append(descriptor)
    return descriptors

# 加载点云数据
pcd = o3d.io.read_point_cloud(r"D:\MyProjects\PointCloudRegistration\KeyPointDetection\datas\pig_view1.pcd")
pcd = pcd.voxel_down_sample(voxel_size = 10)
points = np.asarray(pcd.points)

# 计算法线
normals = compute_normals(points)
print(normals.shape)
x = np.linalg.norm(normals, axis=1)
print(x.shape, x.max(), x.min())
# print()
# plot_histogram(x)
# 选择关键点
key_points = select_key_points(normals)
print(len(key_points))
# # 构建特征描述符
# descriptors = build_feature_descriptor(points, key_points)

# # 打印输出
# for descriptor in descriptors:
#     print(descriptor)
