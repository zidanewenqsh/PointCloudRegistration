import numpy as np
import open3d as o3d

def load_point_cloud(filename):
    # 直接返回PointCloud对象
    return o3d.io.read_point_cloud(filename)

def compute_covariance_matrix(points):
    mean = np.mean(points, axis=0)
    cov = np.zeros((3, 3))
    for p in points:
        dp = p - mean
        cov += np.outer(dp, dp)
    return cov / len(points)

def harris_3d(points, k=0.04, radius=48):
    points_arr = np.asarray(points.points)
    tree = o3d.geometry.KDTreeFlann(points) 
    # downpcd = pcd.voxel_down_sample(voxel_size = 10)
    # kdtree = o3d.geometry.KDTreeFlann(downpcd)

    response = np.zeros(len(points_arr))
    
    for i in range(len(points_arr)):
        # print(points[i])
        # point = points[i]
        try:
            k, idx, _ = tree.search_radius_vector_3d(points_arr[i], radius)
            if k < 3:  # 至少需要3个点来计算协方差矩阵
                continue
            neighbors = points_arr[idx, :]
            cov = compute_covariance_matrix(neighbors)
            eigenvalues = np.linalg.eigvalsh(cov)
            R = np.prod(eigenvalues) - k * np.sum(eigenvalues)**2
            response[i] = R
        except RuntimeError as e:
            print(f"在点索引 {i} 处发生错误：{e}")
            # print(point)
            # print(point.dtype)
            # print(type(point))
            exit(-1)
        # k, idx, _ = tree.search_radius_vector_3d(points[i], radius)
        # if k < 3:  # 至少需要3个点来计算协方差矩阵
        #     continue
        # neighbors = points[idx, :]
        # cov = compute_covariance_matrix(neighbors)
        # eigenvalues = np.linalg.eigvalsh(cov)
        # R = np.prod(eigenvalues) - k * np.sum(eigenvalues)**2
        # response[i] = R
    
    return response

# 示例代码执行
filename = 'D:\MyProjects\PointCloudRegistration\KeyPointDetection\datas\pig_view1.pcd'
pcd = load_point_cloud(filename)
pcd = pcd.voxel_down_sample(voxel_size = 10)
response = harris_3d(pcd)
points_arr = np.asarray(pcd.points)
corners = points_arr[response > np.percentile(response, 95)]  # 取前5%响应值最高的点作为角点

# 角点可视化
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(corners)
# o3d.visualization.draw_geometries([pc])
# 设置原始点云的颜色为绿色
green_color = [0, 1, 0]  # RGB 颜色，绿色
num_points_pcd = np.asarray(pcd.points).shape[0]
pcd.colors = o3d.utility.Vector3dVector([green_color] * num_points_pcd)

# 设置角点的颜色为红色
red_color = [1, 0, 0]  # RGB 颜色，红色
num_points_pc = np.asarray(pc.points).shape[0]
pc.colors = o3d.utility.Vector3dVector([red_color] * num_points_pc)

# 显示两个点云
# o3d.visualization.draw_geometries([pcd, pc])
o3d.visualization.draw_geometries([pc, pcd])
# o3d.visualization.draw_geometries([pc])