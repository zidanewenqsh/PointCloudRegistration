import open3d as o3d
import numpy as np

def compute_iss_keypoints(pcd, salient_radius, non_max_radius, gamma_21, gamma_32):
    # 计算ISS关键点
    keypoints = []
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    
    for i in range(len(points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], salient_radius)
        
        if k < 3:
            continue
        
        neighbors = points[idx[1:], :]
        covariance_matrix = np.cov(neighbors.T)
        eigenvalues, _ = np.linalg.eig(covariance_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        # 添加判断，避免除以零
        if eigenvalues[0] == 0 or eigenvalues[1] == 0:
            continue
        if eigenvalues[1] / eigenvalues[0] < gamma_21 and eigenvalues[2] / eigenvalues[1] < gamma_32:
            [nk, nidx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], non_max_radius)
            # 仅在有足够的邻居时进行非最大抑制检查
            # if len(nidx) > 1:
            #     # 获取邻域内所有点的最大特征值
            #     max_neighbor_eigenvalue = max(
            #         np.linalg.eig(np.cov(points[j].T))[0][0] for j in nidx if j != i
            #     )
            #     # 比较当前点的最大特征值和邻域内其他点的最大特征值
            #     if eigenvalues[0] > max_neighbor_eigenvalue:
            #         keypoints.append(points[i])
            if len(nidx) > 1:
                # 确保不会越界
                if eigenvalues[0] > np.max(eigenvalues[1:]):
                    keypoints.append(points[i])
            if np.all(eigenvalues[0] > eigenvalues[idx[1:]]):
                keypoints.append(points[i])
    
    keypoints_pcd = o3d.geometry.PointCloud()
    keypoints_pcd.points = o3d.utility.Vector3dVector(np.array(keypoints))
    return keypoints_pcd

if __name__ == "__main__":
    # 加载点云
    pcd = o3d.io.read_point_cloud(r"D:\MyProjects\PointCloudRegistration\KeyPointDetection\datas\pig_view2.pcd")
    pcd = pcd.voxel_down_sample(voxel_size = 10)
    # o3d.visualization.draw_geometries([pcd])
    # exit()
    # ISS 参数
    salient_radius = 20
    non_max_radius = 10
    gamma_21 = 0.2
    gamma_32 = 0.2

    # 计算ISS关键点
    keypoints = compute_iss_keypoints(pcd, salient_radius, non_max_radius, gamma_21, gamma_32)

    # 可视化原始点云和关键点
    keypoints.paint_uniform_color([1.0, 0.0, 0.0])  # 红色表示关键点
    pcd.paint_uniform_color([0.0, 1.0, 0.0])  # 红色表示关键点
    o3d.visualization.draw_geometries([keypoints, pcd])
    # o3d.visualization.draw_geometries([pcd, keypoints])
