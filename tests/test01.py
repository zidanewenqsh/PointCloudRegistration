import open3d as o3d
import numpy as np
 

#### Works 100% of the time ####
pcd = o3d.io.read_point_cloud(r'D:\MyProjects\PointCloudRegistration\KeyPointDetection\datas\pig_view1.pcd')
if 0:
    downpcd = pcd.voxel_down_sample(voxel_size = 10)
    kdtree = o3d.geometry.KDTreeFlann(downpcd)
else:
    pcd = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd) # RuntimeError: search_hybrid_vector_3d() error!

point = np.array([0,500,100] , dtype = np.float64)

point = np.array([372.64,   175.88, -2035.16])
point = np.array([372.64370728,   175.88729858, -2035.16101074])
print(point)
print(point.dtype)
print(type(point))
result = kdtree.search_hybrid_vector_3d(point, 48, 1)
result2 = kdtree.search_radius_vector_3d(point, 48)
print(result)
print(result)
#### Works 10% of the time ####
def import_pointcloud(filepath, detail):
    pcd = o3d.io.read_point_cloud(filepath)
    downpcd = pcd.voxel_down_sample(voxel_size = detail)
    kdtree = o3d.geometry.KDTreeFlann(downpcd)

    return kdtree

kdtree = import_pointcloud(r'D:\MyProjects\PointCloudRegistration\KeyPointDetection\datas\pig_view1.pcd', 10)

point = np.array([0,500,100] , dtype = np.float64)
point = np.array([372.64370728,   175.88729858, -2035.16101074])
result = kdtree.search_hybrid_vector_3d(point, 48, 1)
print(result)