# 伪代码：点云对齐以最大化重叠点数量

# 导入必要的库
import numpy as np
import open3d as o3d


def preprocess_point_cloud(pcd, voxel_size):
    """
    点云预处理：下采样和法线估计
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down


def execute_global_registration(source_down, target_down, voxel_size):
    """
    全局配准：特征提取和 RANSAC 初始变换
    """
    distance_threshold = voxel_size * 1.5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, initial_transformation, voxel_size):
    """
    精细配准：ICP 算法
    """
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def align_point_clouds(source_pcd, target_pcd, voxel_size=0.05):
    """
    主函数：对齐两组点云
    """
    # 预处理
    source_down = preprocess_point_cloud(source_pcd, voxel_size)
    target_down = preprocess_point_cloud(target_pcd, voxel_size)

    # 全局配准
    result_global = execute_global_registration(source_down, target_down, voxel_size)

    # 精细配准
    result_icp = refine_registration(source_pcd, target_pcd, result_global.transformation, voxel_size)

    return result_icp.transformation


# 示例使用
if __name__ == "__main__":
    # 加载点云
    source = o3d.io.read_point_cloud("source.pcd")
    target = o3d.io.read_point_cloud("target.pcd")

    # 对齐点云
    transformation = align_point_clouds(source, target, voxel_size=0.05)

    # 应用变换
    source.transform(transformation)

    # 可视化结果
    o3d.visualization.draw_geometries([source, target],
                                      zoom=0.455,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])
