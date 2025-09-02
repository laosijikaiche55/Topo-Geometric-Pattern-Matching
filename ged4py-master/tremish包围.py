import numpy as np
import trimesh
import ifcopenshell
from 几何点集获取 import get_pointset
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_obb(vertices, obb_center, obb_extent, quat):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 将列表转换为 NumPy 数组
    vertices = np.array(vertices)

    # 绘制原始图形的点（绿色）
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='g', marker='o')

    # 绘制包围盒的边
    obb_vertices = np.array([
        obb_center + np.dot(np.diag(obb_extent), np.array([1, 1, 1])),
        obb_center + np.dot(np.diag(obb_extent), np.array([-1, 1, 1])),
        obb_center + np.dot(np.diag(obb_extent), np.array([-1, -1, 1])),
        obb_center + np.dot(np.diag(obb_extent), np.array([1, -1, 1])),
        obb_center + np.dot(np.diag(obb_extent), np.array([1, 1, -1])),
        obb_center + np.dot(np.diag(obb_extent), np.array([-1, 1, -1])),
        obb_center + np.dot(np.diag(obb_extent), np.array([-1, -1, -1])),
        obb_center + np.dot(np.diag(obb_extent), np.array([1, -1, -1])),
    ])

    # 定义包围盒的边
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 正面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 背面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 连接正面和背面
    ]

    # 绘制包围盒的顶点（蓝色）
    ax.scatter(obb_vertices[:, 0], obb_vertices[:, 1], obb_vertices[:, 2], c='b', marker='o')

    # 绘制包围盒的边
    for edge in edges:
        ax.plot([obb_vertices[edge[0], 0], obb_vertices[edge[1], 0]],
                [obb_vertices[edge[0], 1], obb_vertices[edge[1], 1]],
                [obb_vertices[edge[0], 2], obb_vertices[edge[1], 2]], 'r')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.axis('equal')

    plt.show()


def compute_obb2(vertices):
    # 创建一个 Trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices)

    # 计算OBB
    obb = mesh.bounding_box_oriented

    # 获取OBB的信息
    obb_center = obb.centroid
    obb_extent = obb.extents * 0.5

    # 获取变换矩阵
    transform_matrix = obb.transform

    # 提取旋转矩阵部分并创建一个副本
    rotation_matrix = np.array(transform_matrix[:3, :3])

    # 将旋转矩阵转换为四元数
    quat = R.from_matrix(rotation_matrix).as_quat()

    return obb_center, obb_extent, quat


def get_obb_coordinate_system(obb_center, obb_extent, quat):
    """
    获取包围盒的坐标系。

    参数:
    obb_center (np.array): 包围盒的中心点 [x, y, z]
    obb_extent (np.array): 包围盒的尺寸范围 [width, height, depth]
    quat (np.array): 包围盒的旋转四元数 [w, x, y, z]

    返回:
    origin (np.array): 包围盒的原点
    x_axis (np.array): 包围盒的 X 轴向
    y_axis (np.array): 包围盒的 Y 轴向
    z_axis (np.array): 包围盒的 Z 轴向
    """
    # 从四元数计算旋转矩阵
    rotation = R.from_quat(quat)
    rotation_matrix = rotation.as_matrix()

    # 定义包围盒的坐标系
    origin = obb_center

    # 坐标系的三个轴
    x_axis = rotation_matrix[:, 0] * obb_extent[0]
    y_axis = rotation_matrix[:, 1] * obb_extent[1]
    z_axis = rotation_matrix[:, 2] * obb_extent[2]

    return origin, x_axis, y_axis, z_axis


ifc_file_path = r'F:\OneDrive\桌面\标准层与系统\包围盒子构件.ifc'
ifc_file = ifcopenshell.open(ifc_file_path)
elements = ifc_file.by_type('IfcElement')
for element in elements:
     all_coordinates, _ = get_pointset(ifc_file_path, element)
     # 使用给定的点集构建包围盒
     obb_center, obb_extent, quat = compute_obb2(all_coordinates)
     plot_obb(all_coordinates, obb_center, obb_extent, quat)
     origin, x_axis, y_axis, z_axis = get_obb_coordinate_system(obb_center, obb_extent, quat)
     print(origin, x_axis, y_axis, z_axis )


