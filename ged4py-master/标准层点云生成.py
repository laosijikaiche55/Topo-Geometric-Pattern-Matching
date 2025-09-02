import ifcopenshell
import numpy as np
from 构件组生成 import get_element_meanpoint
import open3d as o3d

def build_cloud(ifc_file):
    """
    返回:
      - points_array: (N,3) 的 numpy 数组，每行是一个构件的质心 (x,y,z)
      - element_map: 与 points_array 大小一致的列表，每个元素是对应的 IfcElement 对象
    """
    element_meanpoints = {}
    elements = ifc_file.by_type("IfcElement")

    for element in elements:
        meanpoint = get_element_meanpoint(element)
        if meanpoint is not None:
            element_meanpoints[element.id()] = {
                'element': element,
                'centroid': meanpoint
            }

    # 将 'centroid' 提取为点云数组，并记录对应的 element
    centroids = []
    element_map = []
    for elem_id, data in element_meanpoints.items():
        centroids.append(data['centroid'])   # 假设是 [x, y, z] 列表或元组
        element_map.append(data['element'])

    if len(centroids) == 0:
        print("No centroids found. Check if get_element_meanpoint is working correctly.")
        return None, None

    # 转为 (N,3) numpy 数组便于后续处理
    points_array = np.array(centroids, dtype=float)

    # 输出调试信息
    print(f"Extracted {points_array.shape[0]} element centroids.")
    return points_array, element_map


def visualize_centroids_as_cloud(points_array):
    """
    使用 Open3D 将 (N,3) 的 numpy 数组可视化为点云。
    """
    if points_array is None or len(points_array) == 0:
        print("No points to visualize.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)

    o3d.visualization.draw_geometries([pcd], window_name="Element Centroids")



ifc_file = ifcopenshell.open(r'E:\原电脑d盘新\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\标准层.ifc')
points_array, element_map = build_cloud(ifc_file)
visualize_centroids_as_cloud(points_array)
