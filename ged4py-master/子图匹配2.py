from itertools import combinations

import networkx as nx
from networkx.algorithms import isomorphism
from scipy.spatial.transform import Rotation as R

from 构件组生成 import build_spatial_graph_from_ifc0
from 构件组生成 import build_spatial_graph_from_ifc1
from 构件组生成 import plot_spatial_graph
import ifcopenshell
from 几何点集获取 import get_pointset
import numpy as np
from tremish包围 import compute_obb2
from tremish包围 import plot_obb
from 构件组生成 import get_element_meanpoint


def edge_match(attr1, attr2):
    """
    定义边属性的匹配规则，假设我们需要匹配属性 'distance' 和 'vector'。
    """
    # print(attr1['vector'], attr2['vector'])
    # return True
    distance_match = abs(attr1['distance'] - attr2['distance']) < 1  # 距离允许的误差

    # print(vector_match)
    return distance_match


def graph_matching(main_graph, sub_graph):
    """
    实现子图匹配算法。

    :param main_graph: 主图（NetworkX 图对象）
    :param sub_graph: 子图（NetworkX 图对象）
    :return: 匹配结果，包含所有匹配的子图节点对的列表
    """

    # 使用 NetworkX 提供的子图同构算法，启用边属性匹配
    matcher = isomorphism.GraphMatcher(main_graph, sub_graph, edge_match=edge_match)

    # 查找所有匹配的子图
    matches = []

    for subgraph_mapping in matcher.subgraph_isomorphisms_iter():
        matches.append(subgraph_mapping)

    return matches


def align_subgraph(main_quat, sub_quat, sub_coordinates, main_center, sub_center):
    """
    使用四元数进行旋转配准，并调整子图的点集以对齐到主图

    :param main_quat: 主图的旋转四元数
    :param sub_quat: 子图的旋转四元数
    :param sub_coordinates: 子图的点集，形状为 (n, 3)
    :param main_center: 主图的质心
    :param sub_center: 子图的质心
    :return: 对齐后的子图点集
    """
    # 计算相对旋转四元数
    main_rotation = R.from_quat(main_quat)  # 主图旋转
    sub_rotation = R.from_quat(sub_quat)  # 子图旋转

    relative_rotation = main_rotation * sub_rotation.inv()  # 相对旋转

    # 应用相对旋转到子图点集
    rotated_sub_coordinates = relative_rotation.apply(sub_coordinates)

    # 平移点集，使子图质心与主图质心对齐
    aligned_coordinates = rotated_sub_coordinates + (main_center - sub_center)

    return aligned_coordinates


def build_sub_graph(elements,main_quat, sub_quat):
    """
    根据IFC文件构建构件组无向图，其中每个构件是一个节点，节点属性为构件对象，构件之间的空间关系是无向边（质心矢量模长）。

    :param ifc_file: IFC文件对象
    :return: 无向图
    """
    G = nx.Graph()  # 创建无向图
    element_meanpoints = {}

    for element in elements:

        meanpoint = get_element_meanpoint(element)  # 假设已定义的函数获取质心

        # 计算相对旋转四元数
        main_rotation = R.from_quat(main_quat)  # 主图旋转
        sub_rotation = R.from_quat(sub_quat)  # 子图旋转

        relative_rotation = sub_rotation*main_rotation .inv()  # 相对旋转

        # 应用相对旋转到子图点集
        rotated_sub_coordinates = relative_rotation.apply(meanpoint)

        if meanpoint is not None:
            element_meanpoints[element.id()] = {
                'element': element,  # 保存完整的构件对象作为属性
                'centroid': rotated_sub_coordinates  # 保存质心
            }

    # 添加节点到图中
    for element_id, data in element_meanpoints.items():
        G.add_node(element_id, **data)  # 将构件和质心作为属性添加到节点中

    # 添加无向边，表示构件之间的空间关系
    for i, j in combinations(element_meanpoints.keys(), 2):  # 获取所有构件对
        centroid_i = np.array(element_meanpoints[i]['centroid'])
        centroid_j = np.array(element_meanpoints[j]['centroid'])

        # 计算构件i和j之间的空间关系（质心的矢量和距离）
        vector = centroid_j - centroid_i  # 从i到j的矢量
        distance = np.linalg.norm(vector)  # 矢量的模长

        # 添加无向边
        G.add_edge(i, j, vector=vector.tolist(), distance=distance)  # 将矢量和距离作为边的属性

    return G


ifc_file_path0 = r'E:\原电脑d盘新\BIM模型\BIM\中建壹品汉芯公馆\新建文件夹\A0.ifc'
ifc_file_path1 = r'E:\原电脑d盘新\BIM模型\BIM\中建壹品汉芯公馆\新建文件夹\A3.ifc'

# 示例：加载IFC文件并构建空间关系图
ifc_file0 = ifcopenshell.open(ifc_file_path0 )  # 加载IFC文件
ifc_file1 = ifcopenshell.open(ifc_file_path1)  # 加载IFC文件

spatial_graph0,distance0 = build_spatial_graph_from_ifc0(ifc_file0)
spatial_graph1 = build_spatial_graph_from_ifc1(ifc_file1,distance0)

# #绘制空间关系图
# plot_spatial_graph(spatial_graph0)
# plot_spatial_graph(spatial_graph1)

# 调用图匹配函数
matches = graph_matching(spatial_graph1, spatial_graph0)

# # 输出匹配结果
# print("匹配的子图:")
# for i, match in enumerate(matches):
#     print(f"匹配 {i + 1}: {match}")
#
# print(distance0)

# 遍历所有匹配的子图
for i, match in enumerate(matches):
    print(f"匹配 {i + 1}:")

    main_elements = []
    sub_elements = []

    all_coordinates1 = []
    all_coordinates2 = []

    # 遍历主图中的每个匹配节点
    for main_node, sub_node in match.items():
        # 获取主图中的构件（通过节点ID）
        main_element = spatial_graph1.nodes[main_node]['element']  # 获取主图节点的构件对象
        main_elements.append(main_element)  # 使用 append() 而非 +=

        # 获取子图中的构件（通过节点ID）
        sub_element = spatial_graph0.nodes[sub_node]['element']  # 获取子图节点的构件对象
        sub_elements.append(sub_element)  # 使用 append() 而非 +=

        # 打印构件的相关信息
        print(f"主图节点 {main_node} 匹配子图节点 {sub_node}")
        # print(f"主图构件 ID: {main_element.id()}, 子图构件 ID: {sub_element.id()}")
        # print(f"主图构件: {main_element}, 子图构件: {sub_element}")

    # 处理主图中的所有构件
    for element in main_elements:
        coordinates, _ = get_pointset(ifc_file_path0, element)  # 获取构件点集
        all_coordinates1 += coordinates  # 添加到总坐标列表中

    obb_center1, obb_extent1, quat1 = compute_obb2(all_coordinates1)  # 计算 OBB（包围盒）信息
    # plot_obb(all_coordinates1, obb_center1, obb_extent1, quat1)
    print(f"主图构件 OBB: Center={obb_center1}, Extent={obb_extent1}, Quaternion={quat1}")

    # 处理子图中的所有构件
    for element in sub_elements:
        coordinates, _ = get_pointset(ifc_file_path1, element)  # 获取构件点集
        all_coordinates2 += coordinates  # 添加到总坐标列表中



    obb_center2, obb_extent2, quat2 = compute_obb2(all_coordinates2)  # 计算 OBB（包围盒）信息
    # plot_obb(all_coordinates2, obb_center2, obb_extent2, quat2)
    print(f"子图构件 OBB: Center={obb_center2}, Extent={obb_extent2}, Quaternion={quat2}")

    # # 进行旋转配准
    # aligned_coordinates = align_subgraph(quat1, quat2, sub_coordinates, obb_center1, obb_center2)

    # # 输出对齐后的点集
    # print("对齐后的子图点集:")
    # print(aligned_coordinates)

    main_graph= build_sub_graph(main_elements,quat1, quat2)
    plot_spatial_graph(main_graph)









