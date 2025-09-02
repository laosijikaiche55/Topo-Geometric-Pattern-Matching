import ifcopenshell
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations
from scipy.spatial import KDTree

import numpy as np
import ifcopenshell
from ifcopenshell.util.placement import get_local_placement


def get_element_meanpoint(element):
    """获取构件在世界坐标系下的质心（包含坐标系转换）"""
    # 获取构件的几何表达项
    if not hasattr(element, 'Representation') or not element.Representation:
        return None

    # 获取构件的全局变换矩阵
    placement_matrix = get_global_transform(element)

    # 提取原始几何数据（局部坐标系）
    local_centroid = get_element_meanpoint1(element)
    if local_centroid is None:
        return None

    # 转换为世界坐标系
    global_centroid = apply_transform(placement_matrix, local_centroid)
    return global_centroid


def get_global_transform(element):
    """递归计算从构件局部坐标系到世界坐标系的变换矩阵"""
    matrix = np.eye(4)  # 初始化单位矩阵

    # 沿对象放置链向上遍历
    current_placement = element.ObjectPlacement
    while current_placement:
        local_placement = get_local_placement(current_placement)
        matrix = np.dot(local_placement, matrix)  # 矩阵级联
        current_placement = current_placement.PlacementRelTo  # 处理相对放置

    return matrix


def calculate_local_centroid(element):
    """计算构件在局部坐标系中的原始质心"""
    points = []

    # 遍历几何表达项
    for rep in element.Representation.Representations:
        if rep.RepresentationType == "SweptSolid":
            # 处理拉伸体类型
            for item in rep.Items:
                if item.is_a("IfcExtrudedAreaSolid"):
                    profile = item.SweptArea
                    if profile.is_a("IfcArbitraryClosedProfileDef"):
                        points.extend([v.Coordinates for v in profile.OuterCurve.Points])
        elif rep.RepresentationType == "MappedRepresentation":
            # 处理映射几何
            for item in rep.Items:
                if item.is_a("IfcMappedItem"):
                    points.extend(process_mapped_item(item))

    if not points:
        return None

    return np.mean(points, axis=0)


def apply_transform(matrix, point):
    """将4x4变换矩阵应用到3D点上"""
    # 转换为齐次坐标
    homogeneous_point = np.append(point, 1.0)
    transformed = np.dot(matrix, homogeneous_point)
    return transformed[:3]  # 返回三维坐标

# 使用TkAgg后端，解决PyCharm显示问题
matplotlib.use('TkAgg')

# 从IFC文件中提取构件的几何信息
def get_face_data(surface):
    point_set = []

    if hasattr(surface, 'FaceSurface'):
        if hasattr(surface.FaceSurface, 'ControlPointsList'):
            for points in surface.FaceSurface.ControlPointsList:
                for point in points:
                    point_set.append(point)

        else:
            for bound in surface.Bounds:
                for edge in bound.Bound.EdgeList:
                    k = edge.EdgeEnd.VertexGeometry.Coordinates
                    for point in k:
                        point_set.append(point)

    elif hasattr(surface, 'Bounds'):
        for bound in surface.Bounds:
            for point in bound.Bound.Polygon:
                point_set.append(point)
    return point_set


def find_bspline_surfaces_from_representations(representations):
    """从 Representations 列表递归查找 IfcBSplineSurfaceWithKnots 实体"""
    found_surfaces = []

    # 对 Representations 列表中的每个元素进行递归查找
    for representation in representations:
        if representation.RepresentationIdentifier == "Body":
            for item in representation.Items:
                if item.is_a("IfcMappedItem"):
                    for item1 in item.MappingSource.MappedRepresentation.Items:
                        for face in item1.Outer:
                            found_surfaces.append(face)
                else:
                    for face in item.Outer:
                        found_surfaces.append(face)

    return found_surfaces

def get_element_meanpoint1(element):

    solid_list = []

    surfaces = find_bspline_surfaces_from_representations(element.Representation.Representations)
    for faces in surfaces:
        for surface in faces:
            facepoint = get_face_data(surface)
            solid_list.extend(facepoint)

    meanpoint = np.mean(solid_list, axis=0)
    meanpoint_1d = meanpoint.flatten()

    return meanpoint_1d


def extract_type_ids(element):

        # 获取构件的所有属性集
        property_sets = element.IsDefinedBy

        for prop_set in property_sets:
            if prop_set.is_a("IfcRelDefinesByProperties"):
                # 提取属性集中的具体属性
                properties = prop_set.RelatingPropertyDefinition.HasProperties

                for prop in properties:
                    # 筛选目标属性：Name为'Id类(Type Id)'的IfcPropertySingleValue
                    if prop.Name == 'Id类型(Type Id)' and prop.is_a("IfcPropertySingleValue"):
                        # 提取属性值
                        type_id = prop.NominalValue.wrappedValue
                        # print(f"构件 GlobalId: {element.GlobalId}")
                        # print(f"Type ID: {type_id}\n")

                        return type_id

                        break




# 构建有向图，表示构件之间的空间关系
def build_spatial_graph_from_ifc0(ifc_file):
    """
    根据IFC文件构建构件组无向图，其中每个构件是一个节点，节点属性为构件对象，构件之间的空间关系是无向边（质心矢量模长）。

    :param ifc_file: IFC文件对象
    :return: 无向图
    """
    G = nx.Graph()  # 创建无向图
    element_meanpoints = {}
    distance0 = 0
    type_ids = []

    # 提取构件及其质心
    elements = ifc_file.by_type("IfcElement")
    for element in elements:

        a = extract_type_ids(element)

        type_ids.append(a)


        meanpoint = get_element_meanpoint(element)  # 假设已定义的函数获取质心
        if meanpoint is not None:
            element_meanpoints[element.id()] = {
                'element': element,  # 保存完整的构件对象作为属性
                'centroid': meanpoint,  # 保存质心
                'type_id': a
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

        if distance > distance0:
            distance0 = distance+10

        # 添加无向边
        G.add_edge(i, j, vector=vector.tolist(), distance=distance)  # 将矢量和距离作为边的属性

    return G,distance0,type_ids


def build_spatial_graph_from_ifc1(ifc_file, distance0,type_ids):
    """
    根据IFC文件构建构件组无向图，其中每个构件是一个节点，节点属性为构件对象，
    构件之间的空间关系是无向边（质心矢量模长）。

    :param ifc_file: IFC文件对象
    :param distance0: 距离阈值，控制构件间的空间关系
    :return: 无向图
    """
    G = nx.Graph()  # 创建无向图
    element_meanpoints = {}

    # 提取构件及其质心
    elements = ifc_file.by_type("IfcElement")

    for element in elements:

        a = extract_type_ids(element)

        if a in type_ids:

            meanpoint = get_element_meanpoint(element)  # 假设已定义的函数获取质心

            if meanpoint is not None:
                element_meanpoints[element.id()] = {
                    'element': element,  # 保存完整的构件对象作为属性
                    'centroid': meanpoint,  # 保存质心（三维坐标）
                    'type_id':a
                }

        else:
            continue

    # 获取所有质心的坐标（形状为(n, 3)）
    centroids = np.array([data['centroid'] for data in element_meanpoints.values()])
    elementid_list = np.array([k for k in element_meanpoints.keys()])
    # 构建KDTree进行最近邻查询
    kdtree = KDTree(centroids)

    # 添加节点到图中
    for element_id, data in element_meanpoints.items():
        G.add_node(element_id, **data)  # 将构件和质心作为属性添加到节点中

    # 添加无向边，表示构件之间的空间关系
    for i, data in element_meanpoints.items():
        centroid_i = np.array(data['centroid'])

        # 查找质心i周围的构件（以distance0为阈值）
        nearby_points = kdtree.query_ball_point(centroid_i, distance0)

        for j in nearby_points:
            if i != elementid_list[j]:  # 确保不和自己添加边
                centroid_j = centroids[j]
                # 计算质心i和质心j之间的空间关系（质心的矢量和距离）
                vector = centroid_i - centroid_j  # 从i到j的矢量
                distance = np.linalg.norm(vector)  # 矢量的模长

                # 添加无向边
                G.add_edge(i, elementid_list[j], vector=vector.tolist(), distance=distance)

    return G






# 绘制空间关系图
def plot_spatial_graph(G):
    """
    绘制BIM构件组的空间关系图。

    :param G: 有向图，包含构件的空间关系
    """
    pos = nx.spring_layout(G, seed=42)  # 布局，spring_layout是一种力导向布局

    # 绘制节点和边
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    # 添加每条边的空间矢量标签
    edge_labels = nx.get_edge_attributes(G, 'vector')
    formatted_edge_labels = {k: f'{v}' for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels, font_size=8)

    # # 显示图形
    # plt.title("BIM Components Spatial Relationships")
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()


# # 示例：加载IFC文件并构建空间关系图
# ifc_file = ifcopenshell.open(r'E:\原电脑d盘新\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\三门.ifc')  # 加载IFC文件
# spatial_graph1 = build_spatial_graph_from_ifc1(ifc_file,5000)
# spatial_graph0,k = build_spatial_graph_from_ifc0(ifc_file)

# #绘制空间关系图
# plot_spatial_graph(spatial_graph1)
# plot_spatial_graph(spatial_graph0)
# r'E:\原电脑d盘新\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\三门.ifc'