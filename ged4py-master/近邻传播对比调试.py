import numpy as np
import ifcopenshell
import ifcopenshell.api
from geomdl import BSpline
from geomdl.visualization import VisMPL
import ifcopenshell
import time
import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from ged4py.algorithm import graph_edit_dist
from 近邻传播 import find_seed_node
from 近邻传播 import propagate_neighbors
from itertools import combinations
import multiprocessing

start = time.perf_counter()

matplotlib.rcParams['font.family'] = 'SimHei'  # 使用黑体
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号



import networkx as nx

class Face:
    def __init__(self, vertices):
        """
        初始化一个面对象

        :param control_points: 控制点列表，每个控制点是一个元组 (x, y, z, w)，w 是权重参数
        """
        self.control_points = vertices
        self.metric_tensor = self.compute_metric_tensor()
        self.barycentric_matrix = self.compute_barycentric_matrix()
        self.inertia_tensor = self.compute_inertia_tensor()
        self.mean_point = self.compute_mean_points()

    def __repr__(self):
        return (f"Face(control_points={self.control_points}, "
                f"metric_tensor={self.metric_tensor}, "
                f"barycentric_matrix={self.barycentric_matrix}, "
                f"inertia_tensor={self.inertia_tensor}),"
                f"mean_point={self.inertia_tensor}),"
                )

    def conpute_vector(self,Meanpoint, P1):
        #计算中心点到控制点的向量
        vector = np.array(Meanpoint) - np.array(P1)
        # norm = np.linalg.norm(vector)
        # if norm == 0:
        return vector

    def unit_vector(self,P1, P2):
        """
        计算从 P1 到 P2 的单位向量
        """
        vector = np.array(P2) - np.array(P1)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def compute_metric_tensor(self):
        tensor = []
        for i in range(len(self.control_points)):
            V = []
            if i != len(self.control_points) - 1:
                V = self.unit_vector(self.control_points[i - 1], self.control_points[i])
                tensor.append(V)
            else:
                V = self.unit_vector(self.control_points[i], self.control_points[0])
                tensor.append(V)

        # 将 metric_tensor 转换为 NumPy 数组
        tensor = np.array(tensor)

        # 计算矩阵乘积
        metric_tensor1 = np.dot(tensor.T, tensor)

        metric_tensor = np.round(metric_tensor1, 5)


        return metric_tensor

    def compute_barycentric_matrix(self):
        mean_points = np.mean(self.control_points, axis=0)
        barycentric_matrix = np.zeros((4, 4))
        for i in range(len(self.control_points)):
            vertex = self.control_points[i]
            x, y, z, w = vertex[0]-mean_points[0], vertex[1]-mean_points[1], vertex[2]-mean_points[2], vertex[3]
            barycentric_matrix += np.array([
                [(y ** 2 + z ** 2 + w ** 2), -x * y, -x * z, -x * w],
                [-x * y, (x ** 2 + z ** 2 + w ** 2), -y * z, -y * w],
                [-x * z, -y * z, (x ** 2 + y ** 2 + w ** 2), -z * w],
                [-x * w, -y * w, -z * w, (x ** 2 + y ** 2 + z ** 2)]
            ])

        return barycentric_matrix

    def compute_mean_points(self):
        mean_points = np.mean(self.control_points, axis=0)

        return mean_points

    def compute_inertia_tensor(self):
        barycentric_matrix = self.compute_barycentric_matrix()
        eigenvalues = np.linalg.eigvals(barycentric_matrix)

        # 对特征值从小到大排序
        sorted_eigenvalues = sorted(round(val,4) for val in eigenvalues)

        return sorted_eigenvalues


def common_edge(face1, face2):
    """
    判断两个面是否有公共边
    """
    # 获取交集中的点（仅取前三维）
    set1 = set(map(lambda p: tuple(p[:3]), face1.control_points))  # 只取前三维
    set2 = set(map(lambda p: tuple(p[:3]), face2.control_points))  # 只取前三维
    common_points = list(set1.intersection(set2))

    # 如果交集中小于两个点，返回空值
    if len(common_points) < 2:
        return False
    else:
        mean_points1 = np.mean([point[:3] for point in face1.control_points], axis=0)
        mean_points2 = np.mean([point[:3] for point in face2.control_points], axis=0)
        max_distance = 0
        farthest_points = None
        # 计算交集中所有点对之间的距离（使用前三维）
        for p1, p2 in combinations(common_points, 2):
            # 计算欧氏距离，使用前三个坐标
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            if distance > max_distance:
                max_distance = distance
                farthest_points = (p1, p2)

    return farthest_points

# 计算点到直线的垂直距离
def point_to_line_distance(point, line_start, line_end):
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len  # 单位方向向量
    projection_length = np.dot(point_vec, line_unitvec)  # 投影长度
    projection_point = np.array(line_start) + projection_length * line_unitvec  # 投影点

    vector = np.array(point) - np.array(projection_point)

    return vector

# 计算两个向量之间的夹角
def angle_between_vectors(vec1, vec2):
    # 计算向量的范数
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # 检查向量是否为零向量
    if norm_vec1 == 0 or norm_vec2 == 0:
        dot_product = 0
    else:

        # 归一化向量
        unit_vec1 = vec1 / norm_vec1
        unit_vec2 = vec2 / norm_vec2
        dot_product = np.dot(unit_vec1, unit_vec2)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # 确保在有效范围内
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def create_graph(faces):
    G = nx.Graph()
    for i, face1 in enumerate(faces):
        G.add_node(i, face=face1,label1=face1.metric_tensor,label2=face1.inertia_tensor)
        for j, face2 in enumerate(faces[i + 1:], i + 1):
            farthest_points = common_edge(face1, face2)
            mean_points1 = np.mean([point[:3] for point in face1.control_points], axis=0)
            mean_points2 = np.mean([point[:3] for point in face2.control_points], axis=0)
            if farthest_points:
                vec1 = point_to_line_distance(mean_points1, farthest_points[0],farthest_points[1])
                vec2 = point_to_line_distance(mean_points2, farthest_points[0], farthest_points[1])
                angle = angle_between_vectors(vec1, vec2)
                G.add_edge(i, j, angle=angle)

    return G

class Solid:
    def __init__(self, faces=None):
        """
        初始化一个体对象

        :param faces: 面对象列表
        """
        if faces is None:
            faces = []
        self.faces = faces

    def __repr__(self):
        return f"Solid(faces={self.faces})"

    def add_face(self, face):
        """
        添加一个面到体对象中

        :param face: 面对象
        """
        self.faces.append(face)


class Node:
    def __init__(self):
        self.properties = {}
        self.labels = set()
        self._primarylabel_ = None
        self._primarykey_ = None

    def __setitem__(self, key, value):
        self.properties[key] = value

    def __getitem__(self, key):
        return self.properties[key]

    def add_label(self, label):
        self.labels.add(label)

    def get_labels(self):
        return list(self.labels)

    def set_primary_label(self, label):
        self._primarylabel_ = label

    def set_primary_key(self, key):
        self._primarykey_ = key

    def __repr__(self):
        return f"Node(labels={self.labels}, properties={self.properties})"


def extract_surface_data(surface):
            # 获取控制点
    control_points = surface.ControlPointsList
    control_points = [point.Coordinates for point in control_points]

            # 获取节点向量
    knot_vector_u = surface.KnotVectors[0]
    knot_vector_v = surface.KnotVectors[1]

            # 获取参数化范围
    u_min, u_max = surface.UParameterRange
    v_min, v_max = surface.VParameterRange

    return control_points, knot_vector_u, knot_vector_v, u_min, u_max, v_min, v_max


def calculate_metric_tensor(control_points, knot_vector_u, knot_vector_v, u, v, u_min, u_max, v_min, v_max):
    # 创建 B 样条曲面对象
    surf = BSpline.Surface()

    # 设置 B 样条曲面的阶数 (U 方向 3，V 方向 1)
    surf.degree_u = 3
    surf.degree_v = 1

    # 设置控制点，必须展平处理
    surf.set_ctrlpts(control_points, 4, 2)

    # 设置节点向量，必须满足 n + p + 1 的要求
    surf.knotvector_u = knot_vector_u
    surf.knotvector_v = knot_vector_v

    # 归一化
    u_scaled = (u - u_min) / (u_max - u_min)
    v_scaled = (v - v_min) / (v_max - v_min)

    # 计算曲面在该点处的导数
    derivatives = surf.derivatives(u=u_scaled, v=v_scaled, order=1)

    # 提取导数信息
    du = derivatives[1][0]  # U 方向的一阶导数
    dv = derivatives[0][1]  # V 方向的一阶导数

    # 将导数矢量转换为矩阵形式
    tensor = np.array([du, dv])

    # 计算度量张量：导数矢量的内积
    metric_tensor = np.dot(tensor, tensor.T)

    return metric_tensor


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
                    for face in  item.Outer:
                        found_surfaces.append(face)

    return found_surfaces

def get_weight():
    weight = 1
    return weight

def get_face_data(surface):
    point_set = []

    if hasattr(surface, 'FaceSurface'):
        if hasattr(surface.FaceSurface, 'ControlPointsList'):
            for points in surface.FaceSurface.ControlPointsList:
                j = surface.FaceSurface.ControlPointsList.index(points)
                for point in points:
                    k = points.index(point)
                    if hasattr(surface.FaceSurface, 'WeightsData'):
                        w = surface.FaceSurface.WeightsData[j][k]
                    else:
                        w = get_weight()
                    point_4d = np.append(point, w)
                    point_set.append(point_4d)
        else:
            for bound in surface.Bounds:
                for edge in bound.Bound.EdgeList:
                    w = get_weight()
                    k = edge.EdgeEnd.VertexGeometry.Coordinates
                    point_4d = np.append(edge.EdgeEnd.VertexGeometry.Coordinates, w)
                    point_set.append(point_4d)
                    w = get_weight()
                    point_4d = np.append(edge.EdgeStart.VertexGeometry.Coordinates, w)
                    point_set.append(point_4d)

    elif hasattr(surface, 'Bounds'):
        for bound in surface.Bounds:
            for point in bound.Bound.Polygon:
                w = get_weight()
                point_4d = np.append(point, w)
                point_set.append(point_4d)


    face = Face(point_set)

    return face









def generate_graph(element):


    solid_list = []



    solid = Solid()

    surfaces = find_bspline_surfaces_from_representations(element.Representation.Representations)
    for faces in surfaces:
        for surface in faces:
            face = get_face_data(surface)

            solid.add_face(face)

    solid_list.append(solid)

    mean_point_list = []

    for solid1 in solid_list:
        # 创建图
        G = create_graph(solid1.faces)

        for face in solid1.faces:
            mean_point_list.append(face.mean_point)

        # 假设 mean_point_list 是一个包含多个点坐标的列表
        # 先过滤掉不符合条件的元素
        mean_point_list = [point for point in mean_point_list if
                           isinstance(point, (list, np.ndarray)) and len(point) == 4]

        # 确保过滤后列表非空
        if mean_point_list:
            mean_point = np.mean(mean_point_list, axis=0)
        else:
            mean_point = (0,0,0,1)
            print("mean_point_list 是空的，无法计算平均值")





        # # # 绘制图
        # pos = nx.spring_layout(G)  # 布局
        # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        # plt.show()
        return G,mean_point



        # # 提取面数据
        # control_points, knot_vector_u, knot_vector_v, u_min, u_max, v_min, v_max = extract_surface_data(surface)
        #
        # # 示例参数值
        # u = 0.5
        # v = 2.5
        #
        # # 计算度量张量
        # metric_tensor = calculate_metric_tensor(control_points, knot_vector_u, knot_vector_v, u, v, u_min, u_max, v_min, v_max)
        #
        # print("Metric Tensor:")
        # print(metric_tensor)


def compute_num_faces(element):

    surfaces = find_bspline_surfaces_from_representations(element.Representation.Representations)
    i = 0
    for faces in surfaces:
        i +=len(faces)
    return i


def get_axis2_placement(ifc_file, element):
    """
    获取指定元素的 IfcAxis2Placement3D 实例。

    :param ifc_file: 已打开的 IFC 文件。
    :param element_id: 目标元素的 ID。
    :return: IfcAxis2Placement3D 实例，如果未找到则返回 None。
    """

    # 获取元素的 ObjectPlacement
    object_placement = element.ObjectPlacement
    if object_placement.is_a("IfcLocalPlacement"):
        # 获取相对位置
        placement = object_placement.RelativePlacement if object_placement.RelativePlacement else None

        if placement and placement.is_a("IfcAxis2Placement3D"):
            return placement  # 直接返回 IfcAxis2Placement3D 实例

        # 递归获取相对位置的 IfcAxis2Placement3D
        while placement and placement.is_a("IfcLocalPlacement"):
            placement = placement.RelativePlacement

            if placement and placement.is_a("IfcAxis2Placement3D"):
                return placement

    raise ValueError("未能找到有效的 IfcAxis2Placement3D 实例。")



# 对比元素的处理函数
def process_element(element1, element2, ifc_file):
    # 计算 element1 和 element2 的面数
    num_face1 = compute_num_faces(element1)
    num_face2 = compute_num_faces(element2)

    # 如果面数不相等，跳过该对元素
    if num_face1 != num_face2:
        return None  # 返回 None，表示此对元素不需要进一步处理

    # 生成 element1 和 element2 的图结构和质心
    G1, body_meanpoint1 = generate_graph(element1)
    G2, body_meanpoint2 = generate_graph(element2)

    # 查找种子节点
    seed1, seed2 = find_seed_node(G1, G2)
    if seed1 is not None and seed2 is not None:
        # 如果找到种子节点，进行邻域传播
        result = propagate_neighbors(G1, G2, seed1, seed2)
        if result:
            # 如果邻域传播成功，计算位移
            site_change = body_meanpoint2 - body_meanpoint1
            tx, ty, tz, tw = body_meanpoint2

            # 创建 4x4 的位姿矩阵（仅包含平移信息）
            pose_matrix = np.array([
                [1, 0, 0, tx / 1000],
                [0, 1, 0, ty / 1000],
                [0, 0, 1, tz / 1000],
                [0, 0, 0, 1]
            ])

            # 移除 element2 的现有几何表示
            ifcopenshell.api.run("geometry.remove_representation", ifc_file, representation=element2.Representation)

            # 修改 element2 的位置（使用位姿矩阵）
            ifcopenshell.api.run("geometry.edit_object_placement", ifc_file, product=element2, matrix=pose_matrix)

            # 为 element2 分配新的几何表示（从 element1 中获取）
            for rep in element1.Representation.Representations:
                ifcopenshell.api.run("geometry.assign_representation", ifc_file, product=element2, representation=rep)

            # 返回 element2，表示该元素需要在后续删除
            return element2

    # 如果没有找到合适的种子节点或者邻域传播失败，则返回 None
    return None

# 并行处理对比操作
def parallel_compare(elements1, ifc_file):
    elements_to_remove = []  # 存放需要删除的元素

    # 创建进程池来并行处理元素对比
    with multiprocessing.Pool() as pool:
        # 为每一对元素准备任务
        tasks = [(element1, element2, ifc_file) for element1 in elements1 for element2 in elements1 if element1 != element2]

        # 使用 starmap 并行执行任务
        results = pool.starmap(process_element, tasks)

        # 收集所有需要删除的元素
        elements_to_remove = [res for res in results if res is not None]

    return elements_to_remove

# 主函数入口
if __name__ == "__main__":
    # 假设你有一个元素列表 elements1 和一个 IFC 文件对象
    elements1 = [...]  # 你的元素列表
    ifc_file = ifcopenshell.open(r"E:\原电脑d盘新\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\标准层2.ifc")  # 打开 IFC 文件

    # 执行并行对比操作，返回需要删除的元素
    elements_to_remove = parallel_compare(elements1, ifc_file)

    # 从元素列表中删除需要删除的元素
    for element in elements_to_remove:
        elements1.remove(element)

    # 将修改后的 IFC 文件保存为新的文件
    output_file_path = r"D:\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\生成的文档\标准层副本.ifc"
    ifc_file.write(output_file_path)



# def compare_graph2(ifc_file_path1):
#
#     elements0 = []
#     # 读取 IFC 文件
#     ifc_file = ifcopenshell.open(ifc_file_path1)
#     elements1= ifc_file.by_type("IfcElement")
#
#
#
#     for element1 in elements1:
#
#         num_face1 = compute_num_faces(element1)
#
#         i = 0
#
#         print("\n")
#
#         print("element1", element1)
#
#         # elements1.remove(element1)
#
#
#         elements_to_remove = []
#
#
#         for element2 in elements1:
#             num_face2 = compute_num_faces(element2)
#
#
#             if num_face1 != num_face2:
#                 continue
#
#             else:
#
#                 G1,body_meanpoint1 = generate_graph(element1)
#                 G2,body_meanpoint2 = generate_graph(element2)
#
#                 seed1, seed2 = find_seed_node(G1, G1)
#                 if seed1 is not None and seed2 is not None:
#                     result = propagate_neighbors(G1, G2, seed1, seed2)
#                     if result:
#
#                         print("element2", element2)
#
#                         site_change = body_meanpoint2 - body_meanpoint1
#
#                         tx, ty, tz,tw = body_meanpoint2
#
#                         # 生成 4x4 的位姿矩阵，只包含平移信息
#                         pose_matrix = np.array([
#                             [1, 0, 0, tx/1000],  # 第一行
#                             [0, 1, 0, ty/1000],  # 第二行
#                             [0, 0, 1, tz/1000],  # 第三行
#                             [0, 0, 0, 1]  # 齐次坐标
#                         ])
#
#
#                         # # 创建 IfcRepresentationMap
#                         # rep_map = ifc_file.create_entity("IFCREPRESENTATIONMAP", MappingOrigin=axis2_placement1,
#                         #                                  MappedRepresentation=element1.Representation)
#                         #
#                         # # 添加新的 mapped_item
#                         # mapped_item = ifc_file.create_entity("IFCMAPPEDITEM", MappingSource=rep_map,
#                         #                                      MappingTarget=axis2_placement2)
#                         #
#                         # # 获取 element2 的几何表达
#                         # shape_representation = element2.Representation.Representations[0]
#                         #
#                         # # 获取 shape_representation 的 ContextOfItems
#                         # context_of_items = shape_representation.ContextOfItems
#
#                         ifcopenshell.api.run("geometry.remove_representation", ifc_file, representation=element2.Representation)
#
#                         ifcopenshell.api.run("geometry.edit_object_placement", ifc_file, product=element2, matrix=pose_matrix)
#                         for rep in element1.Representation.Representations:
#
#                             ifcopenshell.api.run("geometry.assign_representation", ifc_file, product=element2,
#                                                  representation=rep)
#
#
#
#                         #
#                         # # 创建新的 IfcShapeRepresentation 使用 context_of_items
#                         # new_representation = ifc_file.create_entity(
#                         #     "IFCSHAPEREPRESENTATION",
#                         #     ContextOfItems=context_of_items,  # 从 shape_representation 中获取
#                         #     RepresentationIdentifier="Body",
#                         #     RepresentationType="MappedRepresentation",
#                         #     Items=[]
#                         # )
#                         #
#                         # # 替换 element2 的 Representation
#                         # new_product_representation = ifc_file.create_entity(
#                         #     "IFCPRODUCTREPRESENTATION",
#                         #     Representations=[new_representation]
#                         # )
#                         #
#                         # # 将 element2 的 Representation 设置为新的几何表达
#                         # element2.Representation = new_product_representation
#
#
#
#                         # ifcopenshell.api.run("geometry.assign_representation", ifc_file, product=element2,
#                         #                      representation=element1.Representation)
#
#                         # ifcopenshell.api.run("geometry.edit_object_placement", ifc_file, product=element2)
#
#
#                         elements_to_remove.append(element2)
#
#
#                     # else:
#                     #     print("两个图不一致")
#                     #     print("element1", element1)
#                     #     print("element2", element2)
#                 else:
#                     print("未找到合适的种子节点")
#
#         for element in elements_to_remove:
#             elements1.remove(element)
#
#     # 4. 将修改后的内容保存为新的 IFC 文件
#     output_file_path = r"D:\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\生成的文档\标准层副本.ifc"
#     ifc_file.write(output_file_path)
#
#
#
#
# #单构件对比
# def compare_graph1(ifc_file_path1):
#
#     elements0 = []
#     # 读取 IFC 文件
#     ifc_file = ifcopenshell.open(ifc_file_path1)
#     elements1= ifc_file.by_type("IfcElement")
#
#     element= ifc_file.by_id("25YldT_Zz0MxyMdyxsKBQO")
#
#     num_face1 = compute_num_faces(element)
#
#     i = 0
#
#     print("\n")
#
#     print("element1", element)
#
#     elements1.remove(element)
#
#     elements_to_remove = []
#
#     for element2 in elements1:
#         num_face2 = compute_num_faces(element2)
#
#         if num_face1 != num_face2:
#             continue
#
#         else:
#
#             G1, body_meanpoint1 = generate_graph(element)
#             G2, body_meanpoint2 = generate_graph(element2)
#
#             seed1, seed2 = find_seed_node(G1, G1)
#             if seed1 is not None and seed2 is not None:
#                 result = propagate_neighbors(G1, G2, seed1, seed2)
#                 if result:
#                     print("element2", element2)
#                     elements_to_remove.append(element2)
#
#                 # else:
#                 #     print("两个图不一致")
#                 #     print("element1", element1)
#                 #     print("element2", element2)
#             else:
#                 print("未找到合适的种子节点")
#
#     for element in elements_to_remove:
#         elements1.remove(element)
#
#
#
#
# # IFC 文件路径
# ifc_file_path1 = r"E:\原电脑d盘新\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\标准层2.ifc"
# # ifc_file_path2 = r"D:\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\半标准层.ifc"
#
# # ifc_file_path1 = r"D:\BIM模型\rhino\IFCrhino\体\不规则曲面形体\组合圆柱.ifc"
# # ifc_file_path2 = r"D:\BIM模型\rhino\IFCrhino\体\不规则曲面形体\组合圆柱1.ifc"
# # generate_graph(ifc_file_path)
# compare_graph2(ifc_file_path1)



end = time.perf_counter()

# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")
