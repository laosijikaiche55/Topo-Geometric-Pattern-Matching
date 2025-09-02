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
from itertools import combinations

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

    def __repr__(self):
        return (f"Face(control_points={self.control_points}, "
                f"metric_tensor={self.metric_tensor}, "
                f"barycentric_matrix={self.barycentric_matrix}, "
                f"inertia_tensor={self.inertia_tensor})")

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

        # 进行矩阵自乘
        metric_tensor1 = np.dot(tensor, tensor.T)

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

    for solid1 in solid_list:
        # 创建图
        G = create_graph(solid1.faces)

        # # # 绘制图
        # pos = nx.spring_layout(G)  # 布局
        # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        # plt.show()
        return G



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
        for surface in faces:
            i +=1
    return i



def compare_graph(ifc_file_path1,ifc_file_path2):

    elements0 = []
    # 读取 IFC 文件
    ifc_file = ifcopenshell.open(ifc_file_path1)
    elements1= ifc_file.by_type("IfcElement")

    # 读取 IFC 文件
    ifc_file = ifcopenshell.open(ifc_file_path2)
    elements2= ifc_file.by_type("IfcElement")

    for element1 in elements1:

        num_face1 = compute_num_faces(element1)

        i = 0


        for element2 in elements2:
            num_face2 = compute_num_faces(element2)

            if num_face1 != num_face2:
                continue

            else:

                G1 = generate_graph(element1)
                G2 = generate_graph(element2)

                distance = graph_edit_dist.compare(G1, G2, False)

                if distance == 0:
                    i+=1
                    elements0.append(element2)

                    print(element2)

                    if i%10 == 0:
                        print(i)






# IFC 文件路径
ifc_file_path1 = r"D:\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\窗户1.ifc"
ifc_file_path2 = r"D:\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\窗户1.ifc"
# generate_graph(ifc_file_path)
compare_graph(ifc_file_path1,ifc_file_path2)




end = time.perf_counter()

# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")
