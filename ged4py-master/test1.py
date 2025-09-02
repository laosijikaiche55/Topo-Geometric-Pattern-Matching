import ifcopenshell
import time
import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from ged4py.algorithm import graph_edit_dist

import networkx as nx

start = time.perf_counter()

matplotlib.rcParams['font.family'] = 'SimHei'  # 使用黑体
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号


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
    set1 = set(map(tuple, face1.control_points))
    set2 = set(map(tuple, face2.control_points))
    return len(set1.intersection(set2)) >= 2

def create_graph(faces):
    G = nx.Graph()
    for i, face1 in enumerate(faces):
        G.add_node(i, face=face1,label1=face1.metric_tensor,label2=face1.inertia_tensor)
        for j, face2 in enumerate(faces[i + 1:], i + 1):
            if common_edge(face1, face2):
                G.add_edge(i, j)
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

def compute_transformation_matrix(surface1, surface2):
    barycentric_matrix1 = surface1.compute_barycentric_matrix()
    barycentric_matrix2 = surface2.compute_barycentric_matrix()

    # 使用SVD分解
    U, _, Vt = np.linalg.svd(np.dot(barycentric_matrix2, np.linalg.inv(barycentric_matrix1)))
    transformation_matrix = np.dot(U, Vt)

    return transformation_matrix

def generate_graph(ifc_file_path):
    ifc_file = ifcopenshell.open(ifc_file_path)
    # 查找Tessellation几何体
    solid_list = []
    for product in ifc_file.by_type("IfcElement"):
        for representation in product.Representation.Representations:
            if representation.RepresentationType == "Tessellation":
                for item in representation.Items:
                    if item.is_a("IfcPolygonalFaceSet"):
                        face_set = item
                        point_list = face_set.Coordinates.CoordList

                        solid = Solid()

                        # 打印控制点
                        control_points = [(p[0], p[1], p[2]) for p in point_list]
                        print("Control Points:", control_points)


                        # 打印每个面的顶点索引
                        for face in face_set.Faces:
                            if face.is_a("IfcIndexedPolygonalFace"):
                                vertex_indices = face.CoordIndex
                                control_points_4d = []


                                default_weight = 1
                                for i in vertex_indices:
                                     control_points_4d .append([control_points[i-1][0],control_points[i-1][1],control_points[i-1][2],default_weight])

                                # print("Face Vertex Indices:", control_points_4d)

                                face = Face(control_points_4d)
                                solid.add_face(face)

                        solid_list.append(solid)
                        # print(solid_list)

    for solid1 in solid_list:
        for face0 in solid1.faces:
            for face in solid1.faces:
                transformation_matrix = compute_transformation_matrix(face0, face)
                # print(transformation_matrix)



    for solid1 in solid_list:
        # 创建图
        G = create_graph(solid1.faces)

        # # 绘制图
        # pos = nx.spring_layout(G)  # 布局
        # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        # plt.show()
        return G

#元素是否接近
def compare_nested_with_isclose(tuple1, tuple2, tolerance):
    if isinstance(tuple1, tuple) and isinstance(tuple2, tuple):
        if len(tuple1) != len(tuple2):
            return False
        for a, b in zip(tuple1, tuple2):
            if not compare_nested_with_isclose(a, b, tolerance):
                return False
        return True
    else:
        # 只对数值进行 isclose 比较，跳过字符串
        if isinstance(tuple1, (int, float)) and isinstance(tuple2, (int, float)):
            return np.isclose(tuple1, tuple2, atol=tolerance)
        return tuple1 == tuple2  # 非数值类型，直接比较相等性


# 基本属性比较
def compare_basic_properties(G1, G2):
    properties = {
        '节点数量': (G1.number_of_nodes(), G2.number_of_nodes()),
        '边的数量': (G1.number_of_edges(), G2.number_of_edges()),
        '度分布': (sorted([d for n, d in G1.degree()]), sorted([d for n, d in G2.degree()]))
    }
    return properties

def compare_graph_node_attributes(G1, G2):
    if G1.number_of_nodes() != G2.number_of_nodes():
        return False, "两个图的节点数量不同"

    def to_hashable(value):
        if isinstance(value, np.ndarray):
            return tuple(to_hashable(v) for v in value)
        elif isinstance(value, (list, tuple)):
            return tuple(to_hashable(v) for v in value)
        return value

    G1_nodes_attributes = set(
        (to_hashable(attr['label1']), to_hashable(attr['label2'])) for attr in G1.nodes.values()
    )
    G2_nodes_attributes = set(
        (to_hashable(attr['label1']), to_hashable(attr['label2'])) for attr in G2.nodes.values()
    )

    print(G1_nodes_attributes)
    print(G2_nodes_attributes)

    if G1_nodes_attributes != G2_nodes_attributes:
        print("节点属性不匹配")
        return False, "节点属性不匹配"
    else:
        print("节点属性匹配")

    return True, "两个图的节点属性相同"




# 节点标签比较
def compare_node_labels(G1, G2):
    labels_G1 = nx.get_node_attributes(G1, 'label')
    labels_G2 = nx.get_node_attributes(G2, 'label')
    return labels_G1 == labels_G2

# 节点度比较；
def compare_node_degrees(G1, G2):
    degrees_G1 = sorted(G1.degree, key=lambda x: x[0])
    degrees_G2 = sorted(G2.degree, key=lambda x: x[0])
    return degrees_G1 == degrees_G2

# 图同构检查
def check_isomorphism(G1, G2):
    return nx.is_isomorphic(G1, G2)


def get_node_identifier(node_attrs, keys_to_compare):
    processed_attrs = {}
    for key in keys_to_compare:
        if key in node_attrs:
            value = node_attrs[key]
            if isinstance(value, np.ndarray):
                # 将矩阵展平并转换为元组
                processed_attrs[key] = tuple(value.flatten())
            elif isinstance(value, list):
                # 将列表转换为元组
                processed_attrs[key] = tuple(value)
            else:
                processed_attrs[key] = value

    return tuple(sorted(processed_attrs.items()))


def compare_graph(ifc_file_path1,ifc_file_path2):
    G1 = generate_graph(ifc_file_path1)
    G2 = generate_graph(ifc_file_path2)

    print(graph_edit_dist.compare(G1, G2, True))

    # keys_to_compare = ["label1", "label2"]
    #
    # # 1. Compare structure (number of nodes and edges)
    # if G1.number_of_nodes() != G2.number_of_nodes():
    #     return False, "Number of nodes differ"
    #
    # print(G1.number_of_edges())
    # print(G2.number_of_edges())
    # if G1.number_of_edges() != G2.number_of_edges():
    #     return False, "Number of edges differ"
    #
    # # 2. Compare node attributes
    # graph1_nodes = {get_node_identifier(attrs, keys_to_compare): node for node, attrs in G1.nodes(data=True)}
    # graph2_nodes = {get_node_identifier(attrs, keys_to_compare): node for node, attrs in G2.nodes(data=True)}
    #
    # # if graph1_nodes.keys() != graph2_nodes.keys():
    # #     m = graph1_nodes.keys()
    # #     n = graph2_nodes.keys()
    # #     return False, "Nodes with corresponding attributes differ"
    #
    # # 创建副本以避免修改原始列表
    # temp_list2 = list(graph2_nodes.keys())



    # # 遍历第一个列表，尝试在第二个列表中移除每个元素
    # for item1 in graph1_nodes.keys():
    #     for item2 in temp_list2:
    #         tolerance = 0.01
    #         result = compare_nested_with_isclose(item1, item2, tolerance)
    #         if result:
    #             temp_list2.remove(item2)
    #
    #
    #
    # # 对比边和连接关系
    # for edge in G1.edges(data=True):
    #     u_attrs = get_node_identifier(G1.nodes[edge[0]], keys_to_compare)
    #     v_attrs = get_node_identifier(G1.nodes[edge[1]], keys_to_compare)
    #     if (u_attrs, v_attrs) not in [(get_node_identifier(G2.nodes[u], keys_to_compare), get_node_identifier(G2.nodes[v], keys_to_compare)) for u, v in G2.edges()]:
    #         return False, f"Edge between nodes with attributes {u_attrs} and {v_attrs} missing in graph2"
    #
    # return True, "Graphs are equivalent"






# 加载IFC文件
ifc_file_path1 = r"C:\Users\visac\Desktop\毕业\对比实验\IFC文件\球54段100移动.ifc"
ifc_file_path2 = r"C:\Users\visac\Desktop\毕业\对比实验\IFC文件\球54段100.ifc"
print(compare_graph(ifc_file_path1,ifc_file_path2))







end = time.perf_counter()

# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")
