import networkx as nx
from networkx.algorithms import isomorphism
from 构件组生成 import build_spatial_graph_from_ifc0
from 构件组生成 import build_spatial_graph_from_ifc1
from 构件组生成 import plot_spatial_graph
import ifcopenshell
import numpy as np
import psutil
import os
import time
from tremish包围 import compute_obb2

start = time.perf_counter()


# ================= 新增节点匹配函数 =================
def node_match(node_attr1, node_attr2):
    """
    定义节点属性匹配规则：
    1. Type ID必须相同
    2. 其他属性（如几何位置）可根据需求扩展
    """

    node_attr1
    # 检查Type ID是否一致
    type_id_match = node_attr1['type_id'] == node_attr2['type_id']

    # 可选：扩展其他属性匹配（例如质心坐标）
    # centroid_match = np.allclose(node_attr1['centroid'], node_attr2['centroid'], atol=1e-3)
    # return type_id_match and centroid_match

    return type_id_match


def edge_match(attr1, attr2):
    """
    定义边属性的匹配规则，假设我们需要匹配属性 'distance' 和 'vector'。
    """
    # print(attr1['vector'], attr2['vector'])
    # return True
    distance_match = abs(attr1['distance'] - attr2['distance']) <10  # 距离允许的误差

    # 将列表转换为 NumPy 数组
    vec1 = np.asarray(attr1['vector'])
    vec2 = np.asarray(attr2['vector'])

    # 矢量相同或反向容差验证
    vector_same = np.allclose(vec1, vec2, atol=1, rtol=1)
    vector_opposite = np.allclose(vec1, -vec2, atol=1, rtol=1)

    # print(vector_match)
    return distance_match and (vector_same or vector_opposite)


def graph_matching(main_graph, sub_graph):
    """
    实现子图匹配算法。

    :param main_graph: 主图（NetworkX 图对象）
    :param sub_graph: 子图（NetworkX 图对象）
    :return: 匹配结果，包含所有匹配的子图节点对的列表
    """

    # 使用 NetworkX 提供的子图同构算法，启用边属性匹配
    matcher = isomorphism.GraphMatcher(main_graph, sub_graph, node_match=node_match,  edge_match=edge_match)

    # 查找所有匹配的子图
    matches = []

    for subgraph_mapping in matcher.subgraph_isomorphisms_iter():
        matches.append(subgraph_mapping)

    return matches

# 示例：加载IFC文件并构建空间关系图
ifc_file0 = ifcopenshell.open(r'C:\Users\admin\Desktop\新建文件夹\测试2.ifc')  # 加载IFC文件
ifc_file1 = ifcopenshell.open(r'C:\Users\admin\Desktop\新建文件夹\测试1.ifc')  # 加载IFC文件

spatial_graph0,distance0,type_ids = build_spatial_graph_from_ifc0(ifc_file0)
spatial_graph1 = build_spatial_graph_from_ifc1(ifc_file1,distance0,type_ids)

#绘制空间关系图
plot_spatial_graph(spatial_graph0)
plot_spatial_graph(spatial_graph1)

# 调用图匹配函数
matches = graph_matching(spatial_graph1, spatial_graph0)

# 输出匹配结果
print("匹配的子图:")
for i, match in enumerate(matches):
    print(f"匹配 {i + 1}: {match}")

print(distance0)


end = time.perf_counter()

# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")

process = psutil.Process(os.getpid())
print(f"当前内存占用：{process.memory_info().rss / 1024**2:.2f} MB")

