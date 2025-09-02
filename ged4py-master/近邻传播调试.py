import networkx as nx
import numpy as np
import numpy as np
import sympy as sp

# 假设 g1 和 g2 是 NetworkX 图，节点属性包含 'metric_tensor' 和 'inertia_tensor' 键
# 容差值
tolerance = 0.01


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
            return np.all(np.isclose(tuple1, tuple2, atol=tolerance))
        return tuple1 == tuple2  # 非数值类型，直接比较相等性

def compare_array(A,B, tolerance):
    # 将NumPy数组转换为SymPy矩阵
    A_sym = sp.Matrix(A)
    B_sym = sp.Matrix(B)

    # 计算行最简形式
    RREF_A, _ = A_sym.rref()
    RREF_B, _ = B_sym.rref()

    # 将行最简形式转换为NumPy数组
    RREF_A_np = np.array(RREF_A).astype(float)
    RREF_B_np = np.array(RREF_B).astype(float)


    # 检查行最简形式是否在容差范围内近似相等
    are_equivalent = np.allclose(RREF_A_np, RREF_B_np, atol=tolerance)

    return are_equivalent



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


def get_node_identifier1(node_attrs, keys_to_compare):
    for key in keys_to_compare:
        if key in node_attrs:
            value = node_attrs[key]
    return tuple(value)


# 找到初始种子节点，两个图中属性相似的节点
def find_seed_node(g1, g2):
    for node1 in g1.nodes(data=True):
        for node2 in g2.nodes(data=True):

            a1 = node1[1]["label1"]
            a2 = node2[1]["label1"]

            b1 = node1[1]["label2"]
            b2 = node2[1]["label2"]

            if ((compare_nested_with_isclose(node1[1]["label1"], node2[1]["label1"],tolerance ) ) and compare_array(node1[1]["label2"], node2[1]["label2"],tolerance)):
                return node1[0], node2[0]  # 返回种子节点的ID
    return None, None


# 基于种子节点使用近邻传播算法来检查图一致性
def propagate_neighbors(g1, g2, seed1, seed2, visited=None):
    if visited is None:
        visited = set()

    stack = [(seed1, seed2)]

    while stack:
        node1, node2 = stack.pop()

        if (node1, node2) in visited:
            continue

        visited.add((node1, node2))

        # 比较邻居
        neighbors1 = set(g1.neighbors(node1))
        neighbors2 = set(g2.neighbors(node2))

        # 如果邻居数量不一样，图不一致
        if len(neighbors1) != len(neighbors2):
            return False

        # 对邻居进行匹配
        for n1 in neighbors1:
            matched = False
            for n2 in neighbors2:

                a = g1.nodes[n1]["label1"]


                tolerance = 1  # 可以根据需要调整容差

                if ((compare_nested_with_isclose(g1.nodes[n1]["label1"] ,g2.nodes[n2]["label1"], tolerance)) and compare_array(g1.nodes[n1]["label2"] ,g2.nodes[n2]["label2"],
                                                                                                           tolerance)):
                    stack.append((n1, n2))
                    neighbors2.remove(n2)
                    matched = True
                    break
            if not matched:
                return False

    return True





