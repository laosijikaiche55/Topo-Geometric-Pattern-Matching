from ged4py.algorithm.abstract_graph_edit_dist import AbstractGraphEditDistance
import networkx as nx
import sys
import numpy as np


class EdgeEditDistance(AbstractGraphEditDistance):
    """
    Calculates the graph edit distance between two edges.
    A node in this context is interpreted as a graph,
    and edges are interpreted as nodes.
    """

    def __init__(self, g1, g2):
        AbstractGraphEditDistance.__init__(self, g1, g2)

    def insert_cost(self, i, j, nodes2):
        if i == j:
            return 1
        return sys.maxsize

    def delete_cost(self, i, j, nodes1):
        if i == j:
            return 1
        return sys.maxsize

    def substitute_cost(self, node1, node2):


        # 假设边属性字典包含 "weight" 和 "type" 作为关键字
        keys_to_compare = ["label1", "label2","angle"]

        # 生成边属性的唯一标识符
        attributes1 = get_node_identifier(node1[0], keys_to_compare)
        attributes2 = get_node_identifier(node2[0], keys_to_compare)

        angle_attributes1 = get_node_identifier(node1[1], keys_to_compare)
        angle_attributes2 = get_node_identifier(node2[1], keys_to_compare)


        # 比较边的属性
        if compare_nested_with_isclose(attributes1, attributes2, 0.01) and compare_nested_with_isclose(angle_attributes1, angle_attributes2, 0.01):
            return 0.  # 边相似，返回 0
        return 1  # 边不同，返回 1

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