# -*- coding: UTF-8 -*-
from __future__ import print_function
from ged4py.algorithm.abstract_graph_edit_dist import AbstractGraphEditDistance
from ged4py.algorithm.edge_edit_dist import EdgeEditDistance
from ged4py.graph.edge_graph import EdgeGraph
import sys
import numpy as np
from networkx import __version__ as nxv



def compare(g1, g2, print_details=False):
    ged = GraphEditDistance(g1, g2)

    if print_details:
        ged.print_matrix()

    return ged.normalized_distance()


class GraphEditDistance(AbstractGraphEditDistance):

    def __init__(self, g1, g2):
        AbstractGraphEditDistance.__init__(self, g1, g2)

    def substitute_cost(self, node1, node2):
        return self.relabel_cost(node1, node2) + self.edge_diff(node1, node2)


    def relabel_cost(self, node1, node2):
        keys_to_compare = ["label1", "label2"]
        # 从图中提取节点属性
        attrs1 = self.g1.nodes[node1]
        attrs2 = self.g2.nodes[node2]

        # 生成唯一标识符
        attributes1 = get_node_identifier(attrs1, keys_to_compare)
        attributes2 = get_node_identifier(attrs2, keys_to_compare)
        tolerance = 0.1  # 可以根据需要调整容差

        if not compare_nested_with_isclose(attributes1, attributes2, tolerance):
            return 1.0  # 属性不匹配的代价
        return 0.0

    def delete_cost(self, i, j, nodes1):
        if i == j:
            return 1
        return sys.maxsize

    def insert_cost(self, i, j, nodes2):
        if i == j:
            return 1
        else:
            return sys.maxsize

    def pos_insdel_weight(self, node):
        return 1

    def edge_diff(self, node1, node2):
        edges1 = list(self.g1.edges(node1))
        edges2 = list(self.g2.edges(node2))
        if len(edges1) == 0 or len(edges2) == 0:
            return max(len(edges1), len(edges2))

        # 获取与 node1 和 node2 连接的所有边和节点的属性
        edges1_attrs = [(self.g1.nodes[edge[1]], self.g1.edges[edge]) for edge in edges1]
        edges2_attrs = [(self.g2.nodes[edge[1]], self.g2.edges[edge]) for edge in edges2]

        edit_edit_dist = EdgeEditDistance(EdgeGraph(node1,edges1_attrs), EdgeGraph(node2,edges2_attrs))
        return edit_edit_dist.normalized_distance()


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