# -*- coding: UTF-8 -*-


class EdgeGraph:
    def __init__(self, init_node, nodes):
        self.init_node = init_node  # 初始化节点
        self.nodes_ = nodes         # 与初始化节点相关的边列表

    def nodes(self):
        return self.nodes_         # 返回边列表

    def size(self):
        return len(self.nodes_)    # 返回边列表的长度

    def __len__(self):
        return len(self.nodes_)    # 返回边列表的长度
