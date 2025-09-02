
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
import psutil
import os



# 将 metric_tensor 转换为 NumPy 数组
tensor = np.array[[203,456,0,0],[-203,-456,4050,0],[0,0,-4050,0]]

# 计算矩阵乘积
metric_tensor1 = np.dot(tensor.T, tensor)
metric_tensor = np.round(metric_tensor1, 5)

print("1")