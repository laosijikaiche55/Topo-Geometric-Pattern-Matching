import os
import sys
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Iterator
import time

import line_profiler

from OCC.Core.gp import gp_Pnt

from OCC.Core.gp import gp_Vec
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.Graphic3d import Graphic3d_ClipPlane

from OCC.Display.SimpleGui import init_display


try:
    import ifcopenshell
    import ifcopenshell.geom
except ModuleNotFoundError:
    print("ifcopenshell package not found.")
    sys.exit(0)

def get_pointset(ifc_file_path, element):

    x_set = []
    y_set = []
    z_set = []
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_PYTHON_OPENCASCADE, True)

    ifc_filename = os.path.join(ifc_file_path)
    assert os.path.isfile(ifc_filename)
    ifc_file = ifcopenshell.open(ifc_filename)

    vertices = []  # 用于存储顶点集合
    center_point = gp_Pnt()  # 初始化中心点

    if element.Representation is not None:
        try:
            pdct_shape = ifcopenshell.geom.create_shape(settings, inst=element)
            compound = pdct_shape.geometry
            iterator = TopoDS_Iterator(compound)

            num_vertices = 0  # 计算顶点数量
            sum_x = sum_y = sum_z = 0  # 用于计算顶点坐标总和

            min_x = min_y = min_z = float('inf')  # 初始化最小值为正无穷
            max_x = max_y = max_z = float('-inf')  # 初始化最大值为负无穷


            existing_points = set()  # 存储已经存在的点

            while iterator.More():
                sub_shape = iterator.Value()

                # 使用 TopExp_Explorer 来探索子形状中的顶点
                explorer = TopExp_Explorer(sub_shape, TopAbs_VERTEX)
                while explorer.More():
                    vertex = explorer.Current()
                    point = BRep_Tool().Pnt(vertex)

                    # 将顶点坐标乘以1000
                    x = point.X() * 1000
                    y = point.Y() * 1000
                    z = point.Z() * 1000


                    # 检查是否存在相同坐标的点
                    if (x, y, z) not in existing_points:
                        vertices.append((x, y, z))
                        existing_points.add((x, y, z))
                        # print(len( vertices))

                    # 更新中心点坐标总和
                    sum_x += x
                    sum_y += y
                    sum_z += z
                    num_vertices += 1

                    # 更新最小值和最大值
                    x_set.append(x)
                    y_set.append(y)
                    z_set.append(z)

                    explorer.Next()

                # 移动到下一个子形状
                iterator.Next()

            # 计算中心点坐标
            if num_vertices > 0:
                center_point.SetCoord(sum_x / num_vertices, sum_y / num_vertices, sum_z / num_vertices)

        except RuntimeError:
            print("Failed to process shape geometry")

    if x_set and y_set and z_set:

        max_x = max(x_set)
        max_y = max(y_set)
        max_z = max(z_set)
        min_x = min(x_set)
        min_y = min(y_set)
        min_z = min(z_set)

        x1 = (max_x + min_x) / 2
        y1 = (max_y + min_y) / 2
        z1 = (max_z + min_z) / 2

        center_coordinates = (x1, y1, z1)
        return vertices, center_coordinates

# # 示例用法
# ifc_file_path = r"C:\Users\visac\Desktop\IFC\IFC模型实验\IFC模型实验\构件模型\单墙开窗.ifc"
# ifc_file = ifcopenshell.open(ifc_file_path)
# elements = ifc_file.by_type('IfcElement')
# for element in elements:
#     print(element)
#     vertices,  center_coordinates = get_pointset(ifc_file_path, element)
#     print(vertices)
#     print(center_coordinates)



# profile = line_profiler.LineProfiler(get_pointset)  # 把函数传递到性能分析器
# profile.enable()  # 开始分析
# get_pointset(ifc_file_path, element)
# profile.disable()  # 停止分析
# profile.print_stats()  # 打印出性能分析结果
