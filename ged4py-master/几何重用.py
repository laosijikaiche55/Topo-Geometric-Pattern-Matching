import ifcopenshell
import ifcopenshell.api
import numpy as np


# 加载 IFC 文件
file = ifcopenshell.open("example.ifc")

# 获取源构件的几何表示（例如通过 GUID 查找）
source_element = file.by_guid("SOURCE_ELEMENT_GUID")

# 提取源构件的几何定义（假设其使用 Body 表示）
source_representation = ifcopenshell.util.representation.get_representation(
    source_element, "Model", "Body", "MODEL_VIEW"
)

# 创建映射几何（IfcMappedItem）
mapped_geometry = ifcopenshell.api.run("geometry.map_representation", file,
    representation=source_representation,
    mapping_target=None,
    mapping_source_matrix=np.eye(4)  # 初始映射矩阵（单位矩阵，表示无偏移）
)

# 创建目标构件（例如新窗户）
target_element = ifcopenshell.api.run("root.create_entity", file, ifc_class="IfcWindow")

# 将映射后的几何关联到目标构件
ifcopenshell.api.run("geometry.assign_representation", file,
    product=target_element,
    representation=mapped_geometry
)

# 调整目标构件的位置（假设需要放置在坐标 (5, 0, 0)）
matrix = np.array([
    [1, 0, 0, 5],  # X 方向平移 5 单位
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
ifcopenshell.api.run("geometry.edit_object_placement", file,
    product=target_element,
    matrix=matrix
)

# 保存修改后的文件
file.write("modified_example.ifc")