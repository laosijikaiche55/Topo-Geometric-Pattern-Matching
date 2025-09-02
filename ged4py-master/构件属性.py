import ifcopenshell
from typing import List, Dict, Any


def build_property_tree(element) -> Dict[str, Any]:
    """构建构件实例的属性关系树

    Args:
        element: IFC构件实例 (如IfcWall, IfcDoor等)

    Returns:
        树状结构字典，格式：
        {
            "type": "Element",
            "guid": element.GlobalId,
            "name": element.Name,
            "children": [关系节点1, 关系节点2...]
        }
    """
    tree = {
        "type": "Element",
        "name": element.Name or "Unnamed",
        "children": []
    }

    # 遍历所有关联的属性定义关系
    for rel in get_defines_by_properties(element):
        rel_node = process_relation(rel)
        tree["children"].append(rel_node)

    return tree


def get_defines_by_properties(element):
    """获取构件关联的所有IfcRelDefinesByProperties关系"""
    return [rel for rel in element.IsDefinedBy
            if rel.is_a("IfcRelDefinesByProperties")]


def process_relation(rel) -> Dict[str, Any]:
    """处理单个关系节点"""
    prop_def = rel.RelatingPropertyDefinition
    return {
        "type": "IfcRelDefinesByProperties",
        "name": f"PropertyRelation",
        "children": [process_property_definition(prop_def)]
    }


def process_property_definition(prop_def) -> Dict[str, Any]:
    """处理属性定义节点"""
    node = {
        "type": prop_def.is_a(),
        "name": prop_def.Name or "Unnamed",
        "children": []
    }

    # 处理不同类型属性定义
    if prop_def.is_a("IfcPropertySet"):
        for prop in prop_def.HasProperties:
            node["children"].append(process_property(prop))
    elif prop_def.is_a("IfcElementQuantity"):
        for quantity in prop_def.Quantities:
            node["children"].append(process_quantity(quantity))

    return node


def process_property(prop) -> Dict[str, Any]:
    """处理单个属性"""
    return {
        "type": prop.is_a(),
        "name": prop.Name,
        "value": get_property_value(prop),
    }


def process_quantity(quantity) -> Dict[str, Any]:
    """处理工程量参数"""
    return {
        "type": quantity.is_a(),
        "name": quantity.Name,
        "value": quantity[3].wrappedValue,  # 提取QuantityValue
    }


def get_property_value(prop):
    """获取属性值（支持多种类型）"""
    if prop.is_a("IfcPropertySingleValue"):
        return prop.NominalValue.wrappedValue if prop.NominalValue else None
    elif prop.is_a("IfcPropertyEnumeratedValue"):
        return [v.wrappedValue for v in prop.EnumerationValues]
    elif prop.is_a("IfcPropertyTableValue"):
        return {"DefiningValues": prop.DefiningValues,
                "DefinedValues": prop.DefinedValues}
    # 其他类型处理...
    return "UnsupportedType"


# 加载IFC文件
ifc_file = ifcopenshell.open(r"E:\原电脑d盘新\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\3窗户.ifc")

# 获取任意构件实例（示例取第一个IfcWall）
for element in ifc_file.by_type("IfcElement"):

    # 生成属性关系树
    property_tree = build_property_tree(element)

    # 打印树结构
    import json
    # print(json.dumps(property_tree, indent=2))


def compare_elements(element1, element2) -> bool:
    """比较两个元素是否具有完全相同的属性结构"""
    tree1 = build_normalized_tree(element1)
    tree2 = build_normalized_tree(element2)
    if not tree1 == tree2:
        print("1")
    return tree1 == tree2


def find_identical_elements(ifc_file) -> List[List[ifcopenshell.entity_instance]]:
    """查找文件中所有属性结构相同的元素组"""
    signature_map = {}

    # 遍历所有可比较元素（可根据需要调整类型过滤）
    for element in ifc_file.by_type("IfcElement"):
        # 生成规范化签名
        signature = get_tree_signature(element)

        if signature not in signature_map:
            signature_map[signature] = []
        signature_map[signature].append(element)

    # 返回包含2个及以上元素的组
    return [group for group in signature_map.values() if len(group) >= 2]


# 以下为支持函数（基于之前代码修改）
def build_normalized_tree(element) -> Dict[str, Any]:
    """生成标准化属性树（忽略GUID）"""
    raw_tree = build_property_tree(element)
    return remove_guids(raw_tree)


def remove_guids(node: Dict) -> Dict:
    """递归移除所有GUID字段"""
    node.pop('guid', None)
    if 'children' in node:
        # 先排序子节点保证顺序一致性
        node['children'] = sorted(
            [remove_guids(child) for child in node['children']],
            key=lambda x: (x['type'], x.get('name', ''))
        )
    return node


def get_tree_signature(element) -> str:
    """生成唯一的树结构签名"""
    normalized = build_normalized_tree(element)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


# 修改后的build_property_tree（增加子节点排序）
def build_property_tree(element):
    tree = {
        "type": "Element",
        "guid": element.GlobalId,
        "name": element.Name or "Unnamed",
        "children": []
    }

    # 获取并按名称排序属性关系
    relations = sorted(get_defines_by_properties(element),
                       key=lambda r: r.RelatingPropertyDefinition.Name)

    for rel in relations:
        rel_node = process_relation(rel)
        tree["children"].append(rel_node)

    return tree


def process_relation(rel):
    prop_def = rel.RelatingPropertyDefinition
    return {
        "type": "IfcRelDefinesByProperties",
        "guid": rel.GlobalId,
        "name": prop_def.Name or "Unnamed",
        "children": [process_property_definition(prop_def)]
    }


def process_property_definition(prop_def):
    node = {
        "type": prop_def.is_a(),
        "guid": prop_def.GlobalId,
        "name": prop_def.Name or "Unnamed",
        "children": []
    }

    # 属性/工程量排序
    if prop_def.is_a("IfcPropertySet"):
        sorted_props = sorted(prop_def.HasProperties, key=lambda p: p.Name)
        for prop in sorted_props:
            node["children"].append(process_property(prop))
    elif prop_def.is_a("IfcElementQuantity"):
        sorted_quantities = sorted(prop_def.Quantities, key=lambda q: q.Name)
        for quantity in sorted_quantities:
            node["children"].append(process_quantity(quantity))

    return node


# matching_groups = find_identical_elements(ifc_file)
#
# for i, group in enumerate(matching_groups, 1):
#     print(f"\n组 {i} 包含 {len(group)} 个相同元素:")
#     for elem in group:
#         print(f"|— {elem.is_a()} [{elem.GlobalId}] {elem.Name or ''}")


element1 = ifc_file.by_guid("22nJrftH56CgMlpkibc6tY")
element2 = ifc_file.by_guid("3SxA4l5uv4IvX5R2fRpnoT")dw

if compare_elements(element1, element2):
    print(f"元素 {element1.GlobalId} 和 {element2.GlobalId} 具有相同属性结构")