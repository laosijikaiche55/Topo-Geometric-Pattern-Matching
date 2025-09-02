import ifcopenshell


def compare_ifc_elements(element1, element2, compare_global_id=False):
    """
    比较两个IFC构件是否为同一类型，并获取属性字典中相同的属性值。

    Args:
        element1: 第一个IFC构件对象。
        element2: 第二个IFC构件对象。
        compare_global_id: 是否将 GlobalId 参与对比，默认为 False。

    Returns:
        str: 比较结果的描述。
    """
    # 判断是否为同一类型
    if element1.is_a() == element2.is_a():
        # 提取属性值
        props1 = get_element_properties(element1)
        props2 = get_element_properties(element2)

        # 可选地将 GlobalId 参与对比
        if compare_global_id is False:
            props1= {key: value for key, value in props1.items() if key != "GlobalId"or""}
            props2= {key: value for key, value in props2.items() if key != "GlobalId"or""}


        # 获取相同的属性键值对
        common_properties = [
            {"key": key, "value": props1[key]}
            for key in props1
            if key in props2 and props1[key] == props2[key]
        ]

        l1 = len(props1)-1
        l2 = len(props2)-1

        # 返回结果
        if common_properties:
            sim = len(common_properties)*2/(l1+l2)
            print(sim)
            return f"两个IFC构件为同一类型，共有 {len(common_properties)} 个相同的属性值，相同属性包括：{common_properties}"

        else:
            return "两个IFC构件为同一类型，但没有相同的属性值。"
    else:
        return "两个IFC构件不是同一类型。"




def get_element_properties(element):
    """
    获取IFC构件的所有属性及其值。

    Args:
        element: IFC构件对象。

    Returns:
        dict: 属性和值的字典。
    """
    properties = {}
    if element.IsDefinedBy:
        for definition in element.IsDefinedBy:
            if definition.is_a("IfcRelDefinesByProperties"):
                property_set = definition.RelatingPropertyDefinition
                if property_set.is_a("IfcPropertySet"):
                    for prop in property_set.HasProperties:
                        if prop.is_a("IfcPropertySingleValue"):
                            properties[prop.Name] = prop.NominalValue.wrappedValue
    return properties


# 示例用法
# 假设加载了IFC文件并获取了两个IFC构件
ifc_file = ifcopenshell.open( r"D:\BIM模型\BIM\中建壹品汉芯公馆\pkpm导出\三门.ifc")
element1 = ifc_file.by_id('2RO06PSDP0iAQ$NGeIThuk')  # 替换为实际ID
element2 = ifc_file.by_id('3trNfaQ8H3E9F2vIHZfS53')  # 替换为实际ID

result = compare_ifc_elements(element1, element2)
print(result)
