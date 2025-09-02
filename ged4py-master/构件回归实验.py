import ifcopenshell


def extract_type_ids(ifc_file_path):
    # 加载IFC文件
    ifc_file = ifcopenshell.open(ifc_file_path)

    # 遍历所有构件（例如IfcBuildingElementProxy）
    for element in ifc_file.by_type("IfcBuildingElementProxy"):
        # 获取构件的所有属性集
        property_sets = element.IsDefinedBy

        for prop_set in property_sets:
            if prop_set.is_a("IfcRelDefinesByProperties"):
                # 提取属性集中的具体属性
                properties = prop_set.RelatingPropertyDefinition.HasProperties

                for prop in properties:
                    # 筛选目标属性：Name为'Id类(Type Id)'的IfcPropertySingleValue
                    if prop.Name == 'Id类型(Type Id)' and prop.is_a("IfcPropertySingleValue"):
                        # 提取属性值
                        type_id = prop.NominalValue.wrappedValue
                        print(f"构件 GlobalId: {element.GlobalId}")
                        print(f"Type ID: {type_id}\n")

                        break


# 示例调用
extract_type_ids(r'E:\原电脑d盘新\BIM模型\BIM\行政宿舍楼装修模型\复杂度实验\开关2.ifc')