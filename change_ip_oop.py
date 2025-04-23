import xml.etree.ElementTree as ET
import os

def change_ip_oop(directory):
    # 递归遍历所有子目录
    for root, dirs, files in os.walk(directory):
        # 筛选当前目录中的 XML 文件
        for filename in files:
            if filename.endswith('.xml'):
                filepath = os.path.join(root, filename)
                # 解析 XML
                tree = ET.parse(filepath)
                root_elem = tree.getroot()
                # 设置属性
                root_elem.set('inplace', '1')
                root_elem.set('outofplace', '1')
                # 写回文件（保留原始格式）
                tree.write(
                    filepath, 
                    encoding='UTF-8', 
                    xml_declaration=False,
                )

change_ip_oop("./Neogen_AG/16GPUs")