import xml.etree.ElementTree as ET
from copy import deepcopy
import os
from pathlib import Path

def change_ring(input_file, output_file, ring):
    # 读取XML文件
    tree = ET.parse(input_file)
    root = tree.getroot()

    for gpu in root.findall('.//gpu'):
        original_tbs = gpu.findall('tb')
        gpu_id = int(gpu.get('id'))
        ring_index = ring.index(gpu_id)
        send = ring[(ring_index + 1) % len(ring)]
        recv = ring[(ring_index - 1) % len(ring)]
        for tb in original_tbs:
            tb.set('send', str(send))
            tb.set('recv', str(recv))
    # 保存修改后的文件
    tree.write(output_file, encoding='UTF-8', xml_declaration=False)

if __name__ == '__main__':
    ring_str = "0  7  6  5  4  3  2  1  9 10 11 12 13 14 15  8 16 23 22 21 20 19 18 17 25 26 27 28 29 30 31 24"
    ring = [int(i) for i in ring_str.split()]
    input = "/Users/yanrui/vscode/nccl/TestXml_ring/Neogen_AG/32GPUs/sccl_ring_1ch_16ins/test.xml"
    output = "/Users/yanrui/vscode/nccl/TestXml_ring/Neogen_AG/32GPUs/sccl_ring_1ch_16ins_copy_nccl_ring/test.xml"
    change_ring(input, output, ring)