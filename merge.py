import xml.etree.ElementTree as ET
from copy import deepcopy
import os
from pathlib import Path

def get_new_tb(tb, tb_index, chan, o_chunks):
    new_tb = deepcopy(tb)
    # 修改chan
    new_tb.set('chan', str(chan)) # 合并的这一份chan都是1
    # 修改id
    new_tb.set('id', str(tb_index))
    # 修改srcoff和dstoff
    for step in new_tb.findall('step'):
        for attr in ['srcoff', 'dstoff']:
            srcbuf = step.get("srcbuf")
            if srcbuf == 'o':
                value = int(step.get(attr))
                step.set(attr, str(value + o_chunks * chan))
    return new_tb

def merge_xml(input1, input2, output, instance):
    tree1 = ET.parse(input1)
    root1 = tree1.getroot()

    nchannels = int(root1.get("nchannels"))
    root1.set("nchannels", str(nchannels * instance))
    nchunksperloop = int(root1.get("nchunksperloop"))
    root1.set("nchunksperloop", str(nchunksperloop * instance))

    tree2 = ET.parse(input2)
    root2 = tree2.getroot()
    for gpu in root1.findall('.//gpu'):
        o_chunks = int(gpu.get('o_chunks'))
        gpu.set('o_chunks', str(o_chunks*instance))
        
        original_tbs = [gpu.findall('tb'), None]
        tb_index = len(original_tbs[0])
        ## 先获得tree2中对应gpu的tb
        tree2_gpus = root2.findall('.//gpu')
        for gpu2 in tree2_gpus:
            if gpu2.get('id') == gpu.get('id'):
                original_tbs[1] = gpu2.findall('tb')
        ## 按照次序合并tb
        for chan in range(1, instance):
            for tb in original_tbs[chan % 2]:
                new_tb = get_new_tb(tb, tb_index, chan, o_chunks)
                tb_index += 1
                gpu.append(new_tb)
     
    tree1.write(output, encoding='UTF-8', xml_declaration=False)


if __name__ == '__main__':
    inputs = [
        "/Users/yanrui/vscode/nccl/TestXml_ring/Neogen_AG/32GPUs/ring8_4/ring_2hosts_32nodes_8_4.xml",
        "/Users/yanrui/vscode/nccl/TestXml_ring/Neogen_AG/32GPUs/ring2_2_2_4/ring_2hosts_32nodes_2_2_2_4.xml",
        "/Users/yanrui/vscode/nccl/TestXml_ring/Neogen_AG/32GPUs/ring4_2_2_2/ring_2hosts_32nodes_4_2_2_2.xml",
    ]
    output_dir = "/Users/yanrui/vscode/nccl/TestXml_ring/Neogen_AG/32GPUs_Merge"
    for index1 in range(len(inputs)):
        for index2 in range(index1+1, len(inputs)):
            input1 = inputs[index1]
            input2 = inputs[index2]
            name1 = input1.split("/")[-2]
            name2 = input2.split("/")[-2]
            for instance in [2, 4, 8, 16]:
                output = f"{output_dir}/merged_{name1}_{name2}_ins_{instance}/merged.xml"
                os.makedirs(os.path.dirname(output), exist_ok=True)
                merge_xml(
                    input1=input1, 
                    input2=input2,
                    output=output,
                    instance=instance,
                )