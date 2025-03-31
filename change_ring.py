import xml.etree.ElementTree as ET
from copy import deepcopy
import os
from pathlib import Path
from gen import multi_copy

TYPE = "AG"
NUMGPUS = 32

ring_inter = {
    32: [
    "0 8 16 24  1 9 17 25  2 10 18 26  3 11 19 27  4 12 20 28  5 13 21 29  6 14 22 30  7 15 23 31",
    "31 23 15 7  30 22 14 6  29 21 13 5  28 20 12 4  27 19 11 3  26 18 10 2  25 17 9 1  24 16 8 0"
],
    16: [
    "0 8  1 9  2 10  3 11  4 12  5 13  6 14  7 15",
    "15 7  14 6  13 5  12 4  11 3  10 2  9 1  8 0"
],}

ring_nccl = {
    32: [
        "0 7 6 5 4 3 2 1   9 10 11 12 13 14 15 8   16 23 22 21 20 19 18 17   25 26 27 28 29 30 31 24",
        "1 2 3 4 5 6 7 0   8 15 14 13 12 11 10 9   17 18 19 20 21 22 23 16   24 31 30 29 28 27 26 25",
        "2 1 0 7 6 5 4 3   11 12 13 14 15 8 9 10   18 17 16 23 22 21 20 19   27 28 29 30 31 24 25 26",
        "3 4 5 6 7 0 1 2   10 9 8 15 14 13 12 11   19 20 21 22 23 16 17 18   26 25 24 31 30 29 28 27",
        "4 3 2 1 0 7 6 5   13 14 15 8 9 10 11 12   20 19 18 17 16 23 22 21   29 30 31 24 25 26 27 28",
        "5 6 7 0 1 2 3 4   12 11 10 9 8 15 14 13   21 22 23 16 17 18 19 20   28 27 26 25 24 31 30 29",
        "6 5 4 3 2 1 0 7   15 8 9 10 11 12 13 14   22 21 20 19 18 17 16 23   31 24 25 26 27 28 29 30",
        "7 0 1 2 3 4 5 6   14 13 12 11 10 9 8 15   23 16 17 18 19 20 21 22   30 29 28 27 26 25 24 31",
    ],
    16: [
        "0 7 6 5 4 3 2 1   9 10 11 12 13 14 15 8",
        "1 2 3 4 5 6 7 0   8 15 14 13 12 11 10 9",
        "2 1 0 7 6 5 4 3   11 12 13 14 15 8 9 10",
        "3 4 5 6 7 0 1 2   10 9 8 15 14 13 12 11",
        "4 3 2 1 0 7 6 5   13 14 15 8 9 10 11 12",
        "5 6 7 0 1 2 3 4   12 11 10 9 8 15 14 13",
        "6 5 4 3 2 1 0 7   15 8 9 10 11 12 13 14",
        "7 0 1 2 3 4 5 6   14 13 12 11 10 9 8 15",
    ],
}

def get_new_tb(tb, tb_index, tb_start_index, chan, o_chunks):
    new_tb = deepcopy(tb)
    # 修改chan
    new_tb.set('chan', str(chan)) # 合并的这一份chan都是1
    # 修改id
    new_tb.set('id', str(tb_index))
    # 修改steps
    for step in new_tb.findall('step'):
        # 修改srcoff和dstoff
        for attr in ['srcoff', 'dstoff']:
            srcbuf = step.get("srcbuf")
            if srcbuf == 'o':
                value = int(step.get(attr))
                step.set(attr, str(value + o_chunks * chan))
        # 修改depid
        depid = int(step.get("depid"))
        if depid >= 0:
            step.set("depid", str(depid + tb_start_index))
    return new_tb

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

def merge_ring(inputs, output, instance, inter_inputs=None, inter_instance=0):
    assert instance % len(inputs) == 0, \
        "instance should be divisible by the number of inputs"
    assert inter_instance == 0 or inter_instance % len(inter_inputs) == 0, \
        "inter_instance should be divisible by the number of inter_inputs"
    tree1 = ET.parse(inputs[0])
    root1 = tree1.getroot()

    nchannels = int(root1.get("nchannels"))
    root1.set("nchannels", str(nchannels * (instance+inter_instance)))
    nchunksperloop = int(root1.get("nchunksperloop"))
    root1.set("nchunksperloop", str(nchunksperloop * (instance+inter_instance)))
    root1.set("outofplace", str(1))

    trees = [ET.parse(input) for input in inputs[1:]]
    roots = [tree.getroot() for tree in trees]
    if inter_inputs is not None:
        inter_trees = [ET.parse(input) for input in inter_inputs]
        inter_roots = [tree.getroot() for tree in inter_trees]
    for gpu in root1.findall('.//gpu'):
        o_chunks = int(gpu.get('o_chunks'))
        gpu.set('o_chunks', str(o_chunks*(instance+inter_instance)))
        gpu_id = int(gpu.get('id'))
        
        original_tbs = [gpu.findall('tb')]
        tb_index = len(original_tbs[0])
        ## 先获得其他tree中对应gpu的tb
        for root in roots:
            gpu2 = root.find(f'.//gpu[@id="{gpu_id}"]')
            original_tbs.append(gpu2.findall('tb'))
        ## 按照次序合并tb
        for chan in range(1, instance):
            tb_start_index = tb_index
            for tb in original_tbs[chan % len(original_tbs)]:
                new_tb = get_new_tb(tb, tb_index, tb_start_index, chan, o_chunks)
                tb_index += 1
                gpu.append(new_tb)
        ## 额外增加inter_ring
        if inter_inputs is not None:
            inter_tbs = []
            for root in inter_roots:
                gpu2 = root.find(f'.//gpu[@id="{gpu_id}"]')
                inter_tbs.append(gpu2.findall('tb'))
            for chan in range(instance, instance+inter_instance):
                tb_start_index = tb_index
                for tb in inter_tbs[(chan - instance) % len(inter_tbs)]:
                    new_tb = get_new_tb(tb, tb_index, tb_start_index, chan, o_chunks)
                    tb_index += 1
                    gpu.append(new_tb)
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tree1.write(output, encoding='UTF-8', xml_declaration=False)

def dump_base_rings():
    ring_strs = ring_nccl[NUMGPUS]
    ring_strs += ring_inter[NUMGPUS]
    for i, ring_str in enumerate(ring_strs):
        ring = [int(i) for i in ring_str.split()]
        input = f"./Neogen_{TYPE}/{NUMGPUS}GPUs/sccl_ring_1ch_1ins/test.xml"
        output = f"./Neogen_{TYPE}/{NUMGPUS}GPUs_merge_sccl_sim_nccl/base_ring_index_{i}/test.xml"
        os.makedirs(os.path.dirname(output), exist_ok=True)
        change_ring(input, output, ring)

def dump_seq_base():
    ring_one_node_str = [
        "0 7 6 5 4 3 2 1",
        "1 0 7 6 5 4 3 2",
        "2 1 0 7 6 5 4 3",
        "3 2 1 0 7 6 5 4",
        "4 3 2 1 0 7 6 5",
        "5 4 3 2 1 0 7 6",
        "6 5 4 3 2 1 0 7",
        "7 6 5 4 3 2 1 0",
    ]
    rings_in_one_node = [[int(i) for i in ring_str.split()] for ring_str in ring_one_node_str]
    rings = []
    node = NUMGPUS // 8
    for i in range(8):
        ring = rings_in_one_node[i].copy()
        for j in range(1, node):
            ring += [k + j * 8 for k in rings_in_one_node[(i+j)%8]]
        rings.append(ring)
    for ring in ring_inter[NUMGPUS]:
        ring = [int(i) for i in ring.split()]
        rings.append(ring)
    for i, ring in enumerate(rings):
        input = f"./Neogen_{TYPE}/{NUMGPUS}GPUs/sccl_ring_1ch_1ins/test.xml"
        output = f"./Neogen_{TYPE}/{NUMGPUS}GPUs_sequence_test/base_ring_index_{i}/test.xml"
        os.makedirs(os.path.dirname(output), exist_ok=True)
        change_ring(input, output, ring)

if __name__ == "__main__":
    TYPE = "AG"
    NUMGPUS = 16
    dump_base_rings()
    dir_path = f"./Neogen_{TYPE}/{NUMGPUS}GPUs_merge_sccl_sim_nccl"
    # dump_seq_base()
    # dir_path = f"./Neogen_{TYPE}/{NUMGPUS}GPUs_sequence_test"
    base_ring = [0, 1, 2, 3, 4, 5, 6, 7]
    instance = 16
    inputs = [f"{dir_path}/base_ring_index_{i}/test.xml" for i in base_ring]
    inter_ring = None
    inter_instance = 0
    inter_inputs = None
    if inter_ring:
        inter_inputs = [f"{dir_path}/base_ring_index_{i}/test.xml" for i in inter_ring]
    output = f"{dir_path}/merged_ring_{'_'.join([str(i) for i in base_ring])}_ins{instance}"
    if inter_ring:
        output += f"_inter_ring_{'_'.join([str(i) for i in inter_ring])}_ins{inter_instance}"
    output += "/test.xml"
    merge_ring(inputs=inputs, output=output, instance=instance, inter_inputs=inter_inputs, inter_instance=inter_instance)