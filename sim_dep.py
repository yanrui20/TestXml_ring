import xml.etree.ElementTree as ET
from copy import deepcopy
import os

class GpuCheck():
    def __init__(self, gpu):
        self.gpu_id = int(gpu.get('id'))
        self.tbs = [tb for tb in gpu.findall('tb')]
        self.tb_steps = [[step for step in tb.findall('step')] for tb in gpu.findall('tb')]
        self.tb_num = len(self.tb_steps)
        self.tb_step_index = [0 for _ in range(self.tb_num)]
        self.tb_all_steps = [len(tb) for tb in self.tb_steps]
        self.recv_tb_index = {}
        for i, tb in enumerate(self.tbs):
            recv = int(tb.get('recv'))
            if recv >= 0:
                if recv not in self.recv_tb_index:
                    self.recv_tb_index[recv] = []
                self.recv_tb_index[recv].append(i)
    
    def get_step(self, tb_id):
        cur_step_index = self.tb_step_index[tb_id]
        if cur_step_index >= self.tb_all_steps[tb_id]:
            return None
        return self.tb_steps[tb_id][cur_step_index]

    def get_recv_tb_index(self, send_gpu_id):
        recv_tb_index = None
        if send_gpu_id not in self.recv_tb_index:
            return recv_tb_index
        for tb_id in self.recv_tb_index[send_gpu_id]:
            step = self.get_step(tb_id)
            if step is None:
                continue
            depid = int(step.get('depid'))
            deps = int(step.get('deps'))
            if self.check_deps(depid, deps):
                recv_tb_index = tb_id
                break
        return recv_tb_index

    def check_deps(self, depid, deps):
        return depid < 0 or self.tb_step_index[depid] > deps

    def finish_one_step(self, tb_id):
        self.tb_step_index[tb_id] += 1

def is_can_do(gpu_checks: list[GpuCheck], gpu_id: int, tb_id):
    step = gpu_checks[gpu_id].get_step(tb_id)
    if step is None:
        return False
    depid = int(step.get('depid'))
    deps = int(step.get('deps'))
    # 检查依赖
    if not gpu_checks[gpu_id].check_deps(depid, deps):
        return False
    t = step.get('type')
    # 如果是send step, 要检查对端是否可以接收
    if t == 's':
        peer = int(gpu_checks[gpu_id].tbs[tb_id].get('send'))
        return gpu_checks[peer].get_recv_tb_index(gpu_id) is not None
    elif t == 'r':
        return False
    elif t == 'nop' or t == 'cpy':
        return True
    else:
        raise ValueError(f'unknown step type {t}')

def next_step(gpu_checks: list[GpuCheck], gpu_id: int, tb_id):
    step = gpu_checks[gpu_id].get_step(tb_id)
    if step is None:
        raise ValueError('step is None')
    t = step.get('type')
    if t == 's':
        # send step完成
        gpu_checks[gpu_id].finish_one_step(tb_id)
        # recv step完成
        peer = int(gpu_checks[gpu_id].tbs[tb_id].get('send'))
        recv_tb_index = gpu_checks[peer].get_recv_tb_index(gpu_id)
        gpu_checks[peer].finish_one_step(recv_tb_index)
    elif t == 'r':
        raise ValueError('r step should not be active executed')
    elif t == 'nop' or t == 'cpy':
        gpu_checks[gpu_id].finish_one_step(tb_id)
    else:
        raise ValueError(f'unknown step type {t}')

def all_finished(gpu_checks: list[GpuCheck]):
    for gc in gpu_checks:
        for tb_id in range(gc.tb_num):
            if gc.tb_step_index[tb_id] < gc.tb_all_steps[tb_id]:
                return False
    return True

def dep_check(input_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    gpu_checks = [GpuCheck(gpu) for gpu in root.findall('gpu')]
    flag = True
    while flag:
        flag = False
        for gc in gpu_checks:
            for tb_id in range(gc.tb_num):
                if is_can_do(gpu_checks, gc.gpu_id, tb_id):
                    next_step(gpu_checks, gc.gpu_id, tb_id)
                    flag = True
    print("all_finished", all_finished(gpu_checks))
    print(gpu_checks[0].tb_step_index)
    print(gpu_checks[1].tb_step_index)
    print(gpu_checks[7].tb_step_index)

if __name__ == '__main__':
    input_file = "/Users/yanrui/vscode/nccl/TestXml_ring/Neogen_AG/32GPUs_pipeline/ring_8_4_pp_16_ins_1/test.xml"
    dep_check(input_file)