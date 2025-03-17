import xml.etree.ElementTree as ET
from copy import deepcopy
import os
from gen import multi_instance

MAX_RECV = 32
MAX_STEP = 64

class PipelineFunc():
    def __init__(self, head_func, tail_func):
        self.is_first_head = head_func
        self.is_first_tail = tail_func

class _tb():
    def __init__(self, tb, gpu_id, func: PipelineFunc):
        self.xml_node = deepcopy(tb)
        self.is_first_head = func.is_first_head(gpu_id, tb)
        self.is_first_tail = func.is_first_tail(gpu_id, tb)
        self.is_recv = int(tb.get('recv')) >= 0

def get_new_wait_step(step_id, depid, deps):
    template = '<step s="0" type="nop" srcbuf="i" srcoff="-1" dstbuf="o" dstoff="-1" cnt="0" depid="8" deps="0" hasdep="0"/>'
    new_step = ET.fromstring(template)
    new_step.set('s', str(step_id))
    new_step.set('depid', str(depid))
    new_step.set('deps', str(deps))
    return new_step

def add_dep_steps(new_tb, wait_steps, step_index):
    # 先递增原本的step_id
    dep_num = len(wait_steps)
    for step in new_tb.findall('step'):
        current_s = int(step.get('s'))
        ori_depid = int(step.get('depid'))
        # 原本的第一个step可能需要添加最后一个wait step的依赖
        if current_s == step_index and ori_depid == -1:
            depid, deps = wait_steps[-1]
            step.set('depid', str(depid))
            step.set('deps', str(deps))
            dep_num -= 1
            break
    for step in new_tb.findall('step'):
        current_s = int(step.get('s'))
        step.set('s', str(current_s + dep_num))
    # 插入新的wait step
    step_id = step_index
    for wait_step in wait_steps[:dep_num]:
        depid, deps = wait_step
        new_step = get_new_wait_step(step_id, depid, deps)
        step_id += 1
        new_tb.append(new_step)
    # 排序step节点
    steps = new_tb.findall('step')
    steps.sort(key=lambda x: int(x.get('s')))
    # 删除原本的step节点
    del new_tb[:]
    # 插入排序好的step节点
    for step in steps:
        new_tb.append(step)
    return new_tb

def how_many_steps_need_append(gpu):
    num_append_steps = {}
    gpu_id = int(gpu.get('id'))
    tbs = []
    tail_num = 0
    # 先储存一圈，然后找一下tail个数，tail会被下一个pp的head依赖
    for tb_xml in gpu.findall('tb'):
        tb = _tb(tb_xml, gpu_id, ppfunc)
        if tb.is_first_tail:
            tail_num += 1
        tbs.append(tb)
    # 计算每个pp增加的steps数量
    for tb in tbs:
        tb_id = int(tb.xml_node.get('id'))
        # 首先每轮固定要生成原本那么多
        num_append_steps[tb_id] = len(tb.xml_node.findall('step'))
        if tb.is_first_head:
            num_append_steps[tb_id] += tail_num
            # 原本自身没有依赖，则会减少一个增加的step
            for step in tb.xml_node.findall('step'):
                current_s = int(step.get('s'))
                ori_depid = int(step.get('depid'))
                if current_s == 0 and ori_depid == -1:
                    num_append_steps[tb_id] -= 1
                    break
    return num_append_steps

def get_depid_deps(depid, deps, pp_index, num_append_steps, increase_depid, gpu):
    distinct_depid = []
    for i in increase_depid[depid][:pp_index+1]:
        if i not in distinct_depid:
            distinct_depid.append(i)
    global_deps = deps + num_append_steps[depid] * pp_index
    for dis_depid in distinct_depid:
        num_steps = len(gpu.find(f'tb[@id="{dis_depid}"]').findall('step'))
        if global_deps >= num_steps:
            global_deps -= num_steps
        else:
            break
    return distinct_depid[-1], global_deps

# 4. 复制stages
def get_new_pipeline_steps(tb: _tb, hold_tb_num, increase_depid, step_index, o_chunks, pp_index, num_append_steps, tail_steps, gpu):
    cur_tb = deepcopy(tb.xml_node)
    # 调整基础数据
    cur_tb.set('id', str(hold_tb_num))
    cur_step_index = step_index
    for step in cur_tb.findall('step'):
        # 修改step index
        step.set('s', str(cur_step_index))
        cur_step_index += 1
        # 修改steps的offset属性
        for attr in ['srcoff', 'dstoff']:
            offset = int(step.get(attr))
            if offset != -1:
                step.set(attr, str(offset + o_chunks * pp_index))
        # 修改depid and deps
        depid = int(step.get("depid"))
        deps = int(step.get("deps"))
        if depid >= 0: # 有依赖
            # 这里是依赖于当前pp, pp_index不为0的时候, 依赖的deps已经发生了变化
            depid, deps = get_depid_deps(depid, deps, pp_index, num_append_steps, increase_depid, gpu)
            step.set("deps", str(deps))
            step.set("depid", str(depid))
    # 增加新的依赖steps
    if tb.is_first_head and pp_index != 0:
        wait_steps = tail_steps.copy()
        for i in range(len(wait_steps)):
            depid, deps = wait_steps[i]
            # 这里依赖的是上一个pp的tail
            # if pp_index > 1:
            depid, deps = get_depid_deps(depid, deps, pp_index-1, num_append_steps, increase_depid, gpu)
            wait_steps[i] = (depid, deps)
        cur_tb = add_dep_steps(cur_tb, wait_steps, step_index)
    
    cur_steps_num = len(cur_tb.findall("step"))
    if step_index + cur_steps_num > MAX_STEP:
        # recv分tb之后，分之前的最后一个和分之后的第一个需要添加依赖, recv不会是head，不用担心调用两次add_dep_steps
        if tb.is_recv:
            tb_index = int(tb.xml_node.get('id'))
            depid = increase_depid[tb_index][pp_index-1]
            last_recv_tb_xml = gpu.find(f'tb[@id="{depid}"]')
            last_recv_step_xml = last_recv_tb_xml.findall('step')[-1]
            last_recv_step_id = int(last_recv_step_xml.get('s'))
            cur_tb = add_dep_steps(cur_tb, [(depid, last_recv_step_id)], step_index)
            last_recv_step_xml.set('hasdep', '1')

        # 限制step index
        s = 0
        for step in cur_tb.findall("step"):
            step.set("s", str(s))
            s += 1

    return cur_tb

class GpuInfo():
    def __init__(self, gpu):
        self.gpu = gpu
        self.chunks = int(gpu.get('chunks'))
        self.gpu_id = int(gpu.get('id'))
        self.tbs = [] # 复制所有tb的信息
        self.tail_steps = {} # 记录每个pp的的第一个阶段的tail tb的最后一个 step
        self.increase_depid = {} # 维护一个全局的depid增长对应关系
        for tb_xml in gpu.findall('tb'):
            # 判断是否是第一个stage，以及是否是head和tail
            tb = _tb(tb_xml, self.gpu_id, ppfunc)
            self.tbs.append(tb)
            if tb.is_first_tail:
                tb_id = int(tb.xml_node.get('id'))
                last_step_id = len(tb.xml_node.findall('step')) - 1
                self.tail_steps = [(tb_id, last_step_id)]
            # 维护一个全局的depid增长关系
            tb_id = int(tb.xml_node.get('id'))
            self.increase_depid[tb_id] = [tb_id]

def multi_pipeline(input_file, output_file, pipeline, ppfunc):
    # 读取XML文件
    tree = ET.parse(input_file)
    root = tree.getroot()
    # 1. 处理root的nchunksperloop
    nchunksperloop = int(root.get("nchunksperloop"))
    root.set("nchunksperloop", str(nchunksperloop * pipeline))
    for gpu in root.findall('.//gpu'):
        # 2. 要处理gpu标签的o_chunks
        o_chunks = int(gpu.get('o_chunks'))
        gpu.set('o_chunks', str(o_chunks*pipeline))
        # 复制并处理所有tb标签
        gpu_id = int(gpu.get('id'))
        original_tbs = gpu.findall('tb')
        tbs = []
        tail_steps = {}
        increase_depid = {}
        for tb_xml in original_tbs:
            # 判断是否是第一个stage，以及是否是head和tail
            tb = _tb(tb_xml, gpu_id, ppfunc)
            tbs.append(tb)
            if tb.is_first_tail:
                tb_id = int(tb.xml_node.get('id'))
                last_step_id = len(tb.xml_node.findall('step')) - 1
                tail_steps = [(tb_id, last_step_id)]
            # 维护一个全局的depid增长关系
            tb_id = int(tb.xml_node.get('id'))
            increase_depid[tb_id] = [tb_id]
        # 4. 复制stage
        num_append_steps = how_many_steps_need_append(gpu)
        hold_tb_num = len(tbs)
        for pp_index in range(1, pipeline):
            for origin_tb_index in range(len(tbs)):
                tb_xml_index = increase_depid[origin_tb_index][-1]
                tb_xml = gpu.find(f'tb[@id="{tb_xml_index}"]')
                tb_copy = tbs[origin_tb_index]
                step_index = len(tb_xml.findall('step'))
                cur_tb = get_new_pipeline_steps(tb_copy, hold_tb_num, increase_depid, step_index, o_chunks, pp_index, num_append_steps, tail_steps, gpu)
                new_steps = cur_tb.findall('step')
                if len(new_steps) > MAX_STEP:
                    raise ValueError(f"Too many steps in one pipeline. MAX_STEP: {MAX_STEP}, current steps: {len(cur_tb.findall('step'))}")
                elif step_index + len(new_steps) > MAX_STEP:
                    tb_xml = cur_tb
                    hold_tb_num += 1
                    gpu.append(cur_tb)
                else:
                    tb_xml.extend(new_steps)
                # 更新每个pp每个tb的id关系
                increase_depid[origin_tb_index].append(int(tb_xml.get('id')))
    # 格式化, 2个空格缩进
    ET.indent(tree, space='  ', level=0)
    # 保存修改后的文件
    tree.write(output_file, encoding='UTF-8', xml_declaration=False)


# 3. 是否是第一个stage head
def is_first_head_ring_8_4(gpu_id, tb_xml):
    send = int(tb_xml.get('send'))
    recv = int(tb_xml.get('recv'))
    if recv == -1 and send >= 0 and (send // 8) != (gpu_id // 8): # send不在同一个8卡中
        return True
    return False

# 3. 是否是第一个stage tail
def is_first_tail_ring_8_4(gpu_id, tb_xml):
    send = int(tb_xml.get('send'))
    recv = int(tb_xml.get('recv'))
    if send == -1 and recv >= 0 and (recv // 8) != (gpu_id // 8): # recv不在同一个8卡中
        return True
    return False

if __name__ == '__main__':
    ppfunc = PipelineFunc(
        head_func=is_first_head_ring_8_4,
        tail_func=is_first_tail_ring_8_4,
    )
    input = "./Neogen_AG/32GPUs/ring8_4/ring_2hosts_32nodes_8_4.xml"
    for pipeline in [16]:
        for instance in [1]:
            output = f"./Neogen_AG/32GPUs_pipeline/ring_8_4_pp_{pipeline}_ins_{instance}/test.xml"
            os.makedirs(os.path.dirname(output), exist_ok=True)
            multi_pipeline(input, output, pipeline, ppfunc)
            multi_instance(output, output, instance)
