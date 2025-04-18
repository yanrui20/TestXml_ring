import xml.etree.ElementTree as ET
import os

AG='allgather'
RS='reducescatter'

SEND = 's' # [src] -> send
RECV = 'r' # recv -> [dst]
COPY = 'cpy' # [src] -> [dst]
# RECV_COPY_SEND = 'rcs' # recv -> [dst] -> send
RECV_REDUCE_COPY_SEND = 'rrcs' # recv + [src] -> [dst] -> send, [src] == [dst]
RECV_REDUCE_COPY = 'rrc' # recv + [src] -> [dst], [src] == [dst]
RECV_REDUCE_SEND = 'rrs' # recv + [src] -> send

class Node():
    def __init__(self, nodeType, parent: "Node" =None):
        self.nodeType = nodeType
        self.parent = None
        if parent:
            parent.add_child(self)
        self.children = []
        self.attributes = {}

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def get_attribute(self, key):
        return self.attributes.get(key)

    def to_xml(self):
        element = ET.Element(self.nodeType)
        for key, value in self.attributes.items():
            element.set(key, value)
        
        for child in self.children:
            child_element = child.to_xml()
            element.append(child_element)
        
        ET.indent(element, space='  ', level=0)
        return element

    def show_xml(self):
        print(ET.tostring(self.to_xml(), encoding='unicode'))

    def store(self, filepath):
        tree = ET.ElementTree(self.to_xml())
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tree.write(filepath, encoding='utf-8', xml_declaration=False)

class AlgoNode(Node):
    def __init__(self, name, nchannels, nchunksperloop, ngpus, coll):
        super().__init__('algo', None)
        self.set_attribute('name', name)
        self.set_attribute('proto', 'Simple')
        self.set_attribute('nchannels', str(nchannels))
        self.set_attribute('nchunksperloop', str(nchunksperloop))
        self.set_attribute('ngpus', str(ngpus))
        self.set_attribute('coll', coll)
        self.set_attribute('inplace', '1')
        self.set_attribute('outofplace', '1')
        self.set_attribute('minBytes', '0')
        self.set_attribute('maxBytes', '0')
        self.num_gpus = ngpus
        self.nchunksperloop = nchunksperloop
        self.coll = coll
    
    def get_gpu(self, gpu_id) -> "GpuNode":
        gpu = self.children[gpu_id]
        if gpu.get_attribute('id') == str(gpu_id):
            return gpu
        for gpu in self.children:
            if gpu.get_attribute('id') == str(gpu_id):
                return gpu
        return None

    def add_gpu(self, gpu_id, i_chunks, o_chunks, s_chunks, one_data_count):
        GpuNode(gpu_id=gpu_id, i_chunks=i_chunks, o_chunks=o_chunks, s_chunks=s_chunks, one_data_count=one_data_count, parent_algo=self)
    
    def check(self):
        if self.coll == AG:
            self.check_ag()
        elif self.coll == RS:
            self.check_rs()

    def check_ag(self):
        is_error = False
        for gpu in self.children:
            for dep in gpu.data_deps:
                if not dep:
                    print(f"GPU {gpu.id} chunk {dep} data is None")
                    is_error = True
                    break
        if not is_error:
            print("Data check passed")
    
    def check_rs(self):
        pass
        print("Not check RS")

class GpuNode(Node):
    def __init__(self, gpu_id, i_chunks, o_chunks, s_chunks, one_data_count, parent_algo:AlgoNode):
        super().__init__('gpu', parent_algo)
        self.set_attribute('id', str(gpu_id))
        self.set_attribute('i_chunks', str(i_chunks))
        self.set_attribute('o_chunks', str(o_chunks))
        self.set_attribute('s_chunks', str(s_chunks))
        self.num_tbs = 0
        self.id = gpu_id
        nchunksperloop = self.parent.nchunksperloop
        ## 记录数据块依赖
        self.data_deps = [None for _ in range(nchunksperloop)]
        ## 原本就有的数据块, init by each algo
        if self.parent.coll == AG:
            # for i in range(gpu_id*one_data_count, (gpu_id+1)*one_data_count):
            #     self.data_deps[i] = (-1, -1)
            pass
        elif self.parent.coll == RS:
            ## 当前gpu所持有的数据块, 是机内rank-1的数据（这一部分无需等待）
            pass
    
    def get_tb(self, tb_id):
        tb = self.children[tb_id]
        if tb.get_attribute('id') == str(tb_id):
            return tb
        for tb in self.children:
            if tb.get_attribute('id') == str(tb_id):
                return tb
        return None
    
    def get_step(self, tb_id, step_id):
        tb = self.get_tb(tb_id)
        if tb:
            step = tb.get_step(step_id)
            if step:
                return step
        return None

    def find_tb(self, send_gpu_id, recv_gpu_id, channel_id):
        for tb in self.children:
            if tb.get_attribute('send') == str(send_gpu_id) and \
                tb.get_attribute('recv') == str(recv_gpu_id) and \
                tb.get_attribute('chan') == str(channel_id):
                return tb
        return None

    def add_tb(self, send_gpu_id, recv_gpu_id, channel_id):
        tb = TbNode(send_gpu_id, recv_gpu_id, channel_id, self)
        return tb
    
    def check_exist(self, start, count, type, flag):
        ## flag == true, data need to be exist
        ## flag == false, data need to be not exist
        for i in range(start, start+count):
            if flag and self.data_deps[i] == None:
                raise ValueError(f"{type}: Data dependency error, the data on chunk {i} in GPU {self.id} is NOT EXIST. start: {start}, end: {start+count}")
            elif not flag and self.data_deps[i] != None:
                raise ValueError(f"{type}: Data dependency error, the data on chunk {i} in GPU {self.id} is EXIST. start: {start}, end: {start+count}")
    
    def set_dep(self, start, count, dep):
        for i in range(start, start+count):
            self.data_deps[i] = dep

    def add_step(self, type, src_gpu: "GpuNode", dst_gpu: "GpuNode", src_chunk_index, dst_chunk_index, count, channel_id):
        send_gpu_id = dst_gpu.id if type in [SEND, RECV_REDUCE_COPY_SEND, RECV_REDUCE_SEND] else -1
        recv_gpu_id = src_gpu.id if type in [RECV, RECV_REDUCE_COPY, RECV_REDUCE_SEND] else -1
        tb = self.find_tb(send_gpu_id, recv_gpu_id, channel_id)
        if not tb:
            tb = self.add_tb(send_gpu_id, recv_gpu_id, channel_id)

        # get deps. if use [src], then steps has deps
        multi_dep = None
        if type in [SEND, COPY, RECV_REDUCE_COPY_SEND, RECV_REDUCE_SEND]:
            multi_dep = set(self.data_deps[src_chunk_index:src_chunk_index+count])
            multi_dep.discard(None)
            multi_dep.discard((-1, -1))
            multi_dep = list(multi_dep)
            for dep in multi_dep:
                if dep[0] == tb.id:
                    multi_dep.remove(dep)
        
        # check deps and set deps
        if type == SEND:
            self.check_exist(src_chunk_index, count, type, flag=True)
        elif type == RECV:
            self.check_exist(dst_chunk_index, count, type, flag=False)
            self.set_dep(dst_chunk_index, count, (tb.id, tb.num_steps))
        elif type == COPY:
            self.check_exist(src_chunk_index, count, type, flag=True)
            self.check_exist(dst_chunk_index, count, type, flag=False)
            self.set_dep(dst_chunk_index, count, (tb.id, tb.num_steps))
            self.set_dep(src_chunk_index, count, None)
        elif type == RECV_REDUCE_COPY:
            assert src_chunk_index == dst_chunk_index, "src and dst chunk index must be same"
            # self.check_exist(src_chunk_index, count, type, flag=True)
            self.set_dep(src_chunk_index, count, (tb.id, tb.num_steps))
        elif type == RECV_REDUCE_COPY_SEND:
            assert src_chunk_index == dst_chunk_index, "src and dst chunk index must be same"
            # self.check_exist(src_chunk_index, count, type, flag=True)
            self.set_dep(src_chunk_index, count, (tb.id, tb.num_steps))
        elif type == RECV_REDUCE_SEND:
            self.check_exist(src_chunk_index, count, type, flag=True)
        
        # add step
        tb.add_step(type, src_chunk_index, dst_chunk_index, count, multi_dep)

class TbNode(Node):
    def __init__(self, send, recv, chan, parent_gpu:GpuNode):
        super().__init__('tb', parent_gpu)
        tb_id = parent_gpu.num_tbs
        parent_gpu.num_tbs += 1
        self.set_attribute('id', str(tb_id))
        self.set_attribute('send', str(send))
        self.set_attribute('recv', str(recv))
        self.set_attribute('chan', str(chan))
        self.num_steps = 0
        self.id = tb_id
    
    def get_step(self, step_id):
        step = self.children[step_id]
        if step.get_attribute('s') == str(step_id):
            return step
        for step in self.children:
            if step.get_attribute('s') == str(step_id):
                return step
        return None

    def add_step(self, type, srcoff, dstoff, cnt, multi_dep):
        if (not multi_dep) or (multi_dep == []):
            depid, deps = (-1, -1)
            StepNode(type, srcoff, dstoff, cnt, depid, deps, self)
        else:
            for i in range(len(multi_dep)):
                depid, deps = multi_dep[i]
                if i < len(multi_dep) - 1:
                    StepNode(type="nop", srcoff=-1, dstoff=-1, cnt=0, depid=depid, deps=deps, parent_tb=self)
                else:
                    StepNode(type, srcoff, dstoff, cnt, depid, deps, self)

class StepNode(Node):
    def __init__(self, type, srcoff, dstoff, cnt, depid, deps, parent_tb:TbNode):
        super().__init__('step', parent_tb)
        if type == "nop":
            srcbuf = 'i'
            dstbuf = 'o'
            srcoff = -1
            dstoff = -1
            cnt = 0
        else:
            if int(self.parent.parent.get_attribute('i_chunks')) > 0:
                srcbuf = dstbuf = 'i'
            elif int(self.parent.parent.get_attribute('o_chunks')) > 0:
                srcbuf = dstbuf = 'o'
            else:
                srcbuf = dstbuf = 's'
        step_id = parent_tb.num_steps
        parent_tb.num_steps += 1
        self.set_attribute('s', str(step_id))
        self.set_attribute('type', type)
        self.set_attribute('srcbuf', srcbuf)
        self.set_attribute('srcoff', str(srcoff))
        self.set_attribute('dstbuf', dstbuf)
        self.set_attribute('dstoff', str(dstoff))
        self.set_attribute('cnt', str(cnt))
        self.set_attribute('depid', str(depid))
        self.set_attribute('deps', str(deps))
        self.set_attribute('hasdep', str(0))

        if depid != -1:
            gpu: GpuNode = self.parent.parent
            dep_step: StepNode = gpu.get_step(depid, deps)
            if dep_step:
                dep_step.set_attribute('hasdep', str(1))
            else:
                raise ValueError(f"Dependency step {depid} not found in GPU {gpu.get_attribute('id')}")

class Chunk():
    def __init__(self, gpu_id, chunk_index, count):
        self.gpu_id = gpu_id
        self.chunk_index = chunk_index
        self.count = count

def init_algo(name, nchannels, nchunksperloop, ngpus, coll) -> AlgoNode:
    algo = AlgoNode(name=name, nchannels=nchannels, nchunksperloop=nchunksperloop, ngpus=ngpus, coll=coll)
    one_data_count = nchunksperloop // ngpus
    i_chunks = o_chunks = s_chunks = 0
    if coll == AG:
        o_chunks = nchunksperloop
    elif coll == RS:
        i_chunks = nchunksperloop
    for i in range(ngpus):
        algo.add_gpu(gpu_id=i, i_chunks=i_chunks, o_chunks=o_chunks, s_chunks=s_chunks, one_data_count=one_data_count)
    return algo

def copy(algo:AlgoNode, src: Chunk, dst: Chunk, channel_id):
    src_gpu = algo.get_gpu(src.gpu_id)
    dst_gpu = algo.get_gpu(dst.gpu_id)
    count = src.count
    src_gpu.add_step(SEND, src_gpu, dst_gpu, src.chunk_index, dst.chunk_index, count, channel_id)
    dst_gpu.add_step(RECV, src_gpu, dst_gpu, src.chunk_index, dst.chunk_index, count, channel_id)

def copy_reduce(algo:AlgoNode, src: Chunk, dst: Chunk, channel_id):
    src_gpu = algo.get_gpu(src.gpu_id)
    dst_gpu = algo.get_gpu(dst.gpu_id)
    count = src.count
    src_gpu.add_step(SEND, src_gpu, dst_gpu, src.chunk_index, dst.chunk_index, count, channel_id)
    dst_gpu.add_step(RECV_REDUCE_COPY, src_gpu, dst_gpu, src.chunk_index, dst.chunk_index, count, channel_id)
