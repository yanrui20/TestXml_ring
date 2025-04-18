import math
from .lang import *

def heterogeneous_channel_ring_only_4GPUs_inter_mechine(coll, dims, channels):
    nchannels = max(channels)
    ngpus = math.prod(dims) # 乘积
    # 每个gpu的一份数据被分成了多少个count
    one_data_count = math.lcm(*channels) # 最小公倍数
    nchunksperloop = ngpus * one_data_count
    
    ## 初始化xml
    algo = init_algo(name="heterogeneous_channel_ring_only_4GPU_inter_mechine", nchannels=nchannels, nchunksperloop=nchunksperloop, ngpus=ngpus, coll=coll)
    if coll == AG:
        # NOTE 假设GPU i的原本数据都在第0块, 每一块是one_data_count = nchunksperloop/ngpus
        ## init data deps
        for gpu in algo.children:
            gpu.set_dep(0, one_data_count, (-1, -1))
        # 7->0,1->2,3->4,5->6
        chunk_size = one_data_count
        assert chunk_size % channels[0] == 0
        count = chunk_size // channels[0]
        for channel_id in range(channels[0]):
            for index in range(1, ngpus, 2):
                src_rank = index
                dst_rank = (index + 1) % dims[0] + index // dims[0] * dims[0]
                chunk_src_index = channel_id * count
                src = Chunk(src_rank, chunk_src_index, count)
                chunk_dst_index = dims[1] * chunk_size + channel_id * count ## 传输过去放在哪里
                dst = Chunk(dst_rank, chunk_dst_index, count)
                copy(algo, src, dst, channel_id)
        # 机间传输
        chunk_size = one_data_count
        assert chunk_size % channels[1] == 0
        count = chunk_size // channels[1]
        for step in range(dims[1]-1):
            for channel_id in range(channels[1]):
                for index in range(ngpus):
                    src_rank = index if index % 2 == 0 else (index + 1) % dims[0] + index // dims[0] * dims[0]
                    dst_rank = (src_rank + dims[0]) % ngpus
                    _step = step if index % 2 == 0 else (step + dims[1])
                    chunk_src_index = _step * chunk_size + channel_id * count
                    src = Chunk(src_rank, chunk_src_index, count)
                    chunk_dst_index = (_step + 1) * chunk_size + channel_id * count
                    dst = Chunk(dst_rank, chunk_dst_index, count)
                    copy(algo, src, dst, channel_id)
        # 7->0,1->2,3->4,5->6 反向回传
        chunk_size = one_data_count * (dims[1]-1)
        this_chan = 2
        assert chunk_size % this_chan == 0
        count = chunk_size // this_chan
        for channel_id in range(this_chan):
            for index in range(1, ngpus, 2):
                src_rank = (index + 1) % dims[0] + index // dims[0] * dims[0]
                dst_rank = index
                chunk_src_index = one_data_count * (dims[1]+1) + channel_id * count
                src = Chunk(src_rank, chunk_src_index, count)
                chunk_dst_index = one_data_count + channel_id * count
                dst = Chunk(dst_rank, chunk_dst_index, count)
                copy(algo, src, dst, channel_id)
        # 机内传输
        chunk_size = one_data_count * dims[1]
        assert chunk_size % channels[0] == 0
        count = chunk_size // channels[0]
        for step in range(dims[0]-1):
            for channel_id in range(channels[0]):
                for index in range(ngpus):
                    if step == 0 and index % 2 == 1:
                        continue
                    rank = index
                    next_rank = (index + 1) % dims[0] + index // dims[0] * dims[0]
                    chunk_src_index = step * chunk_size + channel_id * count
                    src = Chunk(rank, chunk_src_index, count)
                    chunk_dst_index = (step + 1) * chunk_size + channel_id * count
                    dst = Chunk(next_rank, chunk_dst_index, count)
                    copy(algo, src, dst, channel_id)

    elif coll == RS:
        ## 当前gpu所持有的数据块, gpu 0->0,1号数据块，gpu 8->0,1号数据块，gpu 1->2,3号数据块...
        ## 最终, gpu 0->0号数据块，gpu 8->1号数据块，gpu 1->2号数据块...
        ## init data deps
        for gpu in algo.children:
            pre_gpu_id = (gpu.id - 1) % dims[0]
            gpu.set_dep(pre_gpu_id * one_data_count * dims[1], one_data_count * dims[1], (-1, -1))
            if gpu.id % 2 == 1:
                gpu.set_dep((gpu.id % dims[0] * dims[1] + 1) * one_data_count, one_data_count * (dims[1]-1), (-1, -1))
        # 1->0, 3->2, 5->4, 7->6
        chunk_size = one_data_count * (dims[1]-1)
        this_chan = 2
        assert chunk_size % this_chan == 0
        count = chunk_size // this_chan
        for index in range(1, ngpus, 2):
            for channel_id in range(this_chan):
                src_rank = index
                dst_rank = index - 1
                chunk_src_index = (src_rank % dims[0] * dims[1] + 1) * one_data_count + channel_id * count
                src = Chunk(src_rank, chunk_src_index, count)
                dst = Chunk(dst_rank, chunk_src_index, count)
                copy_reduce(algo, src, dst, channel_id)
        # 机内dims[0]卡做rs，每个GPU持有dims[1]块的数据
        chunk_size = one_data_count * dims[1]
        assert chunk_size % channels[0] == 0
        count = chunk_size // channels[0]
        for step in range(dims[0]-1):
            for index in range(ngpus):
                for channel_id in range(channels[0]):
                    if step == dims[0]-1:
                        if index % 2 == 0:
                            continue
                    rank = index
                    next_rank = (rank + 1) % dims[0] + rank // dims[0] * dims[0]
                    chunk_gpu_id = (rank - 1 - step) % dims[0]
                    chunk_index = chunk_gpu_id * chunk_size + channel_id * count
                    src = Chunk(rank, chunk_index, count)
                    dst = Chunk(next_rank, chunk_index, count)
                    copy_reduce(algo, src, dst, channel_id)
        # 机间传输
        chunk_size = one_data_count
        this_chan = channels[1]
        assert chunk_size % this_chan == 0
        count = chunk_size // this_chan
        for step in range(dims[1]-1):
            for index in range(ngpus):
                for channel_id in range(this_chan):
                    ori_rank = index
                    rank = ori_rank if index % 2 == 0 else ori_rank - 1
                    next_rank = (rank + dims[0]) % ngpus
                    chunk_id = ori_rank % dims[0] * dims[1] + (ori_rank // dims[0] + step + 1) % dims[1]
                    chunk_index = chunk_id * chunk_size + channel_id * count
                    src = Chunk(rank, chunk_index, count)
                    dst = Chunk(next_rank, chunk_index, count)
                    copy_reduce(algo, src, dst, channel_id)
        # 1->0, 3->2, 5->4, 7->6 回传
        chunk_size = one_data_count
        this_chan = channels[0]
        assert chunk_size % this_chan == 0
        count = chunk_size // this_chan
        for index in range(1, ngpus, 2):
            for channel_id in range(this_chan):
                src_rank = index - 1
                dst_rank = index
                chunk_id = dst_rank % dims[0] * dims[1] + dst_rank // dims[0]
                chunk_index = chunk_id * chunk_size + channel_id * count
                src = Chunk(src_rank, chunk_index, count)
                dst = Chunk(dst_rank, chunk_index, count)
                copy_reduce(algo, src, dst, channel_id)

    
    return algo


