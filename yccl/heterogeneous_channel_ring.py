import math
from .lang import *

def heterogeneous_channel_ring(coll, dims, channels):
    nchannels = max(channels)
    ngpus = math.prod(dims) # 乘积
    # 每个gpu的一份数据被分成了多少个count
    one_data_count = math.lcm(*channels) # 最小公倍数
    nchunksperloop = ngpus * one_data_count

    if coll == AG:
        dims = dims[::-1]
        channels = channels[::-1]
    
    ## 初始化xml
    algo = init_algo(name="heterogeneous_channel", nchannels=nchannels, nchunksperloop=nchunksperloop, ngpus=ngpus, coll=coll)

    if coll == AG:
        ## 当前gpu所持有的数据块, gpu 0->0,1号数据块，gpu 8->0,1号数据块，gpu 1->2,3号数据块...
        ## init data deps
        for gpu in algo.children:
            gpu.set_dep((gpu.id % dims[1] * dims[0] + gpu.id // dims[1]) * one_data_count, one_data_count, (-1, -1))
        # 机间传输
        chunk_size = one_data_count
        assert chunk_size % channels[0] == 0
        count = chunk_size // channels[0]
        for step in range(dims[0]-1):
            for index in range(ngpus):
                for channel_id in range(channels[0]):
                    rank = index
                    next_rank = (index + dims[1]) % ngpus # dims[1]在AG里是8
                    chunk_id = rank % dims[1] * dims[0] + (rank // dims[1] - step) % dims[0]
                    chunk_src_index = chunk_id * chunk_size + channel_id * count
                    src = Chunk(rank, chunk_src_index, count)
                    dst = Chunk(next_rank, chunk_src_index, count)
                    copy(algo, src, dst, channel_id)
        # 机内传输
        chunk_size = one_data_count * dims[0]
        assert chunk_size % channels[1] == 0
        count = chunk_size // channels[1]
        for step in range(dims[1]-1):
            for channel_id in range(channels[1]):
                for index in range(ngpus):
                    rank = index
                    next_rank = (index + 1) % dims[1] + index // dims[1] * dims[1]
                    chunk_src_index = (rank - step) % dims[1] * chunk_size + channel_id * count
                    src = Chunk(rank, chunk_src_index, count)
                    dst = Chunk(next_rank, chunk_src_index, count)
                    copy(algo, src, dst, channel_id)

    elif coll == RS:
        ## 当前gpu所持有的数据块, gpu 0->0,1号数据块，gpu 8->0,1号数据块，gpu 1->2,3号数据块...
        ## 最终, gpu 0->0号数据块，gpu 8->1号数据块，gpu 1->2号数据块...
        ## init data deps
        for gpu in algo.children:
            pre_gpu_id = (gpu.id - 1) % dims[0]
            gpu.set_dep(pre_gpu_id * one_data_count * dims[1], one_data_count * dims[1], (-1, -1))
        # 机内dims[0]卡做rs，每个GPU持有dims[1]块的数据
        chunk_size = one_data_count * dims[1]
        assert chunk_size % channels[0] == 0
        count = chunk_size // channels[0]
        for step in range(dims[0]-1):
            for index in range(ngpus):
                for channel_id in range(channels[0]):
                    rank = index
                    next_rank = (rank + 1) % dims[0] + rank // dims[0] * dims[0]
                    chunk_gpu_id = (rank - 1 - step) % dims[0]
                    chunk_index = chunk_gpu_id * chunk_size + channel_id * count
                    src = Chunk(rank, chunk_index, count)
                    dst = Chunk(next_rank, chunk_index, count)
                    copy_reduce(algo, src, dst, channel_id)
        # 机外dims[1]卡做rs
        chunk_size = one_data_count
        assert chunk_size % channels[1] == 0
        count = chunk_size // channels[1]
        for step in range(dims[1]-1):
            for index in range(ngpus):
                for channel_id in range(channels[1]):
                    rank = index
                    next_rank = (rank + dims[0]) % ngpus
                    chunk_id = rank % dims[0] * dims[1] + (rank // dims[0] + step + 1) % dims[1]
                    chunk_index = chunk_id * chunk_size + channel_id * count
                    src = Chunk(rank, chunk_index, count)
                    dst = Chunk(next_rank, chunk_index, count)
                    copy_reduce(algo, src, dst, channel_id)
    
    return algo