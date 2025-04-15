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

    # NOTE: 没有考虑GPU自己的数据原本的位置, 假设GPU i的原本数据都在第0块, 每一块是one_data_count = nchunksperloop/ngpus
    if coll == AG:
        # 机间传输
        chunk_size = one_data_count
        count = chunk_size // channels[0]
        for step in range(dims[0]-1):
            for channel_id in range(channels[0]):
                for index in range(ngpus):
                    rank = index
                    next_rank = (index + dims[1]) % ngpus # dims[1]在AG里是8
                    chunk_src_index = step * chunk_size + channel_id * count
                    src = Chunk(rank, chunk_src_index, count)
                    chunk_dst_index = (step + 1) * chunk_size + channel_id * count
                    dst = Chunk(next_rank, chunk_dst_index, count)
                    copy(algo, src, dst, channel_id)
        # 机内传输
        chunk_size = one_data_count * dims[0]
        count = chunk_size // channels[1]
        for step in range(dims[1]-1):
            for channel_id in range(channels[1]):
                for index in range(ngpus):
                    rank = index
                    next_rank = (index + 1) % dims[1] + index // dims[1] * dims[1]
                    chunk_src_index = step * chunk_size + channel_id * count
                    src = Chunk(rank, chunk_src_index, count)
                    chunk_dst_index = (step + 1) * chunk_size + channel_id * count
                    dst = Chunk(next_rank, chunk_dst_index, count)
                    copy(algo, src, dst, channel_id)

    elif coll == RS:
        ## TODO AG和RS是相反的
        pass

    
    return algo