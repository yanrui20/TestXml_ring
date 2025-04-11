import math
from .lang import *

def heterogeneous_channel_ring_only_4GPUs_inter_mechine(coll, dims, channels):
    nchannels = max(channels)
    ngpus = math.prod(dims) # 乘积
    # 每个gpu的一份数据被分成了多少个count
    one_data_count = math.lcm(*channels) # 最小公倍数
    nchunksperloop = ngpus * one_data_count

    if coll == AG:
        dims = dims[::-1]
        channels = channels[::-1]
    
    ## 初始化xml
    algo = init_algo(name="heterogeneous_channel_ring_only_4GPU_inter_mechine", nchannels=nchannels, nchunksperloop=nchunksperloop, ngpus=ngpus, coll=coll)

    # NOTE: 没有考虑GPU自己的数据原本的位置, 假设GPU i的原本数据都在第0块, 每一块是one_data_count = nchunksperloop/ngpus
    if coll == AG:
        # 7->0,1->2,3->4,5->6
        chunk_size = one_data_count
        assert chunk_size % channels[1] == 0
        count = chunk_size // channels[1]
        for channel_id in range(channels[1]):
            for index in range(1, ngpus, 2):
                src_rank = index
                dst_rank = (index + 1) % dims[1] + index // dims[1] * dims[1]
                chunk_src_index = channel_id * count
                src = Chunk(src_rank, chunk_src_index, count)
                chunk_dst_index = dims[0] * chunk_size + channel_id * count ## 传输过去放在哪里
                dst = Chunk(dst_rank, chunk_dst_index, count)
                copy(algo, src, dst, channel_id)
        # 机间传输
        chunk_size = one_data_count
        assert chunk_size % (channels[0] // 2) == 0
        count = chunk_size // channels[0] * 2
        for step in range(dims[0]-1):
            for channel_id in range(channels[0] // 2):
                for index in range(ngpus):
                    src_rank = index if index % 2 == 0 else (index + 1) % dims[1] + index // dims[1] * dims[1]
                    dst_rank = (src_rank + dims[1]) % ngpus # dims[1]在AG里是8
                    _step = step if index % 2 == 0 else (step + dims[0])
                    chunk_src_index = _step * chunk_size + channel_id * count
                    src = Chunk(src_rank, chunk_src_index, count)
                    chunk_dst_index = (_step + 1) * chunk_size + channel_id * count
                    dst = Chunk(dst_rank, chunk_dst_index, count)
                    _channel_id = channel_id if index % 2 == 0 else channel_id + channels[0] // 2
                    copy(algo, src, dst, _channel_id)
        # # 7->0,1->2,3->4,5->6 反向回传
        # chunk_size = one_data_count * (dims[0]-1)
        # this_chan = 2
        # assert chunk_size % this_chan == 0
        # count = chunk_size // this_chan
        # for channel_id in range(this_chan):
        #     for index in range(1, ngpus, 2):
        #         src_rank = (index + 1) % dims[1] + index // dims[1] * dims[1]
        #         dst_rank = index
        #         chunk_src_index = one_data_count * (dims[0]+1) + channel_id * count
        #         src = Chunk(src_rank, chunk_src_index, count)
        #         chunk_dst_index = one_data_count + channel_id * count
        #         dst = Chunk(dst_rank, chunk_dst_index, count)
        #         copy(algo, src, dst, channel_id)
        # 机内传输
        chunk_size = one_data_count * dims[0]
        assert chunk_size % channels[1] == 0
        count = chunk_size // channels[1]
        for step in range(dims[1]-1):
            for channel_id in range(channels[1]):
                for index in range(ngpus):
                    if step == 0 and index % 2 == 1:
                        continue
                    rank = index
                    next_rank = (index + 1) % dims[1] + index // dims[1] * dims[1]
                    chunk_src_index = step * chunk_size + channel_id * count
                    src = Chunk(rank, chunk_src_index, count)
                    chunk_dst_index = (step + 1) * chunk_size + channel_id * count
                    dst = Chunk(next_rank, chunk_dst_index, count)
                    copy(algo, src, dst, channel_id)
        
        # 7->0,1->2,3->4,5->6 反向回传, 机内转完之后, 1有7的数据, 在第(2*dims[0]+1)*one_data_count的位置
        chunk_size = one_data_count * (dims[0]-1)
        assert chunk_size % channels[1] == 0
        count = chunk_size // channels[1]
        for channel_id in range(channels[1]):
            for index in range(1, ngpus, 2):
                src_rank = index
                dst_rank = (index - 2) % dims[1] + index // dims[1] * dims[1]
                chunk_src_index = one_data_count * (2*dims[0]+1) + channel_id * count
                src = Chunk(src_rank, chunk_src_index, count)
                chunk_dst_index = one_data_count + channel_id * count
                dst = Chunk(dst_rank, chunk_dst_index, count)
                copy(algo, src, dst, channel_id)

    elif coll == RS:
        ## TODO AG和RS是相反的
        pass

    
    return algo


