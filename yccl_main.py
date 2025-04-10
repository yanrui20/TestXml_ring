import math
from yccl_lang import *

if __name__ == "__main__":
    dims = [8, 2]
    channels = [32, 16]
    # root = heterogeneous_channel_ring(coll=AG, dims=dims, channels=channels)
    root = heterogeneous_channel_ring_only_4GPUs_inter_mechine(coll=AG, dims=dims, channels=channels)
    # root.show_xml()
    root.check()
    gpus = math.prod(dims)
    file = f"./yccl_AG/{gpus}GPUs/ring_{"_".join([str(i) for i in dims])}" \
        f"_channel_{"_".join([str(i) for i in channels])}_4GPUs_out/yccl_test.xml"
    root.store(file)