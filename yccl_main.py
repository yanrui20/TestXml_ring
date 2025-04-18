import math
from yccl import *

if __name__ == "__main__":
    dims = [8, 2]
    channels = [16, 8]
    coll = AG
    is_4gpus_out = True
    if not is_4gpus_out:
        root = heterogeneous_channel_ring(coll=coll, dims=dims, channels=channels)
    else:
        root = heterogeneous_channel_ring_only_4GPUs_inter_mechine(coll=coll, dims=dims, channels=channels)
    # root.show_xml()
    root.check()
    gpus = math.prod(dims)
    file = f"./yccl_{'AG' if coll == AG else "RS"}/" \
        f"{gpus}GPUs/ring_{"_".join([str(i) for i in dims])}" \
        f"_channel_{"_".join([str(i) for i in channels])}" \
        f"{'_4GPUs_out' if is_4gpus_out else ''}/yccl_test.xml"
    root.store(file)