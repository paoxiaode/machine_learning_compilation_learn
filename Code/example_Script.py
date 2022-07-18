import sys
#sys.path.insert(0, '/home/tv/pytorch/pytorch/build/lib.linux-x86_64-3.9/')
import torch

# %matplotlib inline
from matplotlib import pyplot
import numpy


    
# @torch.jit.script
def fn(x):
    x = x + 1
    return x


def main():
    assert torch.cuda.is_available(), "Some examples need the GPU"
    print(f"the device is {torch.cuda.get_device_name()}")
    
    #################scipting
    print("-------------script---------------")
    scripted_fn = torch.jit.script(fn)
    print(type(scripted_fn))
    print(scripted_fn.graph)
    print(scripted_fn.code)
    
    # t = torch.tensor([1,2])
    # print(t.tolist)
    


    # x = torch.randn(1024, 128, device="cuda")
    # x /= x.norm(p=2, dim=1, keepdim=True).requires_grad_()
    # print(lunif(x))
    # return 0

if __name__ == "__main__":
    main()