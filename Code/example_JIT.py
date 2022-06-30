import sys
#sys.path.insert(0, '/home/tv/pytorch/pytorch/build/lib.linux-x86_64-3.9/')
import torch

# %matplotlib inline
from matplotlib import pyplot
import numpy

def lunif(x, t=2): # copied from the paper
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()
    

def fn(x):
    for _ in range(x.dim()):
        x = x * x
    return x

def foo(x, y):
    return 2 * x + y

def main():
    assert torch.cuda.is_available(), "Some examples need the GPU"
    print(f"the device is {torch.cuda.get_device_name()}")
    ###########################
    traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))
    print(traced_foo.graph)
    print(traced_foo.code)
    #################################
    traced_fn = torch.jit.trace(fn, torch.rand(3))
    print(traced_fn.graph)
    print(traced_fn.code)
    
    ##################tracing
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    input = torch.rand(8,1)
    print(model(input))
    trace_model = torch.jit.trace(model, input)
    print("-------------tracing---------------")
    print(model)
    print(trace_model)
    print(trace_model.graph)
    print(trace_model.code)

    
    #################scipting
    print("-------------script---------------")

    scripted_model = torch.jit.script(model)
    print(scripted_model.code)

    scripted_fn = torch.jit.script(fn)
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