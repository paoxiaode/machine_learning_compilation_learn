import sys
#sys.path.insert(0, '/home/tv/pytorch/pytorch/build/lib.linux-x86_64-3.9/')
import torch

# %matplotlib inline
from matplotlib import pyplot
import numpy
import time
from print_graph import make_graph
    
def origin_func(x): 
    x = x**2 
    x = x**3 
    return x 

def main():
    assert torch.cuda.is_available(), "Some examples need the GPU"
    print(f"the device is {torch.cuda.get_device_name()}")
    
    x = torch.rand(1, 2, 3, 4) 
    jit_model = torch.jit.script(origin_func) 
    print(jit_model.graph) 
    print(jit_model.code)

    pattern = """ 
        graph(%x): 
            %const_2 = prim::Constant[value=2]() 
            %out = aten::pow(%x, %const_2) 
            return (%out) 
    """ 
 
# 替换用的子图定义 
    replacement = """ 
        graph(%x): 
            %out = aten::mul(%x, %x) 
            return (%out) 
    """ 
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(pattern, replacement, 
                                                      jit_model.graph) 
 
    # 结果可视化，pow(x,2)被正确替换为mul(x,x)，pow(x,3)则保留原样不受影响。 
    print(jit_model.graph) 
    print(jit_model.code)

if __name__ == "__main__":
    main()