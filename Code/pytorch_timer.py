import sys
#sys.path.insert(0, '/home/tv/pytorch/pytorch/build/lib.linux-x86_64-3.9/')
import torch
from print_graph import make_graph
# %matplotlib inline
from matplotlib import pyplot
import numpy

def ratio_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    xi = torch.max(x1, x2)                                  # Intersection left
    yi = torch.max(y1, y2)                                  # Intersection top
    wi = torch.clamp(torch.min(x1+w1, x2+w2) - xi, min=0.)  # Intersection width
    hi = torch.clamp(torch.min(y1+h1, y2+h2) - yi, min=0.)  # Intersection height
    area_i = wi * hi                                        # Area Intersection
    area_u = w1 * h1 + w2 * h2 - wi * hi                    # Area Union
    return area_i / torch.clamp(area_u, min=1e-5)           # Intersection over Union

# we make a scripted function

def main():
    ratio_iou_scripted = torch.jit.script(ratio_iou)
    x1, y1, w1, h1, x2, y2, w2, h2 = torch.randn(8, 100, 1000, device='cuda').exp()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = ratio_iou(x1, y1, w1, h1, x2, y2, w2, h2)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
    start.record()
    _ = ratio_iou_scripted(x1, y1, w1, h1, x2, y2, w2, h2)
    end.record()
    torch.cuda.synchronize()
    
    print(start.elapsed_time(end))
    make_graph(ratio_iou_scripted.graph, "ratio_iou")
    make_graph(ratio_iou_scripted.graph_for(x1, y1, w1, h1, x2, y2, w2, h2), "ratio_iou_fuse")
    
    ######################### kernel fuser with script
    
    return 0

if __name__ == "__main__":
    main()
    