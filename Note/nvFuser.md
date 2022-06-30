# nvFuser

Fusion is: User Defined Operations $\rightarrow$Efficient Device Specific Code

support:

* Backward pass
* Bool, Int32, Int64, FP16, BFloat16, FP32, FP64
* Pointwise ops, reductions, normalizations, view
* Dynamic shapes
* Coming soon:
  * channel last, complex, transpose, pooling, matmul

Many DL approaches today focus on fusing existing kernels with “other things”Or mapping operators to pre-compiled fusions (think Conv-BN-Activation)
nvFuser builds kernels from the ground up, allowing it to target novel operators

## Experiment

JOC BERT

Resnet 50

EfficientNet-B4
