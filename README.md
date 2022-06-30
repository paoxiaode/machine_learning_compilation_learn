# Pytorch Script

TorchScript is a language implemented by the PyTorch JIT ("Just in Time compiler")

Reference：

[Lernapparat - Machine Learning](https://lernapparat.de/jit-optimization-intro/)

## Difference between TorchScript and Python

**Typed vs non-typed**

One important difference between TorchScript and Python is that in TorchScript everything is typed. Important types are

* `bool`, `int`, `long`, `double` for numbers (int = 32 bit integer, long = 64 bit integer)
* `Tensor` for tensors (of arbitrary shape, dtype, ...)
* `List[T]` a list with elements of type T (one of the above)
* Tuples are of fixed size with arbitrary but fixed element type, so e.g. `Tuple(Tensor, int)`.
* `Optional[T]` for things that can be `None`

**early binding vs late binding**

Binding refers to the process of converting identifiers (such as variable and performance names) into addresses.

In python, binding takes place in runtime(late binding).

In torchScript binding takes place when compile(early binding),

[Early binding and Late binding in C++ - GeeksforGeeks](https://www.geeksforgeeks.org/early-binding-late-binding-c/)


# JIT workflow(high level)

* tracing to graph
* Then there are a number of compiler passes through the graph to go from `.graph` to an optimized graph (that can be retrieved with `.graph_for(*inputs)`.
* Finally, the `.graph` is compiled to a from of bytecode that is then executed by a virtual machine. We might hope to not meet the bytecode too often, but clearly we want this part to be fast, too. This maintains the operands on a stack and then dispatches to the various operators registered by LibTorch or the *custom operators* that extend the JIT.


## Optimization passes

* Eliminating dead code and common subexpressions, pre-computing things that only involve constants,
  * dead code is code which can never be executed at run-time
* Pooling redundant constants into single values, and some simple "pattern matching" optimizations (like eliminating `.t().t()`),
* Unrolling small loops and batching matrix multiplications that result from unrolling loops.


## Optimization on python

### How PyTorch programs spend their time

At a very high level, you can divide time spent into these parts:

* Python program flow,
* Data "administrative overhead" (creating `Tensor` data structures, autograd `Node`s etc.),
* Data aquisition (I/O),
* Computation roughly as
  * fixed overhead (kernel launches etc.),
  * reading / writing memory,
  * "real computation".
