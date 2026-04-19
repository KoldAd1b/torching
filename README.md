# Torching

Torching is a small educational tensor library inspired by PyTorch. The goal is
to build the core pieces of a deep learning framework from scratch so the
mechanics are visible: tensor storage, device movement, operator overloading,
autograd graph construction, and reverse-mode backpropagation.

This is not intended to replace PyTorch. It is a learning project for
understanding how a PyTorch-like API can be assembled from lower-level NumPy and
CuPy operations.

## What We Are Building

The project is centered around two layers:

- `Array`: a thin wrapper over NumPy and CuPy arrays that handles device
  placement, dtype defaults, factory methods, and NumPy-style operations.
- `Tensor`: an autograd-aware wrapper around `Array` that tracks gradients,
  builds computation graphs, and defines backward passes for tensor operations.

Together, these pieces are meant to support code that feels similar to PyTorch:

```python
from torching import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * 2).sin()

y.backward()
print(x.grad)
```

## Current Capabilities

The current prototype includes:

- CPU/GPU-aware array storage through NumPy and CuPy
- dtype normalization with float32/int32 defaults
- PyTorch-like tensor representation
- `requires_grad` tracking
- basic computation graph construction
- reverse topological backward traversal
- gradient accumulation
- `no_grad` support
- common binary operations such as addition, subtraction, multiplication,
  division, floor division, and matrix multiplication
- common unary operations such as power, exponentiation, logarithm, absolute
  value, clamp, square root, sine, cosine, and tangent
- early support for shape and indexing operations, including reshape, flatten,
  transpose, permute, squeeze, unsqueeze, chunk, and repeat interleave

## Project Status

This repository is still in an early prototype state. The main design direction
is clear, but the package layout and runtime path still need cleanup before this
can be used like a normal Python package.

Known areas that need work:

- finalize the package structure and imports
- add missing module files such as dtype definitions
- make CuPy optional everywhere, so CPU-only use works cleanly
- finish and test graph-parent tracking
- add unit tests for forward operations and gradients
- add examples that compare behavior against PyTorch
- document supported operations and intentional limitations

## Design Goals

Torching should prioritize clarity over completeness. The implementation should
make it easy to understand why each operation has the backward rule it has, how
gradients accumulate, and how device-aware array operations are dispatched.

The library should eventually support:

- a small but coherent tensor API
- reliable scalar and array broadcasting behavior
- CPU-first execution with optional CUDA acceleration
- enough autograd functionality to train simple models
- test cases that validate gradients against PyTorch or finite differences

## Repository Layout

```text
.
├── tensor.py      # Tensor object and autograd operation definitions
└── to_array.py    # Array wrapper for NumPy/CuPy storage and device handling
```

## Development Notes

This project is intentionally close to the metal. When adding new operations,
each operation should define both its forward computation and its backward
function. For gradient-bearing tensors, the output tensor should remember its
parents so `backward()` can traverse the graph in reverse topological order.

As the project grows, the next practical step is to turn the current files into
a real package with tests. Once the package imports cleanly, the core gradient
rules can be validated operation by operation.
