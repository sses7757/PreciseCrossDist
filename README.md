# PreciseCrossDist
A high-performance implementation of precise cross distance matrix computation using CUDA
$$
C = \text{dist}_p(A, B), \text{ i.e., } C_{ij} = \|A_i - B_j\|_p
$$

## Features
- Supports batch cross distance matrix computation
- Supports both forward (compute $C$ from $A$ and $B$) and backward (compute $\partial L / \partial A$ and $\partial L / \partial B$ from $\partial L / \partial C$) computation
- Achieves near machine-precision computation for both forward and backward pass