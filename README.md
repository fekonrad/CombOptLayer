# Combinatorial Optimization Layers
This repository contains an unofficial implementation of "Combinatorial Optimization Layers" in PyTorch, i.e. a neural network 
layer that allows a combinatorial optimization solver to be included into the forward propagation of the network, 
while still maintaining differentiability of the network for backpropagation. 

The implementation is based on the ideas presented in the following two papers: 
- ["Learning with Combinatorial Optimization Layers:
a Probabilistic Approach"](https://arxiv.org/pdf/2207.13513); Guillaume Dalle, Léo Baty, Louis Bouvier and Axel Parmentier
- ["Combinatorial Optimization enriched Machine Learning to solve
the Dynamic Vehicle Routing Problem with Time Windows"](https://arxiv.org/pdf/2304.00789); Léo Baty, Kai Jungel, Patrick S. Klein, Axel Parmentier and Maximilian Schiffer

Note that an implementation of Combinatorial Optimization Layers already exists in Julia, see [InferOpt.jl](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl).

# How To Use 
The main functionality of this repo is the `COptLayer`. It is an `nn.Module` which takes the following arguments at initialization: 
- `solver`: Method, representing the solver of a combinatorial optimization problen, that takes a as an input a `torch.Tensor` of shape `(b, *input_dim)` (where `input_dim` is the dimensionality of the argument of the solver)  and returns a `torch.Tensor` of shape `(b, *sol_dim)`. 
- `num_samples`: Number of samples taken in the Monte-Carlo estimation
- `smoothing`: ...

With given arguments `num_samples=n` and `smoothing=s`, the forward-method of the `COptLayer` returns 

$$\frac{1}{n}\sum_{i=1}^n f(\theta + s Z_i), \hspace{1cm} Z_i\sim \mathcal N(0, \mathbb I_p).$$

For more mathematical details about the forward- and backward-propagation of Combinatorial Optimization Layers, see ???

A few example usages can be found under the `Demo` directory. 

# Parallelized Solvers 
We provide a wrapper which, given a python function that solves one instance of a combinatorial optimization problem on a `torch.Tensor` of shape `input_dim`, returns a callable function that can solve multiple instances of the combinatorial optimization problem on batched inputs of shape `(batch_size, *input_dim)`. 
This is done via the numba library, which compiles the python function to a CUDA kernel.
For an example usage, see [Demo](#Demos).
# Demos 
## Shortest Paths on Warcraft Maps 
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fekonrad/CombOptLayer/blob/main/Demo/COptLayer_Warcraft_Demo.ipynb)

## Stochastic Vehicle Scheduling 
**TODO!**

## Dynamic Vehicle Routing Problem with Time Windows
**TODO!**