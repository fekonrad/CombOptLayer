from typing import Any
import string
import torch
from torch import nn


class COptFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                x: torch.Tensor,
                solver: Any,
                num_samples: int,
                smoothing: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.solver = solver
        ctx.num_samples = num_samples
        ctx.smoothing = smoothing

        sol = solver(x)
        ctx.out_dim = sol.shape
        return sol

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor, None, None, None]:
        x, = ctx.saved_tensors
        solver, num_samples, smoothing, out_dim = ctx.solver, ctx.num_samples, ctx.smoothing, ctx.out_dim
        grad = torch.zeros((*ctx.out_dim, *(x.shape[1:])), dtype=torch.float32, device=x.device)

        einsum_str = generate_einsum_string_batched(out_dim, x.shape)
        for _ in range(num_samples):
            z = torch.randn_like(x, dtype=torch.float32, device=x.device)
            solution = solver(x + smoothing * z)
            grad += torch.einsum(einsum_str, solution, z)

        grad_output_tensor = grad_outputs[0]
        grad_output_tensor = grad_output_tensor.unsqueeze(-1)

        return grad / (num_samples * smoothing) * grad_output_tensor, None, None, None


class COptLayer(nn.Module):
    def __init__(self, solver, num_samples=1, smoothing=1.0):
        """
            ...
        :param solver: Callable function implementing an (approximate) algorithm for a desired optimization problem
        :param num_samples: Integer amount of samples taken in Monte-Carlo gradient estimation
        :param smoothing: Float value determining the variance in the Monte-Carlo gradient estimation
        """
        super().__init__()
        self.solver = solver
        self.num_samples = num_samples
        self.smoothing = smoothing

    def forward(self, x):
        """
        :param x: torch.tensor, shape and values need to be compatible with `solver`
        :return: (torch.tensor) output of `solver` given input `x`
        """
        return COptFunction.apply(x,
                                  self.solver,
                                  self.num_samples,
                                  self.smoothing)


def generate_einsum_string_batched(a_shape, b_shape):
    """
        Generates the string for Einstein summation (torch.einsum) to compute ab^T as (*a.shape, *b.shape) tensor
    """
    a_dims = len(a_shape) - 1
    b_dims = len(b_shape) - 1
    all_letters = string.ascii_lowercase
    batch_letter = 'z'
    a_letters = all_letters[:a_dims]
    b_letters = all_letters[a_dims:a_dims + b_dims]
    input_str = f'{batch_letter}{a_letters},{batch_letter}{b_letters}'
    output_str = f'{batch_letter}{a_letters}{b_letters}'
    einsum_str = f'{input_str}->{output_str}'
    return einsum_str
