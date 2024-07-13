from typing import Any
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

        return solver(x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor, None]:
        x, = ctx.saved_tensors
        solver, num_samples, smoothing = ctx.solver, ctx.num_samples, ctx.smoothing
        # TODO: Make `solver` have the attribute `solver.out_dim` --> Otherwise we need to call `solver(x)` to get out_dim
        grad = torch.zeros((solver.out_dim, x.shape), dtype=torch.float32, device=x.device)
        for _ in range(num_samples):
            z = torch.randn_like(x, dtype=torch.float32, device=x.device)
            solution = solver(x + smoothing * z)
            grad += torch.outer(solution, z)

        return grad / smoothing * grad_outputs


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
                                  solver=self.solver,
                                  num_samples=self.num_samples,
                                  smoothing=self.smoothing)
