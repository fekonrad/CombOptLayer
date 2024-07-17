from typing import Any

import torch
from torch import nn


class PerturbedLossFct(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                theta: torch.tensor,
                y: torch.tensor,
                objective: str,
                solver: Any,
                num_samples: int,
                smoothing: float
                ) -> Any:
        z = torch.rand((num_samples, *theta.shape))
        params = theta + smoothing * z

        ctx.y, ctx.objective, ctx.solver, ctx.params = (
                y, objective, solver, params)               # store for backward function

        if objective == 'min':
            return (torch.dot(y.flatten(), theta.flatten()) -
                    torch.bmm(ctx.solver(params).view(num_samples, 1, -1), params.view(num_samples, -1, 1)).mean())

        return (torch.bmm(ctx.solver(params).view(num_samples, 1, -1), params.view(num_samples, -1, 1)).mean() -
                torch.dot(y.flatten(), theta.flatten()))

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output_tensor = grad_outputs[0]
        grad_output_tensor = grad_output_tensor.unsqueeze(-1)

        if ctx.objective == 'min':
            return (ctx.y - (ctx.solver(ctx.params).mean(dim=0))) * grad_output_tensor, None, None, None, None, None
        return (ctx.solver(ctx.params).mean(dim=0) - ctx.y) * grad_output_tensor, None, None, None, None, None


class PerturbedLoss(nn.Module):
    def __init__(self, solver, objective='min', num_samples=1, smoothing=1.0):
        """
        :param solver: Method that implements a combinatorial optimization algorithm;
                        Needs to be able to take a torch.tensor of shape (b, *input_dim) as input and return
                        an output of shape (b, *out_dim).
        :param objective: String, either 'min' or 'max' depending on whether 'solver' minimizes or maximizes the objective
        :param num_samples: Integer determining the number of random samples drawn in loss forward and backward
        :param smoothing: Float determining the standard deviation of the sampled noise.
        """
        super().__init__()
        self.solver = solver
        self.objective = objective
        self.num_samples = num_samples
        self.smoothing = smoothing

        if not (objective == 'min' or objective == 'max'):
            raise KeyError("Keyword 'objective' either has to be 'min' (default) or 'max'.")

    def forward(self, theta, y):
        """
        :param theta:
        :param y:
        :return:
        """
        return PerturbedLossFct.apply(theta,
                                      y,
                                      self.objective,
                                      self.solver,
                                      self.num_samples,
                                      self.smoothing)


def solver_Simplex(theta: torch.tensor):
    max_ind = torch.argmax(theta, dim=1)
    sol = torch.zeros_like(theta)
    for i, ind in enumerate(max_ind):
        sol[i, ind[0]] = 1.0
    return sol


if __name__ == "__main__":
    # from solvers import solver_Simplex
    theta = torch.tensor([[1.0, 1.0]], dtype=torch.float32, requires_grad=True)
    solver = solver_Simplex

    loss_fn = PerturbedLoss(solver,
                            objective='min',
                            num_samples=100,
                            smoothing=1.0)

    y = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    loss = loss_fn(theta, y)
    print(f"Loss: {loss.item()}")
    loss.backward()
    print(f"Gradient: {theta.grad}")