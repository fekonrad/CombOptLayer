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
        params = theta + smoothing * z          # shape (n, b, h, w)
        p_shape = params.shape
        n, b = p_shape[0], p_shape[1]
        params = params.view(n*b, *p_shape[2:])         # shape (n*b, h, w)

        ctx.y, ctx.objective, ctx.solver, ctx.params = (
                y, objective, solver, params)               # store for backward function
        ctx.n, ctx.b, = n, b

        if objective == 'min':
            return (y.unsqueeze(0) * params.view(n, b, *p_shape[2:]) -
                    ctx.solver(params).view(n, b, *p_shape[2:]) * params.view(n, b, *p_shape[2:])).mean(dim=0).sum()

        return (ctx.solver(params).view(n, b, *p_shape[2:]) * params.view(n, b, *p_shape[2:]) -
                y.unsqueeze(0) * params.view(n, b, *p_shape[2:])).mean(dim=0).sum()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output_tensor = grad_outputs[0]
        grad_output_tensor = grad_output_tensor.unsqueeze(-1)
        params = ctx.params
        p_shape = params.shape
        n, b = ctx.n, ctx.b

        if ctx.objective == 'min':
            return (ctx.y - (ctx.solver(ctx.params).view(n, b, *p_shape[1:]).mean(dim=0))) * grad_output_tensor, None, None, None, None, None

        return (ctx.solver(ctx.params).view(n, b, *p_shape[1:]).mean(dim=0) - ctx.y) * grad_output_tensor, None, None, None, None, None


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
        self.objective = objective.lower()
        self.num_samples = num_samples
        self.smoothing = smoothing

        if not (self.objective == 'min' or self.objective == 'max'):
            raise KeyError("Keyword 'objective' either has to be 'min' (default) or 'max'.")

    def forward(self, theta, y):
        """
        :param theta: torch.tensor of shape (b, *input_dim); `solver` needs to be able to process tensors of such shape
        :param y: torch.tensor of shape (input_dim) representing (approximate) optimal solution.
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
        sol[i, ind.item()] = 1.0
    return sol


if __name__ == "__main__":
    # from solvers import solver_Simplex
    theta = torch.tensor([[1.0, 1.0]], dtype=torch.float32, requires_grad=True)
    solver = solver_Simplex

    loss_fn = PerturbedLoss(solver,
                            objective='max',
                            num_samples=10,
                            smoothing=1.0)

    y = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    loss = loss_fn(theta, y)
    print(f"Loss: {loss.item()}")
    loss.backward()
    print(f"Gradient: {theta.grad.detach().numpy()}")
    """
        I don't think this example works? 
        The gradient here is of the form [a, -a] for a>0. But shouldn't it be the other way around?
    """
