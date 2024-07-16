import torch
from COptLayer import COptLayer
from solvers import solver_Simplex, solver_dijkstra


"""
We perform the following tests for the COptLayer: 
    - If we choose theta = [1.0, 1.0], then f(theta + Z) Z^T should just equal [[Z_1, Z_2], [0, 0]] if Z_1>Z_2 and 
        [[0, 0], [Z_1, Z_2]] if Z_1<Z_2. Both events happen with probability 0.5. 
        Therefore the expected value of theta^T (f(theta + Z) Z^T) should be all 0 (?)
"""
def test_gradient():
    theta = torch.tensor([[1.0, 1.0]], dtype=torch.float32, requires_grad=True)
    layer = COptLayer(solver=solver_Simplex,
                      num_samples=10000,
                      smoothing=1.0)
    sol = layer(theta)
    loss = (theta * sol).sum()
    loss.backward()
    print(f"theta.shape == {theta.shape}")
    print(f"theta.grad == {theta.grad}")
    if torch.norm(theta.grad[0, 0] - theta.grad[0, 1]) > 1e-3:
        raise Exception(f"Entries of Gradient w.r.t. theta=[1.0, 1.0] are not equal. \n theta.grad == {theta.grad.detach().numpy()}")


if __name__ == "__main__":
    test_gradient()
    print("Passed all tests without Errors.")
