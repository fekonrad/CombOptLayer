import torch
from torch import nn


class DijkstraGridSolver(nn.Module):
    """
    A special instance of a Dijkstra algorithm which operates on 2D lattices with weights represented as 2D tensors.
    """
    def __init__(self, diagonal=False):
        """
        :param diagonal: Boolean determining whether diagonal moves on the 2D grid are allowed,
                        i.e. a step from node (i, j) to node (i+1, j+1).
        """
        super().__init__()
        self.diagonal = diagonal

    def forward(self, weights):
        """
        :param weight_tens: torch.Tensor of shape (b, h, w), where h, w denotes the height and width of the grid.
        :return:
        """
        batch_size, size_of_floor, _ = weights.shape
        device = weights.device

        # Initialize distances, visited, and predecessor matrices
        distances = torch.full((batch_size, size_of_floor, size_of_floor), float('inf'), device=device)
        distances[:, 0, 0] = 0
        visited = torch.zeros((batch_size, size_of_floor, size_of_floor), dtype=torch.bool, device=device)
        predecessor = torch.zeros((batch_size, size_of_floor, size_of_floor, 2), dtype=torch.long, device=device)

        # Define directions
        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1],
                                   [-1, -1], [-1, 1], [1, -1], [1, 1]], device=device)

        for _ in range(size_of_floor * size_of_floor):
            # Find the unvisited node with the smallest distance
            unvisited_distances = torch.where(visited, float('inf'), distances)
            current_node = unvisited_distances.view(batch_size, -1).argmin(dim=1)
            current_y, current_x = current_node // size_of_floor, current_node % size_of_floor

            # If we've reached the target node in all batches, we can stop
            if (current_y == size_of_floor - 1) & (current_x == size_of_floor - 1):
                break

            # Mark the current node as visited
            visited[torch.arange(batch_size), current_y, current_x] = True

            # Generate all neighbors
            neighbors = directions.unsqueeze(0) + torch.stack([current_y, current_x], dim=1).unsqueeze(1)
            valid_neighbors = (neighbors >= 0) & (neighbors < size_of_floor)
            valid_neighbors = valid_neighbors.all(dim=2)

            # Update distances for valid neighbors
            for i in range(directions.shape[0]):
                mask = valid_neighbors[:, i]
                if not mask.any():
                    continue

                neighbor_y, neighbor_x = neighbors[mask, i, 0], neighbors[mask, i, 1]
                new_distances = distances[mask, current_y[mask], current_x[mask]] + weights[
                    mask, neighbor_y, neighbor_x]

                update_mask = (new_distances < distances[mask, neighbor_y, neighbor_x]) & (
                    ~visited[mask, neighbor_y, neighbor_x])
                distances[
                    mask & update_mask.unsqueeze(1).repeat(1, size_of_floor, size_of_floor), neighbor_y, neighbor_x] = \
                new_distances[update_mask]
                predecessor[mask & update_mask.unsqueeze(1).repeat(1, size_of_floor,
                                                                   size_of_floor), neighbor_y, neighbor_x, 0] = \
                current_y[mask][update_mask]
                predecessor[mask & update_mask.unsqueeze(1).repeat(1, size_of_floor,
                                                                   size_of_floor), neighbor_y, neighbor_x, 1] = \
                current_x[mask][update_mask]

        # Backtrack to construct paths
        path_matrices = torch.zeros_like(weights, dtype=torch.float32)
        current_y, current_x = torch.full((batch_size,), size_of_floor - 1, device=device), torch.full((batch_size,),
                                                                                                       size_of_floor - 1,
                                                                                                       device=device)
        path_matrices[torch.arange(batch_size), current_y, current_x] = 1

        while (current_y != 0).any() or (current_x != 0).any():
            prev_y, prev_x = predecessor[torch.arange(batch_size), current_y, current_x, 0], predecessor[
                torch.arange(batch_size), current_y, current_x, 1]
            path_matrices[torch.arange(batch_size), prev_y, prev_x] = 1
            current_y, current_x = prev_y, prev_x

        return path_matrices


if __name__ == "__main__":
    from solvers import solver_dijkstra
    import matplotlib.pyplot as plt

    weights = torch.rand((1, 16, 16))
    layer = DijkstraGridSolver(diagonal=True)
    sol1 = layer(weights)
    sol2 = solver_dijkstra(weights)
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(sol1.detach().numpy()[0])
    ax[1].imshow(sol2.detach().numpy()[0])
    plt.suptitle(f"sol1 == sol2: {torch.equal(sol1, sol2)}")
    plt.show()
