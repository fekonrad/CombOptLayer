import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from COptLayer import COptLayer
from losses import PerturbedLoss
from solvers import solver_dijkstra


def valid_node(node, size_of_grid):
    """Checks if node is within the grid boundaries."""
    if node[0] < 0 or node[0] >= size_of_grid:
        return False
    if node[1] < 0 or node[1] >= size_of_grid:
        return False
    return True


def up(node):
    return (node[0] - 1, node[1])


def down(node):
    return (node[0] + 1, node[1])


def left(node):
    return (node[0], node[1] - 1)


def right(node):
    return (node[0], node[1] + 1)


def up_left(node):
    return (node[0] - 1, node[1] - 1)


def up_right(node):
    return (node[0] - 1, node[1] + 1)


def down_left(node):
    return (node[0] + 1, node[1] - 1)


def down_right(node):
    return (node[0] + 1, node[1] + 1)


def dijkstra(weights_tens) -> torch.Tensor:
    """
    :param weights: torch.tensor of shape (b, h, w)
    :return:
    """
    path_matrix_tens = torch.zeros_like(weights_tens)
    for b in range(weights_tens.shape[0]):
        initial_node = [0, 0]
        desired_node = [weights_tens.shape[1] - 1, weights_tens.shape[2] - 1]
        # initialize vertex weights
        weights = torch.clone(weights_tens[b])
        # source and destination are free
        weights[initial_node[0], initial_node[1]] = 0
        weights[desired_node[0], desired_node[1]] = 0

        # initialize maps for distances and visited nodes
        size_of_floor = weights.shape[1]

        # we only want to visit nodes once
        visited = np.zeros([size_of_floor, size_of_floor], bool)

        # store predecessor on current shortest path
        predecessor = np.zeros((size_of_floor, size_of_floor, 2), int)

        # initiate matrix to keep track of distance to source node
        # initial distance to nodes is infinity so we always get a lower actual distance
        distances = np.ones([size_of_floor, size_of_floor]) * np.inf
        # initial node has a distance of 0 to itself
        distances[initial_node[0], initial_node[1]] = 0

        # start algorithm
        current_node = [initial_node[0], initial_node[1]]

        while True:
            directions = [up, down, left, right, up_left, up_right, down_left, down_right]
            for direction in directions:
                potential_node = direction(current_node)
                if valid_node(potential_node, size_of_floor):  # boundary checking
                    if not visited[potential_node[0], potential_node[1]]:  # check if we have visited this node before
                        # update distance to node
                        distance = distances[current_node[0], current_node[1]] + weights[
                            potential_node[0], potential_node[1]]

                        # update distance if it is the shortest discovered
                        if distance < distances[potential_node[0], potential_node[1]]:
                            distances[potential_node[0], potential_node[1]] = distance
                            predecessor[potential_node[0], potential_node[1], 0] = current_node[0]
                            predecessor[potential_node[0], potential_node[1], 1] = current_node[1]

            # mark current node as visited
            visited[current_node[0], current_node[1]] = True

            # select next node by choosing the the shortest path so far
            t = distances.copy()
            # we don't want to visit nodes that have already been visited
            t[np.where(visited)] = np.inf
            # choose the shortest path
            node_index = np.argmin(t)

            # convert index to row,col.
            node_row = node_index // size_of_floor
            node_col = node_index % size_of_floor
            # update current node.
            current_node = (node_row, node_col)

            # stop if we have reached the desired node
            if current_node[0] == desired_node[0] and current_node[1] == desired_node[1]:
                break

        # backtrack to construct path
        path_matrix = torch.zeros(size_of_floor, size_of_floor)
        path_matrix[desired_node[0]][desired_node[1]] = 1
        current = desired_node
        while current != initial_node:
            path_matrix[predecessor[current[0], current[1], 0]][predecessor[current[0], current[1], 1]] = 1
            current = [predecessor[current[0], current[1], 0], predecessor[current[0], current[1], 1]]

        path_matrix_tens[b] = path_matrix
    return path_matrix_tens


class VertexWeightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # define architecture ...
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, padding='same')
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, padding='same')
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, padding='same')
        self.final_layer = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=2, padding='same')

    def forward(self, x):
        """
        :param img: torch.tensor of shape (b, c, h, w)      (in our case c=3, h=w=96)
        :return: torch.tensor of shape (b, c', h', w')      (in our case c=1, h=w=12)
        """
        x = self.conv1(x)             # (b, 16, 96, 96)
        x = nn.ReLU()(x)
        x = self.conv2(x)               # (b, 32, 96, 96)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)        # (b, 32, 48, 48)

        x = self.conv3(x)             # (b, 32, 48, 48)
        x = nn.ReLU()(x)
        x = self.conv4(x)               # (b, 32, 96, 96)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)        # (b, 32, 24, 24)

        x = self.conv5(x)             # (b, 32, 24, 24)
        x = nn.ReLU()(x)
        x = self.conv6(x)               # (b, 32, 24, 24)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)        # (b, 32, 12, 12)

        return nn.Softplus()(self.final_layer(x)).squeeze(1)


class ShortestPathModel(nn.Module):
    def __init__(self, num_samples=100, smoothing=0.8):
        super().__init__()
        self.vertexWeightModel = VertexWeightCNN()
        self.DijkstraLayer = COptLayer(solver=dijkstra,
                                       num_samples=num_samples,
                                       smoothing=smoothing)

    def forward(self, x):
        weights = nn.Softplus()(self.vertexWeightModel(x))   # require non-negative weights
        return self.DijkstraLayer(weights.squeeze(1))


def test_pertLoss():
    model = VertexWeightCNN()
    solver = dijkstra
    loss_fn = PerturbedLoss(dijkstra, objective='min', num_samples=10, smoothing=1.0)

    epochs = 500
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    maps = np.load("warcraft_maps/warcraft_shortest_path_oneskin/12x12/test_maps.npy")
    vertex_weights = np.load("warcraft_maps/warcraft_shortest_path_oneskin/12x12/test_vertex_weights.npy")
    shortest_paths = np.load("warcraft_maps/warcraft_shortest_path_oneskin/12x12/test_shortest_paths.npy")

    # train on one sample data point
    i = np.random.randint(low=0, high=1000)
    map = torch.tensor(maps[i], dtype=torch.float32).permute((2, 0, 1)).unsqueeze(0)
    vertex_weight = torch.tensor(vertex_weights[i], dtype=torch.float32)
    shortest_path = torch.tensor(shortest_paths[i], dtype=torch.float32)

    print(f"map.shape == {map.shape}")
    for _ in range(epochs):
        optimizer.zero_grad()
        vertex_weight_pred = model(map)
        path = solver(vertex_weight_pred.squeeze(1))
        loss = loss_fn(vertex_weight_pred, shortest_path)
        print(f"Loss after {_} Epochs: {loss.item()}")
        loss.backward()
        optimizer.step()

        if _ % 10 == 0:
            fig, ax = plt.subplots(ncols=2, nrows=2)
            ax[0, 0].imshow(vertex_weight.detach().numpy())
            ax[0, 1].imshow(model(map).detach().numpy()[0])
            ax[1, 0].imshow(shortest_path.detach().numpy())
            ax[1, 1].imshow(path[0].detach().numpy())
            plt.show()


if __name__ == "__main__":
    test_pertLoss()
    epochs = 500
    lr = 1e-2

    model = ShortestPathModel(num_samples=10,
                              smoothing=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    maps = np.load("warcraft_maps/warcraft_shortest_path_oneskin/12x12/test_maps.npy")
    vertex_weights = np.load("warcraft_maps/warcraft_shortest_path_oneskin/12x12/test_vertex_weights.npy")
    shortest_paths = np.load("warcraft_maps/warcraft_shortest_path_oneskin/12x12/test_shortest_paths.npy")

    # train on one sample data point
    i = np.random.randint(low=0, high=1000)
    map = torch.tensor(maps[i], dtype=torch.float32).permute((2, 0, 1)).unsqueeze(0)
    vertex_weight = torch.tensor(vertex_weights[i], dtype=torch.float32)
    shortest_path = torch.tensor(shortest_paths[i], dtype=torch.float32)

    print(f"map.shape == {map.shape}")
    for _ in range(epochs):
        optimizer.zero_grad()
        vertex_weight_pred = nn.Softplus()(model.vertexWeightModel(map))
        path = model.DijkstraLayer(vertex_weight_pred.squeeze(1))
        # we switch the two parts in our loss, since we have a minimization problem, cf. (6) in paper
        loss = ((shortest_path * vertex_weight_pred).sum(dim=0) - (path * vertex_weight_pred).sum(dim=0)).mean()
        print(f"Loss aftere {_} Epochs: {loss.item()}")
        loss.backward()
        optimizer.step()

        if _ % 10 == 0:
            fig, ax = plt.subplots(ncols=2, nrows=2)
            ax[0, 0].imshow(vertex_weight.detach().numpy())
            ax[0, 1].imshow(model.vertexWeightModel(map).detach().numpy()[0, 0])
            ax[1, 0].imshow(shortest_path.detach().numpy())
            ax[1, 1].imshow(path[0].detach().numpy())
            plt.show()

    """
        Problem: Our loss is given by theta^T (predicted path) - theta^T (true shortest path). 
                This loss is always non-negative. However, we can trivially minimize this loss by letting theta -> 0. 
                In the paper they suggest to instead use the regularized loss \mathcal L_\epsilon, cf. (7), 
                to prevent the "solution" theta=0.
                --> TODO: Implement this. 
    """