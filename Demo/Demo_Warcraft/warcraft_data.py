import numpy as np
import torch
from torch.utils.data import Dataset


class WarcraftPaths(Dataset):
    def __init__(self, map_path, cost_path, paths_path):
        super().__init__()
        self.maps = torch.tensor(np.load(map_path), dtype=torch.float32).permute(0, 3, 1, 2)
        self.costs = torch.tensor(np.load(cost_path), dtype=torch.float32)
        self.shortest_paths = torch.tensor(np.load(paths_path), dtype=torch.float32)

    def __len__(self):
        return self.maps.shape[0]

    def __getitem__(self, item):
        return self.maps[item], self.costs[item], self.shortest_paths[item]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    warcraftPaths = WarcraftPaths("warcraft_maps/warcraft_shortest_path_oneskin/12x12/test_maps.npy",
                                  "warcraft_maps/warcraft_shortest_path_oneskin/12x12/test_vertex_weights.npy",
                                  "warcraft_maps/warcraft_shortest_path_oneskin/12x12/test_shortest_paths.npy")
    dataloader = DataLoader(warcraftPaths, batch_size=16, shuffle=True)

    for map, weights, path in dataloader:
        fig, ax = plt.subplots(ncols=3)
        ax[0].imshow(map[0].permute(1, 2, 0).detach().numpy() / 255.0)
        ax[0].axis("off")
        ax[0].set_title("Map")
        ax[1].imshow(weights[0].detach().numpy())
        ax[1].axis("off")
        ax[1].set_title("Weights of Cells")
        ax[2].imshow(path[0].detach().numpy())
        ax[2].axis("off")
        ax[2].set_title("Shortest Path")
        plt.show()


