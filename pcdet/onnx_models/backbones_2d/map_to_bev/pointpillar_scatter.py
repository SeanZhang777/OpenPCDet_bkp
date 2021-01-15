import torch
import torch.nn as nn
from typing import List

class PointPillarScatter(nn.Module):
    __constants__ = ['nx', 'ny', 'nz']

    def __init__(self, model_cfg, grid_size):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        grid_x, grid_y, grid_z = grid_size
        self.nx = int(grid_x)
        self.ny = int(grid_y)
        self.nz = int(grid_z)
        assert self.nz == 1

    def forward(self, pillar_features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        batch_spatial_features = []
        batch_size = 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            #batch_mask = coords[:, 0] == batch_idx
            #this_coords = coords[batch_mask, :]
            this_coords = coords
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.long()
            #pillars = pillar_features[batch_mask, :]
            pillars = pillar_features
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)

        return batch_spatial_features
