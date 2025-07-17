import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class TriPlaneSDFModel(nn.Module):
    def __init__(self, num_shapes, plane_feat_dim=32, plane_res=256,
                 num_layers=8, skip_connections=True, inner_dim=256, output_dim=1):
        super().__init__()

        self.plane_feat_dim = plane_feat_dim
        self.plane_res = plane_res
        self.num_layers = num_layers
        self.skip_connections = skip_connections

        # Each shape gets its own set of tri-planes
        self.xy_planes = nn.Parameter(torch.randn(num_shapes, plane_feat_dim, plane_res, plane_res) * 0.001)
        self.yz_planes = nn.Parameter(torch.randn(num_shapes, plane_feat_dim, plane_res, plane_res) * 0.001)
        self.zx_planes = nn.Parameter(torch.randn(num_shapes, plane_feat_dim, plane_res, plane_res) * 0.001)

        input_dim = 3 * plane_feat_dim
        self.skip_tensor_dim = copy.deepcopy(input_dim)

        layers = []
        for i in range(num_layers - 2):  # leave space for skip + final
            layers.append(nn.Sequential(
                nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)),
                nn.ReLU()))
            input_dim = inner_dim

        self.net = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(
            nn.Linear(inner_dim, output_dim),
            nn.Tanh()
        )

        # For the skip connection layer
        if self.skip_connections and num_layers >= 5:
            self.skip_layer = nn.Sequential(
                nn.Linear(inner_dim, inner_dim - self.skip_tensor_dim),
                nn.ReLU()
            )

    def sample_triplane_features(self, coords, xy_plane, yz_plane, zx_plane):
        """
        Sample tri-plane features for a single shape (used in inference).

        Args:
            coords: (B, 3)
            xy_plane, yz_plane, zx_plane: (1, C, H, W)

        Returns:
            features: (B, C)
        """
        B = coords.shape[0]

        # Normalize from [-1, 1] to [-1, 1] for grid_sample
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        xy_coords = torch.stack([x, y], dim=-1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 2)
        yz_coords = torch.stack([y, z], dim=-1).unsqueeze(1).unsqueeze(1)
        zx_coords = torch.stack([z, x], dim=-1).unsqueeze(1).unsqueeze(1)

        # Repeat planes to match B for grid_sample
        #xy_plane = xy_plane.expand(B, -1, -1, -1)  # (B, C, H, W)
        #yz_plane = yz_plane.expand(B, -1, -1, -1)
        #zx_plane = zx_plane.expand(B, -1, -1, -1)

        # Sample features
        xy_feats = F.grid_sample(xy_plane, xy_coords, align_corners=True).squeeze(-1).squeeze(-1)
        yz_feats = F.grid_sample(yz_plane, yz_coords, align_corners=True).squeeze(-1).squeeze(-1)
        zx_feats = F.grid_sample(zx_plane, zx_coords, align_corners=True).squeeze(-1).squeeze(-1)

        return torch.cat([xy_feats, yz_feats, zx_feats], dim=-1)  # (B, 3*C)

    def forward(self, coords, xy_plane=None, yz_plane=None, zx_plane=None, shape_ids=None):
        """
        Supports two modes:
        1. Training: passes shape_ids to lookup per-sample tri-planes (batch)
        2. Inference: passes explicit xy/yz/zx_plane (single shape)

        Args:
            coords: (B, 3)
            xy_plane, yz_plane, zx_plane: (1, C, H, W) tri-planes for one shape
            shape_ids: (B,) — used during training

        Returns:
            sdf: (B, 1)
        """
        if xy_plane is not None and yz_plane is not None and zx_plane is not None:
            # Inference mode — reconstructing one shape
            feats = self.sample_triplane_features(coords, xy_plane, yz_plane, zx_plane)
        elif shape_ids is not None:
            # Training mode — batch of shapes
            feats = self.sample_triplane_features(coords, self.xy_planes[shape_ids], self.yz_planes[shape_ids],
                                                  self.zx_planes[shape_ids])
        else:
            raise ValueError("Either (xy_plane, yz_plane, zx_plane) or shape_ids must be provided.")

        input_feats = feats.clone().detach()

        # Dynamic forward pass with skip connections
        if self.skip_connections and self.num_layers >= 5:
            for i in range(3):
                feats = self.net[i](feats)
            feats = self.skip_layer(feats)
            feats = torch.hstack((feats, input_feats))
            for i in range(3, self.num_layers - 2):
                feats = self.net[i](feats)
        else:
            if self.skip_connections:
                print("[Warning] At least 5 layers are required to enable skip connections.")
            feats = self.net(feats)

        sdf = self.final_layer(feats)
        return sdf

