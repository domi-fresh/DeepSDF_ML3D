import torch
import os
import numpy as np
import trimesh
import yaml

from model.model_triplane import TriPlaneSDFModel
from utils.utils_deepsdf import get_volume_coords, predict_sdf, extract_mesh
import matplotlib.pyplot as plt
import config_files
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_and_save_triplane_features(model, shape_idx, out_dir="vis_triplanes", max_channels=4):
    """
    Visualize and save feature maps from XY, YZ, ZX planes for a specific shape index.
    Each plane has shape: [C, H, W] — we visualize up to `max_channels` channels per plane
    in a grid layout (as square as possible).
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        xy = model.xy_planes[shape_idx].cpu()  # [C, H, W]
        yz = model.yz_planes[shape_idx].cpu()
        zx = model.zx_planes[shape_idx].cpu()

        for name, plane in zip(["xy", "yz", "zx"], [xy, yz, zx]):
            num_channels = min(max_channels, plane.shape[0])

            # Compute grid shape (rows x cols) as square as possible
            n_cols = math.ceil(math.sqrt(num_channels))
            n_rows = math.ceil(num_channels / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = axes.flatten() if num_channels > 1 else [axes]

            for i in range(num_channels):
                ax = axes[i]
                ax.imshow(plane[i], cmap="viridis")
                ax.axis("off")
                ax.set_title(f"{name.upper()} ch{i}")

            # Hide any unused subplots
            for i in range(num_channels, len(axes)):
                axes[i].axis("off")

            plt.tight_layout()
            save_path = os.path.join(out_dir, f"{name}_features.png")
            plt.savefig(save_path)
            print(f"[✓] Saved {name.upper()} to {save_path}")
            plt.close()


def reconstruct_shape(shape_id, model, resolution, save_path):
    coords, grid_size = get_volume_coords(resolution)
    coords = coords.to(device)

    # Split coordinates to avoid out-of-memory
    coords_batches = torch.split(coords, 100000)

    sdf_preds = []

    # Extract tri-planes for the current shape (add batch dim)
    xy_plane = model.xy_planes[shape_id].unsqueeze(0)
    yz_plane = model.yz_planes[shape_id].unsqueeze(0)
    zx_plane = model.zx_planes[shape_id].unsqueeze(0)

    # Evaluate in inference mode (no gradients)
    model.eval()
    with torch.no_grad():
        for coords_batch in coords_batches:
            feats = model.sample_triplane_features(coords_batch, xy_plane, yz_plane, zx_plane)

            # Forward through full model (including skip logic and final layer)
            #sdf_batch, _ = model.forward(coords_batch, xy_plane=xy_plane, yz_plane=yz_plane, zx_plane=zx_plane)
            sdf_batch = model.forward(coords_batch, xy_plane=xy_plane, yz_plane=yz_plane, zx_plane=zx_plane)
            sdf_preds.append(sdf_batch)

    sdf = torch.cat(sdf_preds, dim=0)

    print(f"SDF min: {sdf.min().item():.5f}, max: {sdf.max().item():.5f}")

    vertices, faces = extract_mesh(grid_size, sdf)

    # Save mesh
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(save_path)
    print(f"Saved: {save_path}")


def main(cfg):
    # Load training config
    settings_path = os.path.join("results", "runs_sdf", cfg["folder_sdf"], "settings.yaml")
    with open(settings_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    num_shapes = len(np.load(os.path.join("results", f"samples_dict_{train_cfg['dataset']}.npy"), allow_pickle=True).item())

    # Load model
    model = TriPlaneSDFModel(
        num_shapes=num_shapes,
        plane_feat_dim=train_cfg['plane_feat_dim'],
        plane_res=train_cfg['plane_res'],
        num_layers=train_cfg['num_layers'],
        skip_connections=train_cfg['skip_connections'],
        inner_dim=train_cfg['inner_dim'], 
        mode=train_cfg['model_mode']
    ).float().to(device)

    weights_path = os.path.join("results", "runs_sdf", cfg["folder_sdf"], "weights.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load ID mapping
    str2int_dict = np.load(os.path.join("results", "idx_str2int_dict.npy"), allow_pickle=True).item()

    for obj_id in cfg["obj_ids"]:
        shape_idx = str2int_dict[obj_id]
        save_path = os.path.join("results", "runs_sdf", cfg["folder_sdf"], "meshes_training", f"mesh_{shape_idx}.obj")
        vis_path = os.path.join("results", "runs_sdf", cfg["folder_sdf"], "meshes_training", f"mesh_{shape_idx}_vis_triplanes")
        visualize_and_save_triplane_features(model=model, shape_idx=shape_idx, out_dir= vis_path, max_channels=64)
        
        reconstruct_shape(shape_idx, model, cfg["resolution"], save_path)

        

if __name__ == "__main__":
    cfg_path = os.path.join('config_files', 'reconstruct_triplane.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)
