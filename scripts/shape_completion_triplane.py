import os
import torch
import numpy as np
import yaml
import trimesh
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from model.model_triplane import TriPlaneSDFModel
from utils.utils_deepsdf import get_volume_coords, extract_mesh
from utils import utils_mesh
import data.ShapeNetCoreV2 as ShapeNetCoreV2
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_params(cfg):
    settings_path = os.path.join("results", "runs_sdf", cfg["folder_sdf"], "settings.yaml")
    with open(settings_path, "r") as f:
        return yaml.safe_load(f)


def generate_partial_pointcloud(cfg):
    obj_path = os.path.join('data', 'ShapeNetCoreV2', cfg['obj_ids'], 'models', 'model_normalized.obj')
    mesh = utils_mesh._as_mesh(trimesh.load(obj_path))
    mesh = utils_mesh.shapenet_rotate(mesh)

    samples = np.array(trimesh.sample.sample_surface(mesh, 10000)[0])
    t = [cfg['x_axis_ratio_bbox'], cfg['y_axis_ratio_bbox'], cfg['z_axis_ratio_bbox']]
    v_min, v_max = mesh.bounds
    for i in range(3):
        t_max = v_min[i] + t[i] * (v_max[i] - v_min[i])
        samples = samples[samples[:, i] < t_max]
    return samples


def infer_latent_code(model, pointcloud, sdf_gt, cfg, writer):
    latent = torch.zeros(3 * model.plane_feat_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([latent], lr=cfg['lr'])

    if cfg.get('lr_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg['lr_multiplier'],
            patience=cfg['patience'], threshold=0.0001, threshold_mode='rel'
        )

    for step in range(cfg['epochs']):
        optimizer.zero_grad()

        B = pointcloud.shape[0]
        shape_ids = torch.zeros(B, dtype=torch.long, device=device)

        # Fake tri-planes from latent
        latent_planes = latent.view(3, model.plane_feat_dim, 1, 1)
        xy, yz, zx = latent_planes[0].expand(1, -1, model.plane_res, model.plane_res), \
                     latent_planes[1].expand(1, -1, model.plane_res, model.plane_res), \
                     latent_planes[2].expand(1, -1, model.plane_res, model.plane_res)


        #feats = model.sample_triplane_features(pointcloud, xy, yz, zx)
        #sdf_pred = model.decoder(feats)

        #loss = torch.nn.functional.mse_loss(sdf_pred, sdf_gt)

        batch_size = 1024  # or lower, depending on memory
        loss_total = 0.0

        for i in range(0, pointcloud.shape[0], batch_size):
            coords_batch = pointcloud[i:i + batch_size]
            sdf_batch = sdf_gt[i:i + batch_size]

            feats = model.sample_triplane_features(coords_batch, xy, yz, zx)
            sdf_pred = model.decoder(feats)
            loss = F.mse_loss(sdf_pred, sdf_batch)

            loss_total += loss

        loss = loss_total / (pointcloud.shape[0] // batch_size)

        loss.backward()
        optimizer.step()
        if cfg.get('lr_scheduler', False):
            scheduler.step(loss)

        if step % 10 == 0:
            writer.add_scalar("LatentOpt/Loss", loss.item(), step)

    return latent.detach()


def reconstruct_from_latent(model, latent, resolution, save_path):
    coords, grid_shape = get_volume_coords(resolution)
    coords = coords.to(device)
    sdf_batches = []

    latent_planes = latent.view(3, model.plane_feat_dim, 1, 1)
    xy = latent_planes[0].expand(1, -1, model.plane_res, model.plane_res)
    yz = latent_planes[1].expand(1, -1, model.plane_res, model.plane_res)
    zx = latent_planes[2].expand(1, -1, model.plane_res, model.plane_res)

    with torch.no_grad():
        for coords_batch in torch.split(coords, 25000):
            B = coords_batch.size(0)
            #shape_ids = torch.zeros(B, dtype=torch.long, device=device)
            feats = model.sample_triplane_features(coords_batch, xy, yz, zx)
            sdf_batch = model.decoder(feats)
            sdf_batches.append(sdf_batch)

    sdf = torch.cat(sdf_batches, dim=0)
    verts, faces = extract_mesh(grid_shape, sdf)
    mesh = trimesh.Trimesh(verts, faces)
    mesh.export(save_path)
    print(f"[✓] Mesh saved to: {save_path}")


def main(cfg):
    train_cfg = read_params(cfg)
    run_dir = os.path.join("results", "runs_sdf", cfg["folder_sdf"])
    out_dir = os.path.join(run_dir, f"infer_triplane_{datetime.now().strftime('%m_%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=out_dir)

    # Load model
    num_shapes = len(np.load(os.path.join("results", f"samples_dict_{train_cfg['dataset']}.npy"), allow_pickle=True).item())
    model = TriPlaneSDFModel(
        num_shapes=num_shapes,
        plane_feat_dim=train_cfg["plane_feat_dim"],
        plane_res=train_cfg["plane_res"],
        decoder_hidden_dim=train_cfg["decoder_hidden_dim"]
    ).to(device)

    weights_path = os.path.join(run_dir, "weights.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Generate partial point cloud
    pointcloud_np = generate_partial_pointcloud(cfg)
    np.save(os.path.join(out_dir, 'partial_pointcloud.npy'), pointcloud_np)

    pointcloud = torch.tensor(pointcloud_np, dtype=torch.float32, device=device)
    sdf_gt = torch.zeros_like(pointcloud[:, 0]).view(-1, 1).to(device)

    # Infer best latent code
    latent = infer_latent_code(model, pointcloud, sdf_gt, cfg, writer)

    # Reconstruct full mesh
    out_path = os.path.join(out_dir, "output_mesh.obj")
    reconstruct_from_latent(model, latent, cfg["resolution"], out_path)


if __name__ == "__main__":
    cfg_path = os.path.join("config_files", "shape_completion.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
