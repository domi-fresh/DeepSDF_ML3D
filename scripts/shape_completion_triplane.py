import torch
import os
import numpy as np
import yaml
import trimesh
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from model.model_triplane import TriPlaneSDFModel
from utils.utils_deepsdf import get_volume_coords, extract_mesh
from utils.utils_mesh import _as_mesh, shapenet_rotate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_partial_pointcloud(obj_path, num_points=10000, ratios=(1.0, 1.0, 0.5)):
    mesh_original = _as_mesh(trimesh.load(obj_path))
    mesh = shapenet_rotate(mesh_original)

    samples = np.array(trimesh.sample.sample_surface(mesh, num_points)[0])
    v_min, v_max = mesh.bounds

    for i in range(3):
        t_max = v_min[i] + ratios[i] * (v_max[i] - v_min[i])
        samples = samples[samples[:, i] < t_max]

    return samples


def optimize_triplanes(model, partial_pc, cfg):
    B = partial_pc.shape[0]
    pc_tensor = torch.tensor(partial_pc, dtype=torch.float32).to(device)
    sdf_gt = torch.zeros((B, 1), dtype=torch.float32).to(device)

    xy = torch.randn(1, cfg['plane_feat_dim'], cfg['plane_res'], cfg['plane_res'], device=device, requires_grad=True)
    yz = torch.randn_like(xy)
    zx = torch.randn_like(xy)

    optimizer = torch.optim.Adam([xy, yz, zx], lr=cfg['lr'])
    loss_fn = torch.nn.MSELoss()

    for step in range(cfg['opt_steps']):
        optimizer.zero_grad()
        sdf_pred, _ = model(coords=pc_tensor, xy_plane=xy, yz_plane=yz, zx_plane=zx)
        loss = loss_fn(sdf_pred, sdf_gt)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:04d} | Loss: {loss.item():.6f}")

    return xy, yz, zx


def main(cfg):
    output_dir = os.path.join('results', 'runs_triplane', 'infer_latent_' + datetime.now().strftime('%m_%d_%H%M'))
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=output_dir)

    model = TriPlaneSDFModel(
        num_shapes=1,
        plane_feat_dim=cfg['plane_feat_dim'],
        plane_res=cfg['plane_res'],
        num_layers=cfg['num_layers'],
        skip_connections=True,
        inner_dim=cfg['inner_dim'],
        output_dim=1
    ).to(device)

    model.eval()

    coords, grad_size_axis = get_volume_coords(cfg['resolution'])
    coords_batches = torch.split(coords.to(device), 100000)

    obj_path = os.path.join('data', 'ShapeNetCoreV2', cfg['obj_id'], 'models', 'model_normalized.obj')
    partial_pc = load_partial_pointcloud(obj_path, ratios=(cfg['x_ratio'], cfg['y_ratio'], cfg['z_ratio']))
    np.save(os.path.join(output_dir, 'partial_pointcloud.npy'), partial_pc)

    xy, yz, zx = optimize_triplanes(model, partial_pc, cfg)

    sdf = []
    with torch.no_grad():
        for batch in coords_batches:
            sdf_batch, _ = model(batch, xy_plane=xy, yz_plane=yz, zx_plane=zx)
            sdf.append(sdf_batch.squeeze(-1).cpu().numpy())

    sdf = np.concatenate(sdf, axis=0)
    verts, faces = extract_mesh(grad_size_axis, sdf)
    mesh = trimesh.Trimesh(verts, faces)
    mesh.export(os.path.join(output_dir, 'output_mesh.obj'))


if __name__ == '__main__':
    with open(os.path.join('config_files', 'shape_completion_triplane.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
