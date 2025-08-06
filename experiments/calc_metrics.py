from metrics import compute_trimesh_chamfer, compute_mesh_accuracy, compute_trimesh_emd

import torch
import os
import model.model_sdf as sdf_model
#import model.model_siren as sdf_model
from utils import utils_deepsdf
import trimesh
import numpy as np
import yaml
from utils import utils_mesh
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model.clip_model import ClipObjEmbedder
import json
from pathlib import Path
from glob import glob
"""Compute all metrics"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = str(Path(__file__).parent.parent)

def get_test_split(cfg):
    """Get paths to all objects used for benchmarking"""

    shapenetcore_path = "/cluster/54/bkasper/ShapeNetCore"
    obj_paths = []
    allowed_categories = cfg['category_ids']  # specific categories

    for category in allowed_categories:
        pattern = os.path.join(shapenetcore_path, category, '*', 'models', '*.obj')
        category_paths = sorted(glob(pattern))[600:(600+cfg['benchmark_samples'])]  # sort for consistency, then take test split
        obj_paths.extend(category_paths)

    return obj_paths


def read_params(cfg):
    """Read the settings from the settings.yaml file. These are the settings used during training."""
    training_settings_path = os.path.join(PROJECT_ROOT, "results/runs_sdf",  cfg['folder_sdf'], 'settings.yaml') 
    with open(training_settings_path, 'rb') as f:
        training_settings = yaml.load(f, Loader=yaml.FullLoader)

    return training_settings


def generate_pointcloud(cfg, obj_path):
    """Load mesh and generate point cloud. For partial pc the ratio of the visible bounding box is defined in the config file.
    Args:
        cfg: config file
        obj_path: str: Path to object to generate pointcloud from
    Return:
        samples_full: np.array, shape (N, 3), where N is the number of points in the full cloud.
        samples_partial: np.array, shape (N, 3), where N is the number of points in the point cloud.
        """
    # Load mesh
    mesh = utils_mesh._as_mesh(trimesh.load(obj_path))

    # Sample on the object surface
    samples_full = np.array(trimesh.sample.sample_surface(mesh, 20000)[0]) # only sampling on surface

    # Infer object bounding box and collect samples on the surface of the objects when the x-axis is lower than a certain threshold t.
    # This is to simulate a partial point cloud.
    t = [cfg['x_axis_ratio_bbox'], cfg['y_axis_ratio_bbox'], cfg['z_axis_ratio_bbox']]

    v_min, v_max = mesh.bounds

    for i in range(3):
        t_max = v_min[i] + t[i] * (v_max[i] - v_min[i])
        samples_partial = samples_full[samples_full[:, i] < t_max]
    
    return samples_full, samples_partial


def main(cfg):
    model_settings = read_params(cfg)

    # Set directory and paths
    model_dir = os.path.join(PROJECT_ROOT, "results/runs_sdf", cfg['folder_sdf'])

    # Directory to save reconstructed objects to
    inference_dir = os.path.join(PROJECT_ROOT, "experiments/" f"benchmark_{datetime.now().strftime('%d_%m_%H%M%S')}")
    if not os.path.exists(inference_dir):
        os.mkdir(inference_dir)

    # Set tensorboard writer
    writer = SummaryWriter(log_dir=inference_dir, filename_suffix='inference_tensorboard')

    # Load the model
    weights = os.path.join(model_dir, 'weights.pt')

    model = sdf_model.SDFModel(
        num_layers=model_settings['num_layers'], 
        skip_connections=model_settings['latent_size'], 
        latent_size=model_settings['latent_size'], 
        inner_dim=model_settings['inner_dim']).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))

    # Path and json to save results to
    results_dict_path = os.path.join(PROJECT_ROOT, f"experiments/results_{datetime.now().strftime('%d_%m_%H%M%S')}.json")
    results_dict = {
        **{category : {
            "processed_samples": 0,
            "chamfer": [],
            "emd": [],
            "mesh_acc": []
        } for category in cfg['category_ids']}
    }
   
    # Define coordinates for mesh extraction
    coords, grad_size_axis = utils_deepsdf.get_volume_coords(cfg['resolution'])
    coords = coords.to(device)

    # Split coords into batches because of memory limitations
    coords_batches = torch.split(coords, 100000)

    # get all object paths of test split
    test_split = get_test_split(cfg)

    # iterate through test split
    for i, obj_path in enumerate(test_split):

        # get obj_category_id and obj_shape_id
        obj_category_id = obj_path.split("/")[-4]
        obj_shape_id = obj_path.split("/")[-3]

        print(f"Processing {i+1}-th test sample {obj_category_id}/{obj_shape_id}")
        
        # Generate partial point cloud and save gt_pointcloud for metrics
        gt_pointcloud, pointcloud = generate_pointcloud(cfg, obj_path)
        gt_pointcloud = trimesh.points.PointCloud(gt_pointcloud)

        # Generate torch tensors of zeros that has the same dimension as pointcloud (we only sample on surface)
        pointcloud = torch.tensor(pointcloud, dtype=torch.float32).to(device)
        sdf_gt = torch.zeros_like(pointcloud[:, 0]).view(-1, 1).to(device)

        # Get the average optimised latent code
        results_path = os.path.join(model_dir, 'results.npy')
        results = np.load(results_path, allow_pickle=True).item()
        latent_code = results['best_latent_codes']
        # Get average latent code (across dimensions) 
        latent_code = torch.mean(torch.tensor(latent_code, dtype=torch.float32), dim=0).to(device)
        latent_code.requires_grad = True

        # Infer latent code
        best_latent_code = model.infer_latent_code(cfg, pointcloud, sdf_gt, writer, latent_code)

        # Extract mesh obtained with the latent code optimised at inference
        sdf = utils_deepsdf.predict_sdf(best_latent_code, coords_batches, model)
        
        try:
            vertices, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf)
            output_mesh = utils_mesh._as_mesh(trimesh.Trimesh(vertices, faces))

            # compute metrics
            print("Compute Metrics")
            num_samples = gt_pointcloud.vertices.shape[0]
            chamfer = compute_trimesh_chamfer(gt_pointcloud, output_mesh, num_mesh_samples=num_samples)
            mesh_acc = compute_mesh_accuracy(gt_pointcloud, output_mesh, n_samples=num_samples)
            emd = compute_trimesh_emd(gt_pointcloud, output_mesh, n_samples=500) # reduced bc very computationally expensive
            print(chamfer, mesh_acc)
            # save metrics to results:
            results_dict[obj_category_id]["chamfer"].append(chamfer)
            results_dict[obj_category_id]["mesh_acc"].append(mesh_acc)
            results_dict[obj_category_id]["emd"].append(emd)    
            results_dict[obj_category_id]["processed_samples"] += 1
        except Exception as e:
            print(f"Mesh extraction failed {e}. Skip to next sample")
            continue

        # Save mesh if activated
        if cfg['save_output_meshes']:
            output_mesh_path = os.path.join(inference_dir, f'reconstruction_{obj_shape_id}.obj')
            trimesh.exchange.export.export_mesh(output_mesh, output_mesh_path, file_type='obj')

        # Save results every
        if (i+1) % cfg['save_every'] == 0:
            print("test")
            with open(results_dict_path, "w") as file:
                json.dump(results_dict, file, indent=4)


if __name__ == '__main__':

    cfg_path = os.path.join(PROJECT_ROOT, "config_files/", 'benchmark.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)