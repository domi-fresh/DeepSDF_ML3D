import numpy as np
import results 
import os
from utils import utils_mesh
import point_cloud_utils as pcu
import data.ShapeNetCoreV2 as ShapeNetCoreV2
import data.ShapeNetSofas as ShapeNetSofas
from glob import glob
from datetime import datetime
import config_files
import yaml
import pybullet as pb
import trimesh
"""
For each object, sample points and store their distance to the nearest triangle.
Sampling follows the approach used in the DeepSDF paper.

each object is saved as: 

{
    "sdf": SDF np array of the object
}
"""

class ThreadedExtractor:
    def __init__(self):
        pass

def combine_sample_latent(samples, latent_class):
    """Combine each sample (x, y, z) with the latent code generated for this object.
    Args:
        samples: collected points, np.array of shape (N, 3)
        latent: randomly generated latent code, np.array of shape (1, args.latent_size)
    Returns:
        combined hstacked latent code and samples, np.array of shape (N, args.latent_size + 3)
    """
    latent_class_full = np.tile(latent_class, (samples.shape[0], 1))   # repeat the latent code N times for stacking
    return np.hstack((latent_class_full, samples))


DATASET = ShapeNetSofas

def main(cfg):
  
    # Full paths to all .obj
    obj_paths = sorted(glob(os.path.join(os.path.dirname(DATASET.__file__), '*', '*', 'models', '*.obj')))

    # File to store the samples and SDFs  

    processed = 0    

    for obj_idx, obj_path in enumerate(obj_paths):

        # Object unique index. Str to int by byte encoding

        print(f"Processing {obj_path}")
        try:
            verts, faces = pcu.load_mesh_vf(obj_path)

            # Convert to watertight mesh
            mesh_original = utils_mesh._as_mesh(trimesh.load(obj_path))
            
            if not mesh_original.is_watertight:
                verts, faces = pcu.make_mesh_watertight(mesh_original.vertices, mesh_original.faces, 50000)
                mesh_original = trimesh.Trimesh(vertices=verts, faces=faces)

        except Exception as e:
            print(e)
            continue

        # In Shapenet, the front is the -Z axis with +Y still being the up axis. Rotate objects to align with the canonical axis. 
        mesh = utils_mesh.shapenet_rotate(mesh_original)
        verts = np.array(mesh.vertices)

        # Generate random points in the predefined volume that surrounds all the shapes.
        # NOTE: ShapeNet shapes are normalized within [-1, 1]^3
        p_vol = np.random.rand(cfg['num_samples_in_volume'], 3) * 2 - 1

        # Sample within the object's bounding box. This ensures a higher ratio between points inside and outside the surface.
        v_min, v_max = verts.min(0), verts.max(0)
        p_bbox = np.random.uniform(low=[v_min[0], v_min[1], v_min[2]], high=[v_max[0], v_max[1], v_max[2]], size=(cfg['num_samples_in_bbox'], 3))

        # Sample points on the surface as face ids and barycentric coordinates
        fid_surf, bc_surf = pcu.sample_mesh_random(verts, faces, cfg['num_samples_on_surface'])

        # Compute 3D coordinates and normals of surface samples
        p_surf = pcu.interpolate_barycentric_coords(faces, fid_surf, bc_surf, verts)

        p_total = np.vstack((p_vol, p_bbox, p_surf))

        # Comput the SDF of the random points
        sdf, _, _  = pcu.signed_distance_to_mesh(p_total, verts, faces)

        sdf = sdf.reshape((-1, 1))
        sdf = np.hstack((p_total, sdf))

        filename = os.path.join(os.path.dirname(obj_path), "sdf.npy")
        np.save(filename, sdf)
        processed += 1
        print(f"Saved to: {filename}  {processed}/{len(obj_paths)}")



if __name__=='__main__':
    cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'extract_sdf.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)