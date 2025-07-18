import numpy as np
import os
from glob import glob
from multiprocessing import Pool, cpu_count
import yaml
import trimesh
import point_cloud_utils as pcu
import data.ShapeNetCoreV2 as ShapeNetCoreV2
import config_files
import results
from utils import utils_mesh
import gc

def combine_sample_latent(samples, latent_class):
    latent_class_full = np.tile(latent_class, (samples.shape[0], 1))
    return np.hstack((latent_class_full, samples))

def process_single_mesh(args):
    obj_idx, obj_path, cfg = args
    try:
        verts, faces = pcu.load_mesh_vf(obj_path.strip())
        mesh_original = utils_mesh._as_mesh(trimesh.load(obj_path.strip()))

        if not mesh_original.is_watertight:
            verts, faces = pcu.make_mesh_watertight(mesh_original.vertices, mesh_original.faces, 50000)

        p_vol = np.random.rand(cfg['num_samples_in_volume'], 3) * 2 - 1
        v_min, v_max = verts.min(0), verts.max(0)
        p_bbox = np.random.uniform(low=v_min, high=v_max, size=(cfg['num_samples_in_bbox'], 3))

        fid_surf, bc_surf = pcu.sample_mesh_random(verts, faces, cfg['num_samples_on_surface'])
        p_surf = pcu.interpolate_barycentric_coords(faces, fid_surf, bc_surf, verts)

        p_total = np.vstack((p_vol, p_bbox, p_surf))
        sdf, _, _ = pcu.signed_distance_to_mesh(p_total, verts, faces)

        obj_idx_str = os.sep.join(obj_path.split(os.sep)[-4:-2])
        print('Processed mesh: {}'.format(obj_idx_str))
        result = {
            'idx': obj_idx,
            'idx_str': obj_idx_str,
            'samples_latent_class': combine_sample_latent(p_total, np.array([obj_idx], dtype=np.int32)),
            'sdf': sdf
        }

        filename = obj_path.strip().replace("model_normalized.obj", "sdf")
        np.save(filename, result, allow_pickle=True)

        del verts, faces, mesh_original, p_vol, p_bbox, p_surf, p_total, sdf, result, fid_surf
        gc.collect()
        return
        
    except Exception as e:
        print(f"[!] Error processing {obj_path}: {e}")
        return None

def main(cfg):
    #obj_paths = glob(os.path.join(os.path.dirname(ShapeNetCoreV2.__file__), '*', '*', 'models', '*.obj'))
    with open(f"{os.path.dirname(ShapeNetCoreV2.__file__)}/splits/train.txt", "r") as f:
        obj_paths = f.readlines()
    args_list = [(idx, path, cfg) for idx, path in enumerate(obj_paths)]

    #samples_dict = {}
    #idx_str2int_dict = {}
    #idx_int2str_dict = {}

    with Pool(processes=cpu_count()) as pool:
        for result in pool.imap_unordered(process_single_mesh, args_list):
            print("Saved")
            gc.collect()

    #out_dir = os.path.dirname(results.__file__)
    #np.save(os.path.join(out_dir, f'samples_dict_{cfg["dataset"]}.npy'), samples_dict)
    #np.save(os.path.join(out_dir, 'idx_str2int_dict.npy'), idx_str2int_dict)
    #np.save(os.path.join(out_dir, 'idx_int2str_dict.npy'), idx_int2str_dict)

if __name__ == '__main__':
    cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'extract_sdf.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)