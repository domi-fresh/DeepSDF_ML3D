# Adapted from the original DeepSDF repo: https://github.com/facebookresearch/DeepSDF

# Copy of the original repo's license
#MIT License

#Copyright (c) 2019 Facebook

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
from scipy.optimize import linear_sum_assignment

def compute_trimesh_chamfer(gt_points, gen_mesh, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    #gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    print(gt_to_gen_chamfer, gen_to_gt_chamfer)

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def compute_mesh_accuracy(gt_points_cloud, gen_mesh, n_samples=30000, percentile=90):
    """
    Computes mesh accuracy as the 90th percentile distance from generated points to GT mesh.
    This follows the DeepSDF and AtlasNet definition.

    Args:
        gt_points_cloud (trimesh.points.PointCloud): GT points (as trimesh PointCloud).
        gen_mesh (trimesh.Trimesh): Generated mesh (prediction).
        n_samples (int): Number of points to sample on generated surface.
        percentile (float): Which percentile to return (default 90).

    Returns:
        float: Minimum distance d such that `percentile`% of generated points
               are within d of the GT surface.
    """
    # Sample points from predicted surface
    pred_points, _ = trimesh.sample.sample_surface(gen_mesh, n_samples)
    
    # Ground-truth points as array
    gt_points = np.asarray(gt_points_cloud.vertices)
    
    # KDTree for fast NN queries
    tree = KDTree(gt_points)
    distances, _ = tree.query(pred_points)
    
    # Return 90-th percentile
    d = np.percentile(distances, percentile)
    return d


def compute_trimesh_emd(gt_points_cloud, gen_mesh, n_samples=500):
    """
    Computes the Earth Mover's Distance (EMD) between the ground truth point cloud
    and a sampled version of the generated mesh surface, using exact matching.

    Args:
        gt_points_cloud (trimesh.points.PointCloud): Ground truth points.
        gen_mesh (trimesh.Trimesh): Reconstructed/generated mesh.
        num_points (int): Number of points to use in both sets (as in DeepSDF: 500).

    Returns:
        float: Earth Mover's Distance (mean L2 distance between optimally matched points).
    """

    # Sample num_points from mesh surface
    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, n_samples)[0]

    # Get GT points as (N, 3) numpy array
    gt_points_np = np.asarray(gt_points_cloud.vertices)

    # Downsample GT points because compute intensive
    if len(gt_points_np) > n_samples:
        indices = np.random.choice(len(gt_points_np), size=n_samples, replace=False)
        gt_points_np = gt_points_np[indices]
    elif len(gt_points_np) < n_samples:
        raise ValueError(f"GT point cloud has fewer than {n_samples} points.")

    # Pairwise L2 distance matrix
    cost_matrix = np.linalg.norm(
        gt_points_np[:, np.newaxis, :] - gen_points_sampled[np.newaxis, :, :],
        axis=2
    )

    # Solve optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # mean distance between matched points
    emd = cost_matrix[row_ind, col_ind].mean()
    return emd
