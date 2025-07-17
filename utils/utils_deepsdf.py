import torch
import meshplot as mp
import skimage
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mp.offline()

def clamp(x, delta=torch.tensor([[0.1]]).to(device)):
    """Clamp function introduced in the paper DeepSDF.
    This returns a value in range [-delta, delta]. If x is within this range, it returns x, else one of the extremes.

    Args:
        x: prediction, torch tensor (batch_size, 1)
        delta: small value to control the distance from the surface over which we want to mantain metric SDF
    """
    maximum = torch.amax(torch.vstack((x, -delta)))
    minimum = torch.amin(torch.vstack((delta[0], maximum)))
    return minimum


def SDFLoss_multishape(sdf, prediction, x_latent, sigma):
    """Loss function introduced in the paper DeepSDF for multiple shapes."""
    l1 = torch.mean(torch.abs(prediction - sdf))
    l2 = sigma**2 * torch.mean(torch.linalg.norm(x_latent, dim=1, ord=2))
    loss = l1 + l2
    #print(f'Loss prediction: {l1:.3f}, Loss regulariser: {l2:.3f}')
    return loss, l1, l2

def SDFLoss_triplane_multishape(sdf_gt, prediction, xy_planes, yz_planes, zx_planes, sigma):
    """
    Loss function for DeepSDF with triplane features.
    Args:
        sdf_gt: Ground truth SDF values, shape (B, 1)
        prediction: Predicted SDF values, shape (B, 1)
        xy_planes, yz_planes, zx_planes: (B, C, H, W) sampled planes per shape
        sigma: regularization weight
    Returns:
        - total loss
        - reconstruction loss
        - regularization loss
    """
    # Reconstruction (L1) loss
    l1 = torch.mean(torch.abs(prediction - sdf_gt))

    # Regularize L2 norm of plane weights (across batch)
    reg = (
        torch.mean(xy_planes ** 2) +
        torch.mean(yz_planes ** 2) +
        torch.mean(zx_planes ** 2)
    )

    loss = l1 + sigma**2 * reg
    return loss, l1, reg


def triplane_decorrelation_loss(xy_planes, yz_planes, zx_planes, num_subset_channels=8):
    """
    Compute mean decorrelation loss across all 3 planes for each shape in a batch.

    Args:
        xy_planes, yz_planes, zx_planes: Tensors of shape (B, C, H, W)

    Returns:
        Scalar decorrelation loss
    """
    return (
        channel_decorrelation_loss_subset(xy_planes, num_subset_channels) +
        channel_decorrelation_loss_subset(yz_planes, num_subset_channels) +
        channel_decorrelation_loss_subset(zx_planes, num_subset_channels)
    ) / 3


def channel_decorrelation_loss(plane_batch):
    """
    Computes channel decorrelation loss.
    Args:
        plane_batch: Tensor of shape (B, C, H, W)
    Returns:
        Scalar: Mean squared off-diagonal correlation across the batch
    """
    B, C, H, W = plane_batch.shape

    # Step 1: correct reshape — spatial dims flattened last
    x = plane_batch.view(B, C, -1)  # shape: (B, C, H*W) ✅

    # Step 2: normalize per channel
    x = x - x.mean(dim=2, keepdim=True)
    x = x / (x.std(dim=2, keepdim=True) + 1e-8)

    # Step 3: correlation matrix per sample (C × C)
    x = plane_batch.view(B, -1, C)
    corr = torch.einsum('bci,bcj->bij', x, x) / x.shape[2]  # (B, C, C)
    #print(f"x shape: {x.shape}, corr shape: {corr.shape}")

    # Step 4: subtract identity to get off-diagonal
    eye = torch.eye(C, device=plane_batch.device).unsqueeze(0)  # (1, C, C)
    off_diag = corr - eye

    return (off_diag ** 2).mean()

def channel_decorrelation_loss_subset(plane_batch, num_subset_channels=8):
    """
    Compute decorrelation loss on a random subset of channels.
    plane_batch: (B, C, H, W)
    """
    B, C, H, W = plane_batch.shape
    if num_subset_channels > C:
        raise ValueError(f"num_subset_channels={num_subset_channels} > total channels C={C}")

    idx = torch.randperm(C)[:num_subset_channels].to(plane_batch.device)
    subset = plane_batch[:, idx, :, :]  # (B, S, H, W)
    return channel_decorrelation_loss(subset)


def generate_latent_codes(latent_size, samples_dict):
    """Generate a random latent codes for each shape form a Gaussian distribution
    Returns:
        - latent_codes: np.array, shape (num_shapes, latent_size)
        - dict_latent_codes: key: obj_index, value: corresponding idx in the latent_codes array. 
                                  e.g.  latent_codes = ([ [1, 2, 3], [7, 8, 9] ])
                                        dict_latent_codes[345] = 0, the obj that has index 345 refers to 
                                        the 0-th latent code.
    """
    latent_codes = torch.tensor([], dtype=torch.float32).reshape(0, latent_size).to(device)
    #dict_latent_codes = dict()
    for i, obj_idx in enumerate(list(samples_dict.keys())):
        #dict_latent_codes[obj_idx] = i
        latent_code = torch.normal(0, 0.01, size = (1, latent_size), dtype=torch.float32).to(device)
        latent_codes = torch.vstack((latent_codes, latent_code))
    latent_codes.requires_grad_(True)
    return latent_codes #, dict_latent_codes


def get_volume_coords(resolution = 50):
    """Get 3-dimensional vector (M, N, P) according to the desired resolutions."""
    # Define grid
    grid_values = torch.arange(-1, 1, float(1/resolution)).to(device) # e.g. 50 resolution -> 1/50 
    grid = torch.meshgrid(grid_values, grid_values, grid_values)
    
    grid_size_axis = grid_values.shape[0]

    # Reshape grid to (M*N*P, 3)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)

    return coords, grid_size_axis


def save_meshplot(vertices, faces, path):
    mp.plot(vertices, faces, c=vertices[:, 2], filename=path)


def predict_sdf(latent, coords_batches, model):

    sdf = torch.tensor([], dtype=torch.float32).view(0, 1).to(device)

    model.eval()
    with torch.no_grad():
        for coords in coords_batches:
            latent_tile = torch.tile(latent, (coords.shape[0], 1))
            coords_latent = torch.hstack((latent_tile, coords))
            sdf_batch = model(coords_latent)
            sdf = torch.vstack((sdf, sdf_batch))        

    return sdf


def extract_mesh(grad_size_axis, sdf):
    # Extract zero-level set with marching cubes
    grid_sdf = sdf.view(grad_size_axis, grad_size_axis, grad_size_axis).detach().cpu().numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(grid_sdf, level=0.00)

    # Rescale vertices extracted with marching cubes (https://stackoverflow.com/questions/70834443/converting-indices-in-marching-cubes-to-original-x-y-z-space-visualizing-isosu)
    x_max = np.array([1, 1, 1])
    x_min = np.array([-1, -1, -1])
    vertices = vertices * ((x_max-x_min) / grad_size_axis) + x_min

    return vertices, faces