import torch.nn as nn
import math
import torch
import copy
from tqdm import tqdm
from utils import utils_deepsdf
import numpy as np
"""
Model based on the paper 'SIREN'. 
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0 if is_first else 1
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First laayer initialization
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                # Subsequent layers initialization
                bound = math.sqrt(6/self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SDFModel(torch.nn.Module):
    def __init__(self, num_layers, skip_connections, latent_size, inner_dim=512, output_dim=1):
        """
        SDF model for multiple shapes.
        Args:
            input_dim: 128 for latent space + 3 points = 131
        """
        super(SDFModel, self).__init__()

        # Num layers of the entire network
        self.num_layers = num_layers 

        # If skip connections, add the input to one of the inner layers
        self.skip_connections = skip_connections

        self.latent_size = latent_size

        # Dimension of the input space (3D coordinates)
        dim_coords = 3 
        input_dim = self.latent_size + dim_coords

        # Copy input size to calculate the skip tensor size
        self.skip_tensor_dim = copy.copy(input_dim)

        # Compute how many layers are not Sequential
        num_extra_layers = 2 if (self.skip_connections and self.num_layers >= 8) else 1
        
        # Add sequential layers
        layers = []
        for _ in range(num_layers - num_extra_layers):
            layers.append(SineLayer(input_dim, inner_dim, is_first=(len(layers) == 0), omega_0=30))
            input_dim = inner_dim
        self.net = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(nn.Linear(inner_dim, output_dim))
        self.skip_layer = nn.Sequential(nn.Linear(inner_dim, inner_dim - self.skip_tensor_dim), nn.Identity())


    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor of shape (batch_size, 131). It contains a stacked tensor [latent_code, samples].
        Returns:
            sdf: output tensor of shape (batch_size, 1)
        """      
        input_data = x.clone().detach()

        # Forward pass
        if self.skip_connections and self.num_layers >= 5:
            for i in range(3):
                x = self.net[i](x)
            x = self.skip_layer(x)
            x = torch.hstack((x, input_data))
            for i in range(self.num_layers - 5):
                x = self.net[3 + i](x)
            sdf = self.final_layer(x)
        else:
            if self.skip_connections:
                print('The network requires at least 5 layers to skip connections. Normal forward pass is used.')
            x = self.net(x)
            sdf = self.final_layer(x)
        return sdf


    def infer_latent_code(self, cfg, pointcloud, sdf_gt, writer, latent_code_initial):
        """Infer latent code from coordinates, their sdf, and a trained model."""

        latent_code = latent_code_initial.clone().detach().requires_grad_(True)
        
        optim = torch.optim.Adam([latent_code], lr=cfg['lr'])

        if cfg['lr_scheduler']:
            scheduler_latent = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', 
                                                    factor=cfg['lr_multiplier'], 
                                                    patience=cfg['patience'], 
                                                    threshold=0.001, threshold_mode='rel')

        best_loss = 1000000

        for epoch in tqdm(range(0, cfg['epochs'])):

            latent_code_tile = torch.tile(latent_code, (pointcloud.shape[0], 1))
            x = torch.hstack((latent_code_tile, pointcloud))

            optim.zero_grad()

            predictions = self(x)

            if cfg['clamp']:
                predictions = torch.clamp(predictions, -cfg['clamp_value'], cfg['clamp_value'])

            loss_value, l1, l2 = utils_deepsdf.SDFLoss_multishape(sdf_gt, predictions, x[:, :self.latent_size], sigma=cfg['sigma_regulariser'])
            loss_value.backward()

            if writer is not None:
                writer.add_scalar('Reconstruction loss', l1.data.cpu().numpy(), epoch)
                writer.add_scalar('Latent code loss', l2.data.cpu().numpy(), epoch)

            optim.step()

            if l1.detach().cpu().item() < best_loss:
                best_loss = l1.detach().cpu().item()
                best_latent_code = latent_code.clone()

            # step scheduler and store on tensorboard (optional)
            if cfg['lr_scheduler']:
                scheduler_latent.step(loss_value.item())
                if writer is not None:
                    writer.add_scalar('Learning rate', scheduler_latent._last_lr[0], epoch)

                if scheduler_latent._last_lr[0] < 1e-6:
                    print('Learning rate too small, stopping training')
                    break

            # logging
            if writer is not None:
                writer.add_scalar('Inference loss', loss_value.detach().cpu().item(), epoch)

        return best_latent_code