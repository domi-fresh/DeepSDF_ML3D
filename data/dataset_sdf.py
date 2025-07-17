import numpy as np
import torch
from torch.utils.data import Dataset
import os
import results
from utils.utils_deepsdf import SDFLoss_triplane_multishape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SDFDataset(Dataset):
    """
    TODO: adapting to handle multiple objects
    """
    def __init__(self, dataset_name):
        samples_dict = np.load(os.path.join(os.path.dirname(results.__file__), f'samples_dict_{dataset_name}.npy'), allow_pickle=True).item()
        self.data = dict()
        for obj_idx in list(samples_dict.keys()):  # samples_dict.keys() for all the objects
            for key in samples_dict[obj_idx].keys():   # keys are ['samples', 'sdf', 'latent_class', 'samples_latent_class']
                value = torch.from_numpy(samples_dict[obj_idx][key]).float().to(device)
                if len(value.shape) == 1:    # increase dim if monodimensional, needed to vstack
                    value = value.view(-1, 1)
                if key not in list(self.data.keys()):
                    self.data[key] = value
                else:
                    self.data[key] = torch.vstack((self.data[key], value))
        return

    def __len__(self):
        return self.data['sdf'].shape[0]

    def __getitem__(self, idx):
        latent_class = self.data['samples_latent_class'][idx, :]
        sdf = self.data['sdf'][idx]
        return latent_class, sdf


class SDFTriPlaneDataset(Dataset):
    def __init__(self, dataset_name):
        samples_dict = np.load(
            os.path.join(os.path.dirname(results.__file__), f'samples_dict_{dataset_name}.npy'),
            allow_pickle=True
        ).item()

        self.samples = []
        self.sdfs = []
        self.shape_ids = []

        for shape_id, obj_data in enumerate(samples_dict.values()):
            #samples = obj_data['samples']  # shape: (N, 3)
            #sdfs = obj_data['sdf']  # shape: (N,) or (N, 1)

            samples_latent = obj_data['samples_latent_class']  # shape: (N, latent_size + 3)
            coords = samples_latent[:, -3:]  # Get x, y, z
            sdfs = obj_data['sdf']

            if len(sdfs.shape) == 1:
                sdfs = sdfs[:, None]

            clamp_val = 0.1  # or pass this as an argument if needed
            sdfs = np.clip(sdfs, -clamp_val, clamp_val)

            #self.samples.append(samples)
            self.samples.append(coords)
            self.sdfs.append(sdfs)
            #self.shape_ids.append(np.full((samples.shape[0],), shape_id))
            self.shape_ids.append(np.full((coords.shape[0],), shape_id))

        self.samples = np.vstack(self.samples)         # (total_points, 3)
        self.sdfs = np.vstack(self.sdfs)               # (total_points, 1)
        self.shape_ids = np.concatenate(self.shape_ids)  # (total_points,)

    def __len__(self):
        return len(self.sdfs)

    def __getitem__(self, idx):
        coord = torch.from_numpy(self.samples[idx]).float()
        sdf = torch.from_numpy(self.sdfs[idx]).float()
        shape_id = torch.tensor(self.shape_ids[idx]).long()
        return coord, sdf, shape_id

if __name__=='__main__':
    dataset_name = "ShapeNetCore"
    dataset = SDFDataset(dataset_name)
