import numpy as np
import torch
from torch.utils.data import Dataset
import os
import results

from pathlib import Path
import clip
import json

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
                if len(value.shape) == 1:    # increase dim if monodimensional, needed to vstack (add singleton dim)
                    value = value.view(-1, 1)
                if key not in list(self.data.keys()): # add sdf key or samples_latent_class key if first time
                    self.data[key] = value
                else:
                    self.data[key] = torch.vstack((self.data[key], value)) # if key exists stack old values and new values as tensor --> dict{key: tensor}

        self.idx_int2str_dict = np.load(Path(__file__).parent.parent/"results/idx_int2str_dict.npy", allow_pickle=True).item() # sample_id, class/shape id pair
        with open("data/shape_info.json", "r") as file: # Label-Class id lookup table TODO: implement for large dataset
            self.class_label_dict = json.load(file)

        return

    def __len__(self):
        return self.data['sdf'].shape[0]

    def __getitem__(self, idx):
        

        latent_class = self.data['samples_latent_class'][idx, :] # latent_class is [latent_class(shape) (int), x, y, z]
        sdf = self.data['sdf'][idx]

        return latent_class, sdf # --> [latent_shape_class, x, y, z], sdf

if __name__=='__main__':
    dataset_name = "ShapeNetCore"
    dataset = SDFDataset(dataset_name)
