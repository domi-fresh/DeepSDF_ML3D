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
                if len(value.shape) == 1:    # increase dim if monodimensional, needed to vstack
                    value = value.view(-1, 1)
                if key not in list(self.data.keys()):
                    self.data[key] = value
                else:
                    self.data[key] = torch.vstack((self.data[key], value)) # originally every sample is stored as a key value pair --> stack them in a tensor

        self.idx_int2str_dict = np.load(Path(__file__).parent.parent/"results/idx_int2str_dict.npy") # sample_id, class/shape id pair
        with open("ShapeNetCoreV2/class_label.json") as file: # Label-Class id lookup table TODO: implement for large dataset
            self.class_label_dict = json.load(file)

        return

    def __len__(self):
        return self.data['sdf'].shape[0]

    def __getitem__(self, idx):
        latent_class = self.data['samples_latent_class'][idx, :] # latent_class is [latent_class(shape) (int), x, y, z]
        sdf = self.data['sdf'][idx]

        class_id = self.idx_int2str_dict[idx].split("/")[0] #get class id for sample at index idx
        class_label = self.class_label_dict[class_id] # e.g. chair
        class_label_token = clip.tokenize(class_label) # tokenized(chair) Tensor(77,)

        return latent_class, sdf, class_label_token # --> [latent_shape_class, x, y, z], sdf,

if __name__=='__main__':
    dataset_name = "ShapeNetCore"
    dataset = SDFDataset(dataset_name)
