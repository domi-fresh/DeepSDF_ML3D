import numpy as np
import torch
from torch.utils.data import Dataset
import os
import results
from pathlib import Path

import data.ShapeNetCoreV2 as ShapeNetCore
from glob import glob
import os

DATASET = ShapeNetCore

class ShapeNetSDFDataset(Dataset):
    def __init__(self, split, num_sample_points=1024):

        super().__init__()

        self.num_sample_points = num_sample_points
        items = Path(f"{os.path.dirname(DATASET.__file__)}/splits/{split}.txt").read_text().splitlines() 
        self.items = [item.replace("\n", "").replace("model_normalized.obj", "sdf.npy") for item in items]
        self.items = sorted(self.items)

        self.cls_int2str_dict = {}
        self.cls_str2int_dict = {}
        self.obj_int2str_dict = {}
        self.obj_str2int_dict = {}

        classes = os.listdir(os.path.dirname(DATASET.__file__))         # map all classes
        self.classes = sorted([cls for cls in classes if cls.isnumeric()])

        for idx, cls_name in enumerate(self.classes):
            self.cls_int2str_dict[idx] = cls_name
            self.cls_str2int_dict[cls_name] = idx

        for idx, filename in enumerate(self.items):                     # map only the listed objects in the txt file
            parts = filename.split("/")
            obj_name = f"{parts[-4]}/{parts[-3]}"
            self.obj_int2str_dict[idx] = obj_name
            self.obj_str2int_dict[obj_name] = idx

    def __getitem__(self, index):

        # get shape_id at index
        item = self.items[index]

        # get path to sdf data
        sdf_samples_path = item

        sdf_samples = np.load(sdf_samples_path, allow_pickle=True).item()

        points = torch.from_numpy(sdf_samples["samples_latent_class"][:, 1:]).float()
        sdf = torch.from_numpy(sdf_samples["sdf"]).unsqueeze(1).float()

        train_size = int(sdf.shape[0] * 0.95)
        points_train, points_val = points[:train_size], points[train_size:]
        sdf_train, sdf_val = sdf[:train_size], sdf[train_size:]

        indices_train = np.random.choice(points_train.shape[0], self.num_sample_points, replace=False) # sample from train 
        indices_val = np.random.choice(points_val.shape[0], int(self.num_sample_points*0.1), replace=False) # sample from val

        points_val, sdf_val = points_val[indices_val], sdf_val[indices_val]
        points_train, sdf_train = points_train[indices_train], sdf_train[indices_train]

        # truncate sdf values
        sdf_train = torch.clamp(sdf_train, -0.1, 0.1)
        sdf_val = torch.clamp(sdf_val, -0.1, 0.1)

        # get shape index and class index 
        parts = sdf_samples_path.split("/")
        obj_name = f"{parts[-4]}/{parts[-3]}"
        shape_id = self.obj_str2int_dict[obj_name]
        class_name = parts[-4]
        class_id = self.cls_str2int_dict[class_name]

        return {
            "shape_id": shape_id,  
            "class_id": class_id,  
            "points_train": points_train,  
            "sdf_train": sdf_train,
            "points_val": points_val,  
            "sdf_val": sdf_val  
        }

    def __len__(self):
        """
        :return: length of the dataset
        """
        # TODO: Implement
        return len(self.items)

    @staticmethod
    def batch_to_device(batch, device):
        batch['shape_id'] = batch['shape_id'].to(device)
        batch['class_id'] = batch['class_id'].to(device)
        batch['points_train'] = batch['points_train'].to(device)
        batch['sdf_train'] = batch['sdf_train'].to(device)
        batch['points_val'] = batch['points_val'].to(device)
        batch['sdf_val'] = batch['sdf_val'].to(device)
        
if __name__ == "__main__":
    dataset = ShapeNetSDFDataset(split="train", num_sample_points=1024)
    dataset[0]
    print(dataset.dir)
    