import numpy as np
import torch
from torch.utils.data import Dataset
import os
import results

import data.ShapeNetCoreV2 as ShapeNetCore
import data.ShapeNetSofas as ShapeNetSofas
from glob import glob
import os

DATASET = ShapeNetSofas

class ShapeNetSDFDataset(Dataset):
    def __init__(self, split:str):
        super().__init__()
        
        self.dir = os.path.dirname(DATASET.__file__)
        with open(f"{self.dir}/splits/{split}.txt") as f:
            self.files = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sdf_samples =  np.load(self.files[index])
        points = sdf_samples[:,:3]
        sdf = sdf_samples[:,3:]
        parts = self.files[index].split("/")
        shape_id = f"{parts[-4]}/{parts[-3]}"
        class_id = parts[-4]
        #sdf_clamped = torch.clamp(sdf, -0.1, 0.1)       # TODO: is this clamp correct?
        return {
            "id": shape_id,
            "class_id":class_id,
            "indices": index,
            "points": points,
            "sdf": sdf
        }

    
if __name__ == "__main__":
    dataset = ShapeNetSDFDataset(split="train")
    print(dataset.dir)
    