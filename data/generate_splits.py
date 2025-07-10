import numpy as np 

import ShapeNetCoreV2 as ShapeNetCore
import ShapeNetSofas 
from glob import glob
import os
import random

DATASET = ShapeNetSofas

def generate_splits(ratios):
    dirname = os.path.dirname(DATASET.__file__)
    allfiles = sorted(glob(os.path.join(dirname, '*', '*', 'models', 'sdf.npy')))
    random.shuffle(allfiles)
    print(f"Generating splits for {dirname} with ratios {ratios}")
    os.makedirs(f'{dirname}/splits', exist_ok=True)

    dataset_len = len(allfiles)

    for name, value in ratios.items():
        end_index = int(dataset_len * value)
        files = allfiles[:end_index]
        del allfiles[:end_index]
        
        with open(f"{dirname}/splits/{name}.txt", "w") as f:
            for filename in files:
                f.write(filename + '\n')

if __name__ == "__main__":
    generate_splits(
        {
        "train": 0.8,
        "val": 0.1,
        "overfit": 0.02
        }
    )