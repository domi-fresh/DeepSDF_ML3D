import numpy as np 

import ShapeNetCoreV2 as ShapeNetCore
import ShapeNetSofas 
from glob import glob
import os
import random

DATASET = ShapeNetCore

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
                f.writelines(filename)
            
def generate_splits_uniform(obj_per_cls):
    dirname = os.path.dirname(DATASET.__file__)
    classes = os.listdir(dirname)

    train_files = []
    val_files = []
    test_files = []

    os.makedirs(f'{dirname}/splits', exist_ok=True)

    for cls in classes: 
        objs = sorted(glob(os.path.join(f"{dirname}/{cls}", '*', 'models', 'model_normalized.obj')))
        n_train_objs = obj_per_cls[0]
        n_val_objs = obj_per_cls[1]
        n_test_objs = obj_per_cls[2]

        train_files += objs[:n_train_objs]
        val_files += objs[n_train_objs:n_train_objs+n_val_objs]
        test_files += objs[n_train_objs+n_val_objs:n_train_objs+n_val_objs+n_test_objs]

    with open(f"{dirname}/splits/train.txt", "w") as f:
        for filename in train_files:
            f.write(filename + '\n')
    
    with open(f"{dirname}/splits/val.txt", "w") as f:
        for filename in val_files:
            f.write(filename + '\n')

    with open(f"{dirname}/splits/test.txt", "w") as f:
        for filename in test_files:
            f.write(filename + '\n')

if __name__ == "__main__":
    generate_splits_uniform([600, 100, 100])
    #generate_splits(
    #    {
    #    "train": 0.8,
     #   "val": 0.1,
    #    "overfit": 0.02
     #   }
    #)