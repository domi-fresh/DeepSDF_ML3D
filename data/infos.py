import data.ShapeNetCoreV2 as ShapeNetCoreV2
import os
import json

DATASET = ShapeNetCoreV2

def print_dset_infos():
    current_file_path = os.path.abspath(DATASET.__file__)
    print("\n \n========== DATASET INFOS =========\n \n")
    current_dir = os.path.dirname(current_file_path)
    print(f"Dataset dir: {current_dir}")

    classes = [dir for dir in os.listdir(current_dir) if dir.isdigit()]
    print(f"Class IDs: {classes} \n")
    
    for cls in classes: 
        objs = os.listdir(f"{current_dir}/{cls}")
        print(f"{cls}: {len(objs)} objects")

print_dset_infos()