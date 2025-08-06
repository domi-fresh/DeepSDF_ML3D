import json
import numpy as np

# Load JSON data from file
with open("", "r") as f:
    data = json.load(f)

def compute_statistics(entry):
    chamfer = np.array(entry["chamfer"])
    emd = np.array(entry["emd"])
    mesh_acc = np.array(entry["mesh_acc"])

    stats = {
        "median_chamfer": float(np.median(chamfer)),
        "mean_chamfer": float(np.mean(chamfer)),
        "mean_emd": float(np.mean(emd)),
        "mean_mesh_acc": float(np.mean(mesh_acc)),
    }

    return stats

# Process each category
for category_id, entry in data.items():
    stats = compute_statistics(entry)
    print(f"Statistics for category {category_id}:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")
