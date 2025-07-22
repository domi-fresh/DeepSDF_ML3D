import numpy as np
import os

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(results_dir, exist_ok=True)


target_classes = {
    "sofa": "04256520",
}
max_per_class = 600  # amount per class

samples_path = os.path.join(results_dir, "samples_dict_ShapeNetCore.npy")
samples_dict = np.load(samples_path, allow_pickle=True).item()
idx2str_path = os.path.join(results_dir, "idx_int2str_dict.npy")
idx2str = np.load(idx2str_path, allow_pickle=True).item()
str2idx_path = os.path.join(results_dir, "idx_str2int_dict.npy")
str2idx = np.load(str2idx_path, allow_pickle=True).item()

#new filter dictionaries
filtered_samples_dict = {}
filtered_idx2str = {}
filtered_str2idx = {}

new_index = 0


for class_name, class_id in target_classes.items():
    print(f"\n⏳ Filtering class: {class_name} ({class_id})")
    
    matching_ids = [
        idx for idx, path in idx2str.items()
        if path.startswith(class_id + "/")
    ][:max_per_class]

    for old_idx in matching_ids:
        key = idx2str[old_idx]

        # new index assign
        filtered_samples_dict[new_index] = samples_dict[old_idx]
        filtered_idx2str[new_index] = key
        filtered_str2idx[key] = new_index
        new_index += 1

    
np.save(os.path.join(results_dir, "samples_dict_sofa.npy"), filtered_samples_dict)
# np.save(os.path.join(results_dir, "idx_int2str_dict.npy"), filtered_idx2str)
# np.save(os.path.join(results_dir, "idx_str2int_dict.npy"), filtered_str2idx)

