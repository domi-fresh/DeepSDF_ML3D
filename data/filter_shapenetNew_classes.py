import numpy as np
import os


target_classes = {
    "bench": "02828884",
    "guitar": "03467517",
}
max_per_class = 600  # amount per class


samples_dict = np.load("../results/samples_dict_ShapeNetCore.npy", allow_pickle=True).item()
idx2str = np.load("../results/idx_int2str_dict.npy", allow_pickle=True).item()
str2idx = np.load("../results/idx_str2int_dict.npy", allow_pickle=True).item()

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

    
np.save("../results/samples_dict_new.npy", filtered_samples_dict)
np.save("../results/idx_int2str_dict_new.npy", filtered_idx2str)
np.save("../results/idx_str2int_dict_new.npy", filtered_str2idx)