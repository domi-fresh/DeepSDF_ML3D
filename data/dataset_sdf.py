import numpy as np
import torch
from torch.utils.data import Dataset
import os
import results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class SDFDataset(Dataset): #benjamin
    """
    Loads the entire data dict. Then for every __getitem__ samples self.num_samples per object.
    """
    def __init__(self, dataset_name):
        
        # data dict with structure {sample_1: {sdf: 0, samples_latent_class: [latent_class, x, y, z]}, sample_2: ...}
        self.data = np.load(os.path.join(os.path.dirname(results.__file__), f'samples_dict_{dataset_name}.npy'), allow_pickle=True).item()
        
        self.num_samples = 4096 # num of samples to randomply sample from each object per forward pass
        assert self.num_samples <= 4096, "Num samples to high! Only for < 4096 it can be guaranteed to sample 50/50 pos/neg SDF"

        all_latent_classes = set()
        for obj in self.data.values():
            latent_ids = obj['samples_latent_class'][:, 0]
            all_latent_classes.update(latent_ids.tolist())
        
        unique_classes = sorted(list(all_latent_classes))
        self.latent_class_id_to_idx = {cls_id: idx for idx, cls_id in enumerate(unique_classes)}
        self.num_latent_classes = len(unique_classes)
        

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        
        obj = self.data[idx]  #{sdf: 0, samples_latent_class: [latent_class, x, y, z]}
        sdf = obj["sdf"] # shape (23000,1)
        latent_class = obj['samples_latent_class'].copy() # shape (23000,4)

        # Map raw latent class IDs to zero-based indices for the first column
        latent_ids_raw = latent_class[:, 0].astype(int)
        latent_class[:, 0] = np.array([self.latent_class_id_to_idx[i] for i in latent_ids_raw])

        # get points with positive and negative sdf
        latent_class_pos = torch.from_numpy(latent_class[np.where(sdf >= 0)[0]]).float()
        latent_class_neg = torch.from_numpy(latent_class[np.where(sdf < 0)[0]]).float()
        sdf_pos = torch.from_numpy(sdf[np.where(sdf >= 0)[0]]).float()
        sdf_neg = torch.from_numpy(sdf[np.where(sdf < 0)[0]]).float()

        num_pos = latent_class_pos.shape[0]
        num_neg = latent_class_neg.shape[0]

        # sample s.t. half of self.num_samples samples are inside and the other half outside of the shape
        num_samples_half = int(self.num_samples/2)
        if num_pos > num_samples_half and num_neg > num_samples_half:
            num_samples_from_pos = int(self.num_samples/2)
            num_samples_from_neg = self.num_samples - num_samples_from_pos
        elif num_pos > num_samples_half: # not enough neg samples
            num_samples_from_neg = num_neg # use all neg samples, fill rest w positive
            num_samples_from_pos = self.num_samples - num_samples_from_neg
        elif num_neg > num_samples_half: # not enough pos samples
            num_samples_from_pos = num_pos # use all pos samples, fill rest w negative
            num_samples_from_neg = self.num_samples - num_samples_from_pos
        

        indices_pos = np.random.choice(num_pos, size=num_samples_from_pos, replace=False) 
        indices_neg = np.random.choice(num_neg, size=num_samples_from_neg, replace=False)

        samples_latent_class = torch.cat((latent_class_pos[indices_pos], latent_class_neg[indices_neg]), dim=0).to(device) 
        samples_sdf = torch.cat((sdf_pos[indices_pos], sdf_neg[indices_neg]), dim=0).to(device)
        # Debug print: unique latent class indices (zero-based)
        
        


        return samples_latent_class, samples_sdf # [latent_shape_class, x, y, z], sdf shape (self.num_samples, 4), (self.num_samples,1)




if __name__=='__main__':
    dataset_name = "sofa"
    dataset = SDFDataset(dataset_name)



