import torch
import model.model_sdf as sdf_model
import data.dataset_shapenet as dataset
from torch.utils.data import DataLoader
import results.runs_sdf as runs
from utils.utils_deepsdf import SDFLoss_multishape
import os
from datetime import datetime
import numpy as np
import time
from utils import utils_deepsdf
import results
from torch.utils.tensorboard import SummaryWriter
import yaml
import config_files

# Select device. The 'mps' device (macOS M1 architecture) is not supported as it cannot currently handle weith normalisation. 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

class Trainer():
    def __init__(self, train_cfg):
        self.train_cfg = train_cfg

    def __call__(self):
        # directories
        self.timestamp_run = datetime.now().strftime('%d_%m_%H%M%S')   # timestamp to use for logging data
        self.runs_dir = os.path.dirname(runs.__file__)               # directory fo all runs
        self.run_dir = os.path.join(self.runs_dir, self.timestamp_run)  # directory for this run
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        
        # Logging
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.log_path = os.path.join(self.run_dir, 'settings.yaml')
        with open(self.log_path, 'w') as f:
            yaml.dump(self.train_cfg, f)

        # instantiate model and optimisers
        self.model = sdf_model.SDFModel(
                self.train_cfg['num_layers'], 
                self.train_cfg['skip_connections'], 
                inner_dim=self.train_cfg['inner_dim'],
                latent_size=self.train_cfg['latent_size']
            ).float().to(device)

        train_dataset = dataset.ShapeNetSDFDataset("overfit", 1024)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,   
            batch_size=1,   
            shuffle=True,    
            num_workers=0,  
            #pin_memory=True 
        )

        self.latent_codes = torch.nn.Embedding(len(train_dataset), self.train_cfg['latent_size'], max_norm=1.0).to(device)

        self.optimizer = torch.optim.Adam([
        {
            'params': self.model.parameters(),
            'lr': self.train_cfg['lr_model']
        },
        {
            'params': self.latent_codes.parameters(),
            'lr': self.train_cfg['lr_latent']
        }
        ])
        
        self.results = {
            'best_latent_codes' : []
        }

        best_loss = 10000000000

        # save int2str, str2int mappings
        np.save(f"{self.run_dir}/idx_int2str_dict.npy", train_dataset.obj_int2str_dict)
        np.save(f"{self.run_dir}/idx_str2int_dict.npy", train_dataset.obj_str2int_dict)
        np.save(f"{self.run_dir}/cls_int2str_dict.npy", train_dataset.cls_int2str_dict)
        np.save(f"{self.run_dir}/cls_str2int_dict.npy", train_dataset.cls_str2int_dict)
 
        for epoch in range(self.train_cfg['epochs']):
            print(f'============================ Epoch {epoch} ============================')
            self.epoch = epoch

            total_loss_train = 0.0
            total_loss_rec = 0.0
            total_loss_latent = 0.0
            total_loss_val = 0.0
            iterations = 0
            for batch_idx, batch in enumerate(train_dataloader):
                iterations += 1
                dataset.ShapeNetSDFDataset.batch_to_device(batch, device)

                total_loss_train += self.train(batch)
                
                with torch.no_grad():
                    val_loss, loss_rec, loss_latent = self.validate(batch)

                total_loss_val += val_loss
                total_loss_rec += loss_rec
                total_loss_latent += loss_latent
            
            avg_train_loss = total_loss_train/iterations
            avg_val_loss = total_loss_val/iterations
            avg_loss_rec = total_loss_rec/iterations
            avg_loss_latent = total_loss_latent/iterations

            if avg_val_loss < best_loss:
                best_loss = np.copy(avg_val_loss)
                best_weights = self.model.state_dict()
                best_latent_codes = self.latent_codes.weight.detach().cpu().numpy()
                self.results['best_latent_codes'] = best_latent_codes

                np.save(os.path.join(self.run_dir, 'results.npy'), self.results)
                torch.save(best_weights, os.path.join(self.run_dir, 'weights.pt'))
                

            print(f'Training: loss {avg_train_loss}')
            print(f'Validation: loss {avg_val_loss}')
            self.writer.add_scalar('Validation loss', avg_val_loss, self.epoch)
            self.writer.add_scalar('Reconstruction loss', avg_loss_rec, self.epoch)
            self.writer.add_scalar('Latent code loss', avg_loss_latent, self.epoch)
                

    def train(self, batch):
        self.model.train()

        self.optimizer.zero_grad()

        num_points_per_batch = batch['points_train'].shape[0] * batch['points_train'].shape[1]
            
        batch_latent_vectors = self.latent_codes(batch['shape_id']).unsqueeze(1).expand(-1, batch['points_train'].shape[1], -1)
        batch_latent_vectors = batch_latent_vectors.reshape((num_points_per_batch, self.train_cfg["latent_size"]))

        # reshape points and sdf for forward pass
        points = batch['points_train'].reshape((num_points_per_batch, 3))
        
        x = torch.concat((batch_latent_vectors,points), dim=1)
        y = batch['sdf_train'].reshape((num_points_per_batch, 1))
    
        predictions = self.model(x)  # (batch_size, 1)
        if self.train_cfg['clamp']:
            predictions = torch.clamp(predictions, -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])
        
        loss_value, loss_rec, loss_latent = self.train_cfg['loss_multiplier'] * SDFLoss_multishape(y, predictions, x[:, :self.train_cfg['latent_size']], sigma=self.train_cfg['sigma_regulariser'])
        loss_value.backward()       

        self.optimizer.step()

        return loss_value.item()
    
    def validate(self, batch):
        self.model.eval()

        num_points_per_batch = batch['points_val'].shape[0] * batch['points_val'].shape[1]
            
        batch_latent_vectors = self.latent_codes(batch['shape_id']).unsqueeze(1).expand(-1, batch['points_val'].shape[1], -1)
        batch_latent_vectors = batch_latent_vectors.reshape((num_points_per_batch, self.train_cfg["latent_size"]))

        # reshape points and sdf for forward pass
        points = batch['points_val'].reshape((num_points_per_batch, 3))
        
        x = torch.concat((batch_latent_vectors,points), dim=1)
        y = batch['sdf_val'].reshape((num_points_per_batch, 1))
    
        predictions = self.model(x)  # (batch_size, 1)
        if self.train_cfg['clamp']:
            predictions = torch.clamp(predictions, -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])
        
        loss_value, loss_rec, loss_latent = self.train_cfg['loss_multiplier'] * SDFLoss_multishape(y, predictions, x[:, :self.train_cfg['latent_size']], sigma=self.train_cfg['sigma_regulariser'])

        return loss_value.item(), loss_rec.item(), loss_latent.item()



if __name__=='__main__': 
    train_cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'train_sdf.yaml')
    with open(train_cfg_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(train_cfg)
    trainer()