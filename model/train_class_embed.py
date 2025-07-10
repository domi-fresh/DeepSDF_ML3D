import torch
import model.model_sdf as sdf_model
import torch.optim as optim
import data.dataset_sdf as dataset
from torch.utils.data import random_split
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

from data.dataset_shapenet import ShapeNetSDFDataset

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

        # dataloaders
        train_loader = DataLoader(
            ShapeNetSDFDataset("overfit"),
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            drop_last=True
        )

        val_loader = DataLoader(
            ShapeNetSDFDataset("val"),
            batch_size=self.train_cfg["batch_size"],
            shuffle=False,
            drop_last=True
        )

        # instantiate model and optimisers
        self.model = sdf_model.SDFModel(
                self.train_cfg['num_layers'], 
                self.train_cfg['skip_connections'], 
                inner_dim=self.train_cfg['inner_dim'],
                latent_size=self.train_cfg['latent_size']
            ).float().to(device)

        # define optimisers
        self.optimizer_model = optim.Adam(self.model.parameters(), lr=self.train_cfg['lr_model'], weight_decay=0)
        
        # generate a unique random latent code for each shape
        self.latent_codes = torch.nn.Embedding(len(ShapeNetSDFDataset("train")), self.train_cfg["latent_size"]).float()
        self.optimizer_latent = optim.Adam(self.latent_codes.parameters(), lr=self.train_cfg['lr_latent'], weight_decay=0)
        
        for epoch in range(self.train_cfg['epochs']):
            print(f'============================ Epoch {epoch} ============================')
            self.epoch = epoch
            avg_train_loss = self.train(train_loader, self.latent_codes)

    def train(self, train_loader, latent_codes):
        total_loss = 0.0
        iterations = 0.0
        self.model.train()
        for batch in train_loader:
            # batch[0]: [class, x, y, z], shape: (batch_size, 4)
            # batch[1]: [sdf], shape: (batch size)
            iterations += 1.0

            self.optimizer_model.zero_grad()
            self.optimizer_latent.zero_grad()

            latent_code = latent_codes(batch['indices']).unsqueeze(1).expand(-1, batch['points'].shape[1], -1).float()
            latent_code = torch.flatten(latent_code, end_dim=1)
            latent_code = latent_code.to(device)
            points = batch["points"].float()
            points = torch.flatten(points, end_dim=1)
            points = points.to(device)
            y = batch["sdf"].float()
            y = torch.flatten(y, end_dim=1)
            y = y.to(device)

            x = torch.concat((latent_code, points), dim=1)
            predictions = self.model(x)  # (batch_size, 1)
            if self.train_cfg['clamp']:
                predictions = torch.clamp(predictions, -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])
            
            loss_value, loss_rec, loss_latent = self.train_cfg['loss_multiplier'] * SDFLoss_multishape(y, predictions, x[:, :self.train_cfg['latent_size']], sigma=self.train_cfg['sigma_regulariser'])
            loss_value.backward()       

            self.optimizer_latent.step()
            self.optimizer_model.step()
            total_loss += loss_value.data.cpu().numpy()  

        avg_train_loss = total_loss/iterations
        print(f'Training: loss {avg_train_loss}')
        self.writer.add_scalar('Training loss', avg_train_loss, self.epoch)

        return avg_train_loss

    def validate(self, val_loader, latent_codes):
        total_loss = 0.0
        total_loss_rec = 0.0
        total_loss_latent = 0.0
        iterations = 0.0
        self.model.eval()

        for batch in val_loader:
            # batch[0]: [class, x, y, z], shape: (batch_size, 4)
            # batch[1]: [sdf], shape: (batch size)
            iterations += 1.0            

            x, y, _, latent_codes_batch = self.generate_xy(batch)

            predictions = self.model(x)  # (batch_size, 1)
            if train_cfg['clamp']:
                predictions = torch.clamp(predictions, -train_cfg['clamp_value'], train_cfg['clamp_value'])

            loss_value, loss_rec, loss_latent = self.train_cfg['loss_multiplier'] * SDFLoss_multishape(y, predictions, latent_codes_batch, self.train_cfg['sigma_regulariser'])          
            total_loss += loss_value.data.cpu().numpy()   
            total_loss_rec += loss_rec.data.cpu().numpy() 
            total_loss_latent += loss_latent.data.cpu().numpy()

        avg_val_loss = total_loss/iterations
        avg_loss_rec = total_loss_rec/iterations
        avg_loss_latent = total_loss_latent/iterations
        print(f'Validation: loss {avg_val_loss}')
        self.writer.add_scalar('Validation loss', avg_val_loss, self.epoch)
        self.writer.add_scalar('Reconstruction loss', avg_loss_rec, self.epoch)
        self.writer.add_scalar('Latent code loss', avg_loss_latent, self.epoch)

        return avg_val_loss

if __name__=='__main__':
    train_cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'train_sdf.yaml')
    with open(train_cfg_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(train_cfg)
    trainer()