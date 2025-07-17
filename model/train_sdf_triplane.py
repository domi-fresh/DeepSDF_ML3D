import torch
import model.model_sdf as sdf_model
import torch.optim as optim
import data.dataset_sdf as dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import results.runs_sdf as runs
from utils.utils_deepsdf import SDFLoss_multishape, SDFLoss_triplane_multishape, triplane_decorrelation_loss
import os
from datetime import datetime
import numpy as np
import time
from utils import utils_deepsdf
import results
from torch.utils.tensorboard import SummaryWriter
import yaml
import config_files
import time

from model.model_triplane import TriPlaneSDFModel

# Select device. The 'mps' device (macOS M1 architecture) is not supported as it cannot currently handle weith normalisation. 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

class Trainer():
    def __init__(self, train_cfg):
        self.train_cfg = train_cfg

    def __call__(self):
        # directories
        self.timestamp_run = datetime.now().strftime('%m_%d_%H%M')   # timestamp to use for logging data
        self.runs_dir = os.path.join('results', 'runs_sdf') #os.path.dirname(runs.__file__)               # directory fo all runs
        self.run_dir = os.path.join(self.runs_dir, (self.timestamp_run + "_triplane"))  # directory for this run
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        
        # Logging
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.log_path = os.path.join(self.run_dir, 'settings.yaml')
        with open(self.log_path, 'w') as f:
            yaml.dump(self.train_cfg, f)

        # calculate num objects in samples_dictionary, wich is the number of keys
        samples_dict_path = os.path.join(os.path.dirname(results.__file__), f'samples_dict_{train_cfg["dataset"]}.npy')
        samples_dict = np.load(samples_dict_path, allow_pickle=True).item()

        num_shapes = len(samples_dict)

        # instantiate model and optimisers
        #self.model = sdf_model.SDFModel(
        #        self.train_cfg['num_layers'],
        #        self.train_cfg['skip_connections'],
        #        inner_dim=self.train_cfg['inner_dim'],
        #        latent_size=self.train_cfg['latent_size']
        #    ).float().to(device)

        #########################################################
        ### Triplane Model
        self.model = TriPlaneSDFModel(
            num_shapes=num_shapes,
            plane_feat_dim=train_cfg['plane_feat_dim'],
            plane_res=train_cfg['plane_res'],
            num_layers=train_cfg['num_layers'],
            skip_connections=self.train_cfg['skip_connections'],
            inner_dim=self.train_cfg['inner_dim']
        ).float().to(device)

        #########################################################

        # define optimisers
        self.optimizer_model = optim.Adam(self.model.parameters(), lr=self.train_cfg['lr_model'], weight_decay=0)

        self.lambda_ = 0.001

        # generate a unique random latent code for each shape
        #self.latent_codes = utils_deepsdf.generate_latent_codes(self.train_cfg['latent_size'], samples_dict)
        #self.optimizer_latent = optim.Adam([self.latent_codes], lr=self.train_cfg['lr_latent'], weight_decay=0)
        
        # Load pretrained weights and optimisers to continue training
        if self.train_cfg['pretrained']:
            # load pretrained weights
            self.model.load_state_dict(torch.load(self.train_cfg['pretrain_weights'], map_location=device))

            # load pretrained optimisers
            self.optimizer_model.load_state_dict(torch.load(self.train_cfg['pretrain_optim_model'], map_location=device))

            # retrieve latent codes from results.npy file
            results_path = self.train_cfg['pretrain_optim_model'].split(os.sep)
            results_path[-1] = 'results.npy'
            results_path = os.sep.join(results_path)
            # load latent codes from results.npy file
            results_latent_codes = np.load(results_path, allow_pickle=True).item()
            #self.latent_codes = torch.tensor(results_latent_codes['best_latent_codes']).float().to(device)
            #self.optimizer_latent = optim.Adam([self.latent_codes], lr=self.train_cfg['lr_latent'], weight_decay=0)
            self.optimizer_latent.load_state_dict(torch.load(self.train_cfg['pretrain_optim_latent'], map_location=device))

        if self.train_cfg['lr_scheduler']:
            self.scheduler_model =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_model, mode='min', factor=self.train_cfg['lr_multiplier'], patience=self.train_cfg['patience'], threshold=0.0001, threshold_mode='rel')
            #self.scheduler_latent =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_latent, mode='min', factor=self.train_cfg['lr_multiplier'], patience=self.train_cfg['patience'], threshold=0.0001, threshold_mode='rel')
            
        # get data
        train_loader, val_loader = self.get_loaders()

        #self.results = {'best_latent_codes' : []}
        self.results = {}

        best_loss = 10000000000
        start = time.time()
        for epoch in range(self.train_cfg['epochs']):
            print(f'============================ Epoch {epoch} ============================')
            self.epoch = epoch

            avg_train_loss = self.train(train_loader)

            with torch.no_grad():
                avg_val_loss = self.validate(val_loader)

                if avg_val_loss < best_loss:
                    best_loss = np.copy(avg_val_loss)
                    best_weights = self.model.state_dict()
                    #best_latent_codes = self.latent_codes.detach().cpu().numpy()
                    optimizer_model_state = self.optimizer_model.state_dict()
                    #optimizer_latent_state = self.optimizer_latent.state_dict()

                    np.save(os.path.join(self.run_dir, 'results.npy'), self.results)
                    torch.save(best_weights, os.path.join(self.run_dir, 'weights.pt'))
                    torch.save(optimizer_model_state, os.path.join(self.run_dir, 'optimizer_model_state.pt'))
                    #torch.save(optimizer_latent_state, os.path.join(self.run_dir, 'optimizer_latent_state.pt'))
                    #self.results['best_latent_codes'] = best_latent_codes

                if self.train_cfg['lr_scheduler']:
                    self.scheduler_model.step(avg_val_loss)
                    #self.scheduler_latent.step(avg_val_loss)

                    self.writer.add_scalar('Learning rate (model)', self.scheduler_model._last_lr[0], epoch)
                    #self.writer.add_scalar('Learning rate (latent)', self.scheduler_latent._last_lr[0], epoch)
            
        end = time.time()
        print(f'Time elapsed: {end - start} s')

    def get_loaders(self):
        data = dataset.SDFDataset(self.train_cfg['dataset'])

        if self.train_cfg['clamp']:
            data.data['sdf'] = torch.clamp(data.data['sdf'], -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])

        train_size = int(0.85 * len(data))
        val_size = len(data) - train_size
        train_data, val_data = random_split(data, [train_size, val_size])
        train_loader = DataLoader(
                train_data,
                batch_size=self.train_cfg['batch_size'],
                shuffle=True,
                drop_last=True#, num_workers=4, pin_memory=True
            )
        val_loader = DataLoader(
            val_data,
            batch_size=self.train_cfg['batch_size'],
            shuffle=False,
            drop_last=True#, num_workers=4, pin_memory=True
            )
        return train_loader, val_loader


    def generate_xy(self, batch):
        """
        Combine latent code and coordinates.
        Return:
            - x: latent codes + coordinates, torch tensor shape (batch_size, latent_size + 3)
            - y: ground truth sdf, shape (batch_size, 1)
            - latent_codes_indices_batch: all latent class indices per sample, shape (batch size, 1).
                                            e.g. [[2], [2], [1], ..] eaning the batch contains the 2nd, 2nd, 1st latent code
            - latent_batch_codes: all latent codes per sample, shape (batch_size, latent_size)
        Return ground truth as y, and the latent codes for this batch.
        """
        #latent_classes_batch = batch[0][:, 0].view(-1, 1).to(torch.long)               # shape (batch_size, 1)
        coords = batch[0][:, 1:]                                  # shape (batch_size, 3)
        #latent_codes_batch = self.latent_codes[latent_classes_batch.view(-1)]    # shape (batch_size, 128)

        x = torch.hstack((latent_codes_batch, coords))                  # shape (batch_size, 131)
        y = batch[1]     # (batch_size, 1)

        return x, y, latent_classes_batch.view(-1), latent_codes_batch

    def generate_xy_triplane(self, batch):
        """
        Extract shape indices and 3D coordinates.

        Returns:
            - coords: (batch_size, 3)
            - sdf_gt: (batch_size, 1)
            - shape_ids: (batch_size,) — integer index per shape
        """
        shape_ids = batch[0][:, 0].long().to(device)  # shape (batch_size,)
        coords = batch[0][:, 1:].to(device)  # shape (batch_size, 3)
        sdf_gt = batch[1].to(device)  # shape (batch_size, 1)

        return coords, sdf_gt, shape_ids

    def train(self, train_loader, debug = False):
        total_loss = 0.0
        iterations = 0.0

        if False:
            start = time.time()
            for i, batch in enumerate(train_loader):
                if i == 100:
                    break
            print("100 batches load time:", time.time() - start)

        self.model.train()
        for i, batch in enumerate(train_loader, start=1):
            # batch[0]: [class, x, y, z], shape: (batch_size, 4)
            # batch[1]: [sdf], shape: (batch size)
            with torch.autograd.profiler.profile(use_cuda=False) as prof:
                if debug:
                    batch_time = time.time()
                if iterations % 250 == 0:
                    print(f"Iteration: {iterations}; Training batch {i}/{len(train_loader)}")

                iterations += 1.0
                self.optimizer_model.zero_grad()
                #self.optimizer_latent.zero_grad()

                #x, y, latent_codes_indices_batch, latent_codes_batch = self.generate_xy(batch)
                #predictions = self.model(x)  # (batch_size, 1)
                if debug:
                    start = time.time()
                coords, sdf_gt, shape_ids = self.generate_xy_triplane(batch)

                if debug:
                    print("generate_xy time:", time.time() - start)

                if debug:
                    forward = time.time()
                predictions = self.model(coords=coords, shape_ids=shape_ids)
                if debug:
                    print("forward time:", time.time() - forward)

                if self.train_cfg['clamp']:
                    predictions = torch.clamp(predictions, -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])

                #loss_value, loss_rec, loss_latent = self.train_cfg['loss_multiplier'] * SDFLoss_multishape(y, predictions, x[:, :self.train_cfg['latent_size']], sigma=self.train_cfg['sigma_regulariser'])
                #loss_value, loss_rec, _ = self.train_cfg['loss_multiplier'] * SDFLoss_multishape(y, predictions, x[:, :self.train_cfg['latent_size']], sigma=self.train_cfg['sigma_regulariser'])
                loss_value, loss_rec, loss_reg = self.train_cfg['loss_multiplier'] * SDFLoss_triplane_multishape(
                    sdf_gt, predictions,
                    self.model.xy_planes[shape_ids],
                    self.model.yz_planes[shape_ids],
                    self.model.zx_planes[shape_ids],
                    sigma=self.train_cfg['sigma_regulariser']
                )

                #decorrelation_loss = triplane_decorrelation_loss(self.model.xy_planes[shape_ids],
                #                                                 self.model.yz_planes[shape_ids],
                #                                                 self.model.zx_planes[shape_ids], num_subset_channels=4)

                #loss_value += self.lambda_ * decorrelation_loss
                if debug:
                    backward = time.time()
                loss_value.backward()
                if debug:
                    print("backward time:", time.time() - backward)

                #self.optimizer_latent.step()
                if debug:
                    optimizer_step_time = time.time()
                self.optimizer_model.step()
                if debug:
                    print("optimizer_step time:", time.time() - optimizer_step_time)
                total_loss += loss_value.data.cpu().numpy()

                if debug:
                    print("batch time:", time.time() - batch_time)
                print(prof.key_averages().table(sort_by="cuda_time_total"))

        avg_train_loss = total_loss/iterations
        print(f'Training: loss {avg_train_loss}')
        self.writer.add_scalar('Training loss', avg_train_loss, self.epoch)

        return avg_train_loss

    def validate(self, val_loader):
        total_loss = 0.0
        total_loss_rec = 0.0
        #total_loss_latent = 0.0
        iterations = 0.0
        self.model.eval()

        for batch in val_loader:
            # batch[0]: [class, x, y, z], shape: (batch_size, 4)
            # batch[1]: [sdf], shape: (batch size)
            iterations += 1.0

            if iterations % 250 == 0:
                print(f"Iteration: {iterations}; Validation batch {iterations}/{len(val_loader)}")

            # x, y, latent_codes_indices_batch, latent_codes_batch = self.generate_xy(batch)
            # predictions = self.model(x)  # (batch_size, 1)

            coords, sdf_gt, shape_ids = self.generate_xy_triplane(batch)
            predictions = self.model(coords=coords, shape_ids=shape_ids)

            if train_cfg['clamp']:
                predictions = torch.clamp(predictions, -train_cfg['clamp_value'], train_cfg['clamp_value'])


            #loss_value, loss_rec, loss_latent = self.train_cfg['loss_multiplier'] * SDFLoss_multishape(y, predictions, latent_codes_batch, self.train_cfg['sigma_regulariser'])
            # loss_value, loss_rec, _ = self.train_cfg['loss_multiplier'] * SDFLoss_multishape(y, predictions, x[:, :self.train_cfg['latent_size']], sigma=self.train_cfg['sigma_regulariser'])
            loss_value, loss_rec, loss_reg = self.train_cfg['loss_multiplier'] * SDFLoss_triplane_multishape(
                sdf_gt, predictions,
                self.model.xy_planes[shape_ids],
                self.model.yz_planes[shape_ids],
                self.model.zx_planes[shape_ids],
                sigma=self.train_cfg['sigma_regulariser']
            )

            #decorrelation_loss = triplane_decorrelation_loss(self.model.xy_planes[shape_ids],
             #                                                self.model.yz_planes[shape_ids],
              #                                               self.model.zx_planes[shape_ids], num_subset_channels=4)

            #loss_value += self.lambda_ * decorrelation_loss

            total_loss += loss_value.data.cpu().numpy()
            total_loss_rec += loss_rec.data.cpu().numpy() 
            #total_loss_latent += loss_latent.data.cpu().numpy()

        avg_val_loss = total_loss/iterations
        avg_loss_rec = total_loss_rec/iterations
        #avg_loss_latent = total_loss_latent/iterations
        print(f'Validation: loss {avg_val_loss}')
        self.writer.add_scalar('Validation loss', avg_val_loss, self.epoch)
        self.writer.add_scalar('Reconstruction loss', avg_loss_rec, self.epoch)
        self.writer.add_scalar('Regularization loss', loss_reg.item(), self.epoch)
       # self.writer.add_scalar('Decorrelation loss', decorrelation_loss.item(), self.epoch)
        #self.writer.add_scalar('Latent code loss', avg_loss_latent, self.epoch)

        return avg_val_loss

if __name__=='__main__':


    train_cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'train_sdf.yaml')
    with open(train_cfg_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(train_cfg)
    trainer()