import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils.utils_deepsdf import SDFLoss_triplane_multishape


class ResidualConvBlock(nn.Module):
        def __init__(self, channels, bottleneck_ratio=0.5):
            super().__init__()
            hidden = int(channels * bottleneck_ratio)
            self.block = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, channels, kernel_size=1)
            )
        def forward(self, x):
            return x + self.block(x)
        
        

class TriPlaneSDFModel(nn.Module):
    def __init__(self, num_shapes, plane_feat_dim=32, plane_res=256,
                 num_layers=8, skip_connections=True, inner_dim=256, output_dim=1, mode=0, device=None):# mode 0 for base, 1 for fusion-mlp, 
                                                                                            # 2 for plane-conv, 3 for resnet
        super().__init__()

        self.plane_feat_dim = plane_feat_dim
        self.plane_res = plane_res
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
        self.mode = mode
        self.device=device

        # Each shape gets its own set of tri-planes
        #self.xy_planes = nn.Parameter(torch.randn(num_shapes, plane_feat_dim, plane_res, plane_res) * 0.0001)
        #self.yz_planes = nn.Parameter(torch.randn(num_shapes, plane_feat_dim, plane_res, plane_res) * 0.0001)
        #self.zx_planes = nn.Parameter(torch.randn(num_shapes, plane_feat_dim, plane_res, plane_res) * 0.0001)

        #self.input_dim_pre_fusion = 3 * plane_feat_dim
        input_dim = 3 * plane_feat_dim + 3
        self.skip_tensor_dim = copy.deepcopy(input_dim)

        layers = []
        
        match mode:
            case 1:
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU()
                )
            case 2:
                self.plane_conv = nn.Sequential(
                    nn.Conv2d(plane_feat_dim, plane_feat_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(plane_feat_dim, plane_feat_dim, kernel_size=1)
                )
            case 3: 
                self.plane_encoder = ResidualConvBlock(plane_feat_dim)
            case 4:
                self.patch_cnn = nn.Sequential(
                    nn.Conv2d(plane_feat_dim, plane_feat_dim, 3, padding=1),  # 3×3 conv
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1)  # reduce to (B, C, 1, 1)
                )

        
        #self.attn_mlp = nn.Sequential(
        #    nn.Linear(3 * self.plane_feat_dim, self.plane_feat_dim),
        #    nn.ReLU(),
        #    nn.Linear(self.plane_feat_dim, 3)  # Attention logits for 3 planes
        #)
       
        
        for i in range(num_layers - 2):  # leave space for skip + final
            layers.append(nn.Sequential(
                #nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)),
                nn.Linear(input_dim, inner_dim),#nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)),
                nn.ReLU()))
            input_dim = inner_dim

        self.net = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(
            nn.Linear(inner_dim, output_dim),
            nn.Tanh()
        )

        # For the skip connection layer
        if self.skip_connections and num_layers >= 5:
            self.skip_layer = nn.Sequential(
                nn.Linear(inner_dim, inner_dim - self.skip_tensor_dim),
                nn.ReLU()
            )

    def sample_triplane_features(self, coords, xy_plane, yz_plane, zx_plane):
        """
        Sample tri-plane features for a single shape (used in inference).

        Args:
            coords: (B, 3)
            xy_plane, yz_plane, zx_plane: (1, C, H, W)

        Returns:
            features: (B, C)
        """
        B = coords.shape[0]

        # Normalize from [-1, 1] to [-1, 1] for grid_sample
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        xy_coords = torch.stack([x, y], dim=-1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 2)
        yz_coords = torch.stack([y, z], dim=-1).unsqueeze(1).unsqueeze(1)
        zx_coords = torch.stack([z, x], dim=-1).unsqueeze(1).unsqueeze(1)


        # Repeat planes to match B for grid_sample
        xy_plane = xy_plane.expand(B, -1, -1, -1)  # (B, C, H, W)
        yz_plane = yz_plane.expand(B, -1, -1, -1)
        zx_plane = zx_plane.expand(B, -1, -1, -1)

        # Sample features
        xy_feats = F.grid_sample(xy_plane, xy_coords, align_corners=True).squeeze(-1).squeeze(-1)
        yz_feats = F.grid_sample(yz_plane, yz_coords, align_corners=True).squeeze(-1).squeeze(-1)
        zx_feats = F.grid_sample(zx_plane, zx_coords, align_corners=True).squeeze(-1).squeeze(-1)

        triplane_feats = torch.cat([xy_feats, yz_feats, zx_feats], dim=-1)  # (B, 3*C)

        return torch.cat([triplane_feats, coords], dim=-1) # (B, 3*C + 3)
    
    
    def sample_windowed_triplane_features(self, coords, xy_plane, yz_plane, zx_plane):
        """
        Sample local-window tri-plane features for a batch of coordinates.

        Args:
            coords: (B, 3), in [-1, 1] cube
            xy_plane, yz_plane, zx_plane: (1, C, H, W)
            res: spatial resolution of the plane

        Returns:
            features: (B, C * 3 + 3) → fused tri-plane features + coords
        """
        B = coords.shape[0]
        C = xy_plane.shape[1]  # number of channels

        # Normalize coordinates (already in [-1, 1], assumed)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        xy_coords = torch.stack([x, y], dim=-1)  # (B, 2)
        yz_coords = torch.stack([y, z], dim=-1)
        zx_coords = torch.stack([z, x], dim=-1)

        def sample_patch(plane, coords_2d):
            """
            Sample 3x3 patch per point from a tri-plane.

            Args:
                plane: (C, H, W)
                coords_2d: (B, 2) in [-1, 1]

            Returns:
                (B, C * 9) flattened patch
            """
            B = coords_2d.shape[0]
            C = self.plane_feat_dim

            if plane.dim() == 3:
                plane = plane.unsqueeze(0)  # (1, C, H, W)

            # 3x3 offsets in normalized coords
            offsets = torch.tensor([
                [-1, -1], [0, -1], [1, -1],
                [-1,  0], [0,  0], [1,  0],
                [-1,  1], [0,  1], [1,  1],
            ], device=self.device).float() / (self.plane_res / 2)

            grid = coords_2d.unsqueeze(1) + offsets.unsqueeze(0)  # (B, 9, 2)
            grid = grid.view(1, B * 9, 1, 2)  # (1, B*9, 1, 2)

            # Sample
            sampled = F.grid_sample(plane, grid, align_corners=True, mode="bilinear")  # (1, C, 1, B*9)

            # Reshape correctly
            sampled = sampled.view(1, C, B, 9)      # (1, C, B, 9)
            sampled = sampled.squeeze(0).permute(1, 2, 0)  # → (C, B, 9) → (B, C, 9)
            sampled = sampled.reshape(B, C * 9)       # flatten patch → (B, C×9)

            return sampled


        # Sample each plane
        xy_feat = sample_patch(xy_plane, xy_coords)
        yz_feat = sample_patch(yz_plane, yz_coords)
        zx_feat = sample_patch(zx_plane, zx_coords)
        
        # From [B, C×9] → [B, C, 3, 3]
        xy_feat = xy_feat.view(-1, self.plane_feat_dim, 3, 3)
        yz_feat = yz_feat.view(-1, self.plane_feat_dim, 3, 3)
        zx_feat = zx_feat.view(-1, self.plane_feat_dim, 3, 3)
        
        # Optional: reduce dimensionality
        # self.patch_fc: nn.Linear(C * 9, C)
        xy_proj = self.patch_cnn(xy_feat).squeeze(-1).squeeze(-1)  # [B, 32]
        yz_proj = self.patch_cnn(yz_feat).squeeze(-1).squeeze(-1)  # [B, 32]
        zx_proj = self.patch_cnn(zx_feat).squeeze(-1).squeeze(-1)  # [B, 32]

        triplane_feats = torch.cat([xy_proj, yz_proj, zx_proj], dim=-1)  # (B, 3*C)

        return torch.cat([triplane_feats, coords], dim=-1)  # (B, 3*C + 3) 3)

    
    
    def sample_quadrant_masked_plane_features(self, coords, xy_plane, yz_plane, zx_plane):
        """
        Quadrant-aware sampling from tri-planes with channel masking and zero-fill.

        Args:
            coords: (B, 3)
            xy_plane, yz_plane, zx_plane: (1, C, H, W)

        Returns:
            features: (B, 3*C)
        """

        def sample_masked(plane, coord2d, debug=False):
            B, C, H, W = plane.shape
            assert C % 4 == 0, "Channel count must be divisible by 4"
            c_per_quad = C // 4
            out_feats = torch.zeros((coord2d.shape[0], C), device=coords.device)


            # Normalize coords from [-1, 1] → [0, 1]
            norm_coords = (coord2d + 1) / 2
            quad_x = (norm_coords[:, 0] >= 0.5).long()  # 0=left, 1=right
            quad_y = (norm_coords[:, 1] >= 0.5).long()  # 0=top, 1=bottom
            quadrant = quad_y * 2 + quad_x  # 0=TL, 1=TR, 2=BL, 3=BR

            torch.cuda.synchronize()

            
            for q in range(4):
                if debug:
                    print(f"[Debug] quadrant shape: {quadrant.shape}, dtype: {quadrant.dtype}, min: {quadrant.min()}, max: {quadrant.max()}")
                    print(f"[Debug] q: {q}")

                idxs = (quadrant == q).nonzero(as_tuple=True)[0]
                if idxs.numel() == 0:
                    continue

                coords_q = coord2d[idxs].unsqueeze(1).unsqueeze(1)  # (Nq, 1, 1, 2)
                
                coords_q_raw = coord2d[idxs]
                if torch.isnan(coords_q_raw).any():
                    print(f"[!] NaNs in coords_q_raw for quadrant {q}, aborting.")
                    print(coords_q_raw)
                    raise ValueError("NaNs in coords")

                # 1. Get coordinates in [0, 1]
                coords_q_raw = coord2d[idxs]
                coords_01 = (coords_q_raw + 1) / 2  # [B, 2]

                # 2. Extract only the quadrant portion and stretch to [0, 1]
                # Each quadrant spans a 0.5×0.5 region:
                if q == 0:  # top-left
                    coords_qr = coords_01 * 2
                elif q == 1:  # top-right
                    coords_qr = torch.stack([(coords_01[:, 0] - 0.5) * 2, coords_01[:, 1] * 2], dim=-1)
                elif q == 2:  # bottom-left
                    coords_qr = torch.stack([coords_01[:, 0] * 2, (coords_01[:, 1] - 0.5) * 2], dim=-1)
                elif q == 3:  # bottom-right
                    coords_qr = (coords_01 - 0.5) * 2

                # 3. Map back to [-1, 1] and clamp
                coords_q = coords_qr * 2 - 1
                coords_q = coords_q.clamp(-1 + 1e-4, 1 - 1e-4).unsqueeze(1).unsqueeze(1)
                
                if torch.any(coords_q.isnan()) or torch.any(coords_q > 1) or torch.any(coords_q < -1):
                    print(f"[!] Invalid coords_q in quadrant {q}: min={coords_q.min()}, max={coords_q.max()}, shape={coords_q.shape}")
                
                start = q * c_per_quad
                end = start + c_per_quad

                # Narrow to quadrant channels only
                plane_q = plane[:, start:end, :, :] # (1, C/4, H, W)
                if plane_q.shape[1] != c_per_quad:
                    print(f"[!] Unexpected plane_q shape in quadrant {q}: {plane_q.shape}")
                
                plane_q = plane_q.expand(len(idxs), -1, -1, -1)  # (Nq, C//4, H, W)
                
                
                try:
                    feats_q = F.grid_sample(plane_q, coords_q, align_corners=True).squeeze(-1).squeeze(-1)
                except Exception as e:
                    print(f"[!] grid_sample error in quadrant {q}")
                    print(f"plane_q shape: {plane_q.shape}, coords_q shape: {coords_q.shape}")
                    raise e

                if feats_q.shape[1] != (end - start):
                    print(f"[!] Feature shape mismatch: feats_q={feats_q.shape}, expected={end - start}")

                try:
                    if debug:
                        print(f"[Debug] Assigning to out_feats[{idxs.shape}, {start}:{end}]")
                        print(f"[Debug] feats_q.shape = {feats_q.shape}, out_feats.shape = {out_feats.shape}")
                        print(f"[Debug] Expected feats_q.shape = ({len(idxs)}, {end - start})")
                        print(f"Devices: feats_q={feats_q.device}, out_feats={out_feats.device}, idxs={idxs.device}")
                    assert feats_q.shape == (len(idxs), end - start), "Mismatch in shape before assignment"

                    out_feats[idxs, start:end] = feats_q
                except Exception as e:
                    print(f"[!] Assignment error: out_feats[{idxs.shape}, {start}:{end}] = feats_q {feats_q.shape}")
                    raise e

            return out_feats

        # Build 2D coords per plane
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        xy_coords = torch.stack([x, y], dim=-1)
        yz_coords = torch.stack([y, z], dim=-1)
        zx_coords = torch.stack([z, x], dim=-1)

        xy_feats = sample_masked(xy_plane, xy_coords)
        yz_feats = sample_masked(yz_plane, yz_coords)
        zx_feats = sample_masked(zx_plane, zx_coords)

        return torch.cat([xy_feats, yz_feats, zx_feats], dim=-1)  # (B, 3*C)

    
    def conv_with_residual(self, x):
        if self.mode == 3:
            return self.plane_encoder(x)    
        return x + self.plane_conv(x)
        

    def forward(self, coords, xy_plane=None, yz_plane=None, zx_plane=None, shape_ids=None, epoch=0):
        """
        Supports two modes:
        1. Training: passes shape_ids to lookup per-sample tri-planes (batch)
        2. Inference: passes explicit xy/yz/zx_plane (single shape)

        Args:
            coords: (B, 3)
            xy_plane, yz_plane, zx_plane: (1, C, H, W) tri-planes for one shape
            shape_ids: (B,) — used during training

        Returns:
            sdf: (B, 1)
        """
        if xy_plane is not None and yz_plane is not None and zx_plane is not None:
            # Inference mode — reconstructing one shape
            
            if self.mode == 2 or self.mode == 3:
                xy = self.conv_with_residual(xy_plane)
                yz = self.conv_with_residual(yz_plane)
                zx = self.conv_with_residual(zx_plane)
                feats = self.sample_triplane_features(coords, xy, yz, zx) #for convolution
            #elif self.mode == 4:
             #   feats = self.sample_windowed_triplane_features(coords, xy_plane, yz_plane, zx_plane)
            else:
                feats = self.sample_triplane_features(coords, xy_plane, yz_plane, zx_plane)
            
            #feats = self.sample_quadrant_masked_plane_features(coords, xy_plane, yz_plane, zx_plane)
        elif shape_ids is not None and False:
            # Training mode — batch of shapes
            if self.mode == 2 or self.mode == 3:
                xy = self.conv_with_residual(self.xy_planes[shape_ids])
                yz = self.conv_with_residual(self.yz_planes[shape_ids])
                zx = self.conv_with_residual(self.zx_planes[shape_ids])
                feats = self.sample_triplane_features(coords, xy, yz, zx)
            else:
                feats = self.sample_triplane_features(coords, self.xy_planes[shape_ids], self.yz_planes[shape_ids],
                                                       self.zx_planes[shape_ids])
            
            #sid = shape_ids.item()  # scalar index of the shape
            #xy = self.xy_planes[sid].unsqueeze(0)
            #yz = self.yz_planes[sid].unsqueeze(0)
            #zx = self.zx_planes[sid].unsqueeze(0)

            #feats = self.sample_quadrant_masked_plane_features(coords, xy, yz, zx)
        else:
            raise ValueError("Either (xy_plane, yz_plane, zx_plane) or shape_ids must be provided.")

        
        if self.mode == 1:
            feats = self.fusion_mlp(feats)
        
        ########
        #xy_feats, yz_feats, zx_feats = torch.chunk(feats, 3, dim=-1)

        # Stack into (B, 3, C)
        #stacked_feats = torch.stack([xy_feats, yz_feats, zx_feats], dim=1)  # for weighted sum later

        # Compute attention weights: (B, 3)
        #attn_logits = self.attn_mlp(feats)  # input is (B, 3C)
        #attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (B, 3, 1)
        
        # Fuse via attention-weighted sum: (B, C)
        #feats = torch.sum(attn_weights * stacked_feats, dim=1)
        ########
        
        #if epoch > 40:
         #   print("Feats max:", feats.abs().max().item())

        feats = torch.clamp(feats, -10, 10)

        
        input_feats = feats.clone().detach()

        # Dynamic forward pass with skip connections
        if self.skip_connections and self.num_layers >= 5:
            for i in range(3):
                feats = self.net[i](feats)
            feats = self.skip_layer(feats)
            feats = torch.hstack((feats, input_feats))
            for i in range(3, self.num_layers - 2):
                feats = self.net[i](feats)
        else:
            if self.skip_connections:
                print("[Warning] At least 5 layers are required to enable skip connections.")
            feats = self.net(feats)

        sdf = self.final_layer(feats)
        return sdf#, attn_weights
    
    
    
    def infer_triplane_code(self, cfg, pointcloud, sdf_gt, device=None, writer=None, xy_init=None, yz_init=None, zx_init=None):
        """
        Infer tri-plane codes from a partial point cloud (coords + sdf_gt).
        Returns optimized tri-plane tensors: xy, yz, zx
        """
        
        def init_plane():
            return (0.0001 * torch.randn(1, self.plane_feat_dim, self.plane_res, self.plane_res, device=device)).requires_grad_()

        B = pointcloud.shape[0]
        pc_tensor = pointcloud.to(device)
        sdf_gt = sdf_gt.to(device)

        #xy = xy_init.clone().detach().to(device).requires_grad_() if xy_init is not None else init_plane()
        #yz = yz_init.clone().detach().to(device).requires_grad_() if yz_init is not None else init_plane()
        #zx = zx_init.clone().detach().to(device).requires_grad_() if zx_init is not None else init_plane()

        #xy = init_plane()
        #yz = init_plane()
        #zx = init_plane()
        
        xy = (1.0*xy_init + 0.0 * torch.randn_like(xy_init)).detach().clone().requires_grad_()
        yz = (1.0*yz_init + 0.0 * torch.randn_like(yz_init)).detach().clone().requires_grad_()
        zx = (1.0*zx_init + 0.0 * torch.randn_like(zx_init)).detach().clone().requires_grad_()

        optimizer = torch.optim.Adam([xy, yz, zx], lr=cfg['lr'])

        if cfg.get('lr_scheduler', False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                factor=cfg['lr_multiplier'],
                                                                patience=cfg['patience'],
                                                                threshold=0.001,
                                                                threshold_mode='rel')
        best_loss = float('inf')
        best_planes = (xy, yz, zx)
        
        history = {"xy": [], "yz": [], "zx": []}

        for epoch in range(cfg['epochs']):
            
            optimizer.zero_grad()
            sdf_pred = self(coords=pc_tensor, xy_plane=xy, yz_plane=yz, zx_plane=zx)

            if cfg['clamp']:
                sdf_pred = torch.clamp(sdf_pred, -cfg['clamp_value'], cfg['clamp_value'])
              
            loss, loss_rec, loss_reg = SDFLoss_triplane_multishape(
                sdf_gt, sdf_pred,
                xy,
                yz, 
                zx,
                sigma=cfg['sigma_regulariser']
            )
            
            if epoch%100 == 0:
                print(f"[Debug] Epoch {epoch} | Loss: {loss}")
                print(f"[Debug] Epoch {epoch} | SDF min: {sdf_pred.min().item():.4f}, max: {sdf_pred.max().item():.4f}")
                
                #print(f"xy.requires_grad: {xy.requires_grad}, is_leaf: {xy.is_leaf}")
                #print(f"grad before: {xy.grad}")s
            
            #l2_plane_reg = (xy - xy_init).pow(2).mean() + \
             #           (yz - yz_init).pow(2).mean() + \
              #          (zx - zx_init).pow(2).mean()

            #loss += 0.01 * l2_plane_reg


            

            loss.backward()
            
            optimizer.step()

            if writer:
                writer.add_scalar("Reconstruction loss", loss.item(), epoch)
                
            history["xy"].append(xy.detach().cpu().clone())
            history["yz"].append(yz.detach().cpu().clone())
            history["zx"].append(zx.detach().cpu().clone())

            if loss_rec.item() < best_loss:
                best_loss = loss_rec.item()
                best_planes = (xy.clone(), yz.clone(), zx.clone())

            if cfg.get('lr_scheduler', False):
                scheduler.step(loss.item())
                if writer:
                    writer.add_scalar('Learning rate', scheduler._last_lr[0], epoch)

                if scheduler._last_lr[0] < 1e-8:
                    print('[Stopping] LR too low')
                    break 
            if epoch > 70000: # tested noise injection  - unsucessfully
                xy = xy + 0.0001 * torch.randn(1, self.plane_feat_dim, self.plane_res, self.plane_res, device=device) * (1-epoch/cfg['epochs'])
                yz = yz + 0.0001 * torch.randn(1, self.plane_feat_dim, self.plane_res, self.plane_res, device=device) * (1-epoch/cfg['epochs'])
                zx = zx + 0.0001 * torch.randn(1, self.plane_feat_dim, self.plane_res, self.plane_res, device=device) * (1-epoch/cfg['epochs'])
            
        return best_planes, history


