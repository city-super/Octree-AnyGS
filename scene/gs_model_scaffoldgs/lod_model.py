#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
import torch
import math
import numpy as np
from torch import nn
from einops import repeat
from functools import reduce
from torch_scatter import scatter_max
from utils.general_utils import get_expon_lr_func, knn
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from scene.embedding import Embedding
from scene.basic_model import BasicModel
    
class GaussianLoDModel(BasicModel):

    def __init__(self, **model_kwargs):

        for key, value in model_kwargs.items():
            setattr(self, key, value)
        
        self._anchor = torch.empty(0)
        self._level = torch.empty(0)
        self._extra_level = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
                
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions()
    
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(self.view_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()
            
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()
        
        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
        ).cuda()
    
        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim+self.appearance_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()

    def capture(self):
        param_dict = {}
        param_dict['optimizer'] = self.optimizer.state_dict()
        param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
        param_dict['cov_mlp'] = self.mlp_cov.state_dict()
        param_dict['color_mlp'] = self.mlp_color.state_dict()
        if self.use_feat_bank:
            param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
        if self.appearance_dim > 0:
            param_dict['appearance'] = self.embedding_appearance.state_dict()
        return (
            self.voxel_size,
            self.standard_dist,
            self._anchor,
            self._level,
            self._extra_level,
            self._offset,
            self._scaling,
            self._rotation,
            self.opacity_accum, 
            self.anchor_demon,
            self.offset_gradient_accum,
            self.offset_denom,
            param_dict,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.voxel_size,
        self.standard_dist,
        self._anchor,
        self._level,
        self._extra_level,
        self._offset,
        self._scaling,
        self._rotation,
        self.opacity_accum, 
        self.anchor_demon,
        self.offset_gradient_accum,
        self.offset_denom,
        param_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(param_dict['optimizer'])
        self.mlp_opacity.load_state_dict(param_dict['opacity_mlp'])
        self.mlp_cov.load_state_dict(param_dict['cov_mlp'])
        self.mlp_color.load_state_dict(param_dict['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(param_dict['feature_bank_mlp'])
        if self.appearance_dim > 0:
            self.embedding_appearance.load_state_dict(param_dict['appearance'])

    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def get_level(self):
        return self._level
    
    @property
    def get_extra_level(self):
        return self._extra_level
        
    @property
    def get_anchor_feat(self):
        return self._anchor_feat

    @property
    def get_offset(self):
        return self._offset
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()
        else:
            self.embedding_appearance = None

    @property
    def get_appearance(self):
        return self.embedding_appearance
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity   

    @property
    def get_cov_mlp(self):
        return self.mlp_cov
    
    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    def set_coarse_interval(self, opt):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1/opt.coarse_factor
            a1 = opt.coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_level(self, points, cameras, scales):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                dist_max = torch.quantile(dist, self.dist_ratio)
                dist_min = torch.quantile(dist, 1 - self.dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, self.dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - self.dist_ratio)
        self.standard_dist = dist_max
        if self.levels == -1:
            self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
        if self.init_level == -1:
            self.init_level = int(self.levels/2)
            
    def octree_sample(self, data):
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda() 
        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            new_positions = torch.unique(torch.round((data - self.init_pos) / cur_size), dim=0) * cur_size + self.init_pos
            new_positions += self.padding * cur_size
            new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self._level = torch.concat((self._level, new_level), dim=0)
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def create_from_pcd(self, pcd, spatial_lr_scale, logger, *args):
        points = torch.tensor(pcd.points).float().cuda()
        self.set_level(points, *args)
        self.spatial_lr_scale = spatial_lr_scale
        box_min = torch.min(points)*self.extend
        box_max = torch.max(points)*self.extend
        box_d = box_max - box_min
        if self.base_layer < 0:
            default_voxel_size = 0.02
            self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()
        self.octree_sample(points)

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)
        
        logger.info(f'Branches of Tree: {self.fork}')
        logger.info(f'Base Layer of Tree: {self.base_layer}')
        logger.info(f'Visible Threshold: {self.visible_threshold}')
        logger.info(f'Appearance Embedding Dimension: {self.appearance_dim}')
        logger.info(f'LOD Levels: {self.levels}')
        logger.info(f'Initial Levels: {self.init_level}')
        logger.info(f'Initial Voxel Number: {self.positions.shape[0]}')
        logger.info(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
        logger.info(f'Max Voxel Size: {self.voxel_size}')

        fused_point_cloud = self.positions
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._level = self._level.unsqueeze(dim=1)
        self._extra_level = torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")

    def weed_out(self, anchor_positions, anchor_levels):
        visible_count = torch.zeros(anchor_positions.shape[0], dtype=torch.int, device="cuda")
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((anchor_positions - cam_center)**2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)   
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (anchor_levels <= int_level).int()
        visible_count = visible_count/len(self.cam_infos)
        weed_mask = (visible_count > self.visible_threshold)
        mean_visible = torch.mean(visible_count)
        return anchor_positions[weed_mask], anchor_levels[weed_mask], mean_visible, weed_mask

    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        dist = torch.sqrt(torch.sum((self.get_anchor - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        
        if self.progressive:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)

    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        dist = torch.sqrt(torch.sum((self.get_anchor - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)

    def training_setup(self, training_args):

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        ]
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})
        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
    def construct_list_of_attributes(self):
        l = []
        l.append('x')
        l.append('y')
        l.append('z')
        l.append('level')
        l.append('extra_level')
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, iteration):
        mkdir_p(os.path.dirname(path))

        if self.progressive:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        level_mask = (self._level <= coarse_index-1).squeeze(-1)
        anchor = self._anchor[level_mask].detach().cpu().numpy()
        levels = self._level[level_mask].detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1)[level_mask].detach().cpu().numpy()
        anchor_feats = self._anchor_feat[level_mask].detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous()[level_mask].cpu().numpy()
        scales = self._scaling[level_mask].detach().cpu().numpy()
        rots = self._rotation[level_mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, offsets, anchor_feats, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        plydata = PlyData([el], obj_info=[
            'standard_dist {:.6f}'.format(self.standard_dist),
            'levels {:.6f}'.format(self.levels),
            ])
        plydata.write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        infos = plydata.obj_info
        for info in infos:
            var_name = info.split(' ')[0]
            self.__dict__[var_name] = float(info.split(' ')[1])
    
        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int16)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.levels = round(self.levels)
        if self.init_level == -1:
            self.init_level = int(self.levels/2)

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._level = self._level[valid_points_mask]    
        self._extra_level = self._extra_level[valid_points_mask]

    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask, overlap):
        init_length = self.get_anchor.shape[0]
        grads[~offset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        for cur_level in range(self.levels):
            update_value = self.fork ** update_ratio
            level_mask = (self.get_level == cur_level).squeeze(dim=1)
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)
            if torch.sum(level_mask) == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            ds_size = cur_size / self.fork
            # update threshold
            cur_threshold = threshold * (update_value ** cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * extra_ratio
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat([candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)   
            
            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)
            candidate_extra_mask = torch.logical_and(candidate_extra_mask, level_mask)
            if ~self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level += extra_up * candidate_extra_mask.float()    

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)

            grid_coords = torch.round((self.get_anchor[level_mask]-self.init_pos)/cur_size - self.padding).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round((selected_xyz-self.init_pos)/cur_size - self.padding).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if overlap:
                remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            elif selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.zeros(selected_grid_coords_unique.shape[0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            grid_coords_ds = torch.round((self.get_anchor[level_ds_mask]-self.init_pos)/ds_size-self.padding).int()
            selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
            selected_grid_coords_ds = torch.round((selected_xyz_ds-self.init_pos)/ds_size-self.padding).int()
            selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                if overlap:
                    remove_duplicates_ds =  torch.ones(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device="cuda")
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                elif selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                    remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                    new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')

            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:
                
                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda()
                
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_feat_ds = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_ds_mask]
                new_feat_ds = scatter_max(new_feat_ds, inverse_indices_ds.unsqueeze(1).expand(-1, new_feat_ds.size(1)), dim=0)[0][remove_duplicates_ds]
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)
                
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).repeat([1,2]).float().cuda()*ds_size # *0.05
                new_scaling = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = torch.log(new_scaling)
                
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:,0] = 1.0

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets_ds = torch.zeros_like(candidate_anchor_ds).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)

                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device='cuda')
                new_extra_level_ds = torch.zeros(candidate_anchor_ds.shape[0], dtype=torch.float, device='cuda')
                new_extra_level = torch.cat([new_extra_level, new_extra_level_ds])
                
                d = {
                    "anchor": new_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                }   

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)
    
    def run_densify(self, iteration, opt):
        # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > opt.update_interval * opt.success_threshold * 0.5).squeeze(dim=1)
        
        self.anchor_growing(iteration, grads_norm, opt.densify_grad_threshold, opt.update_ratio, opt.extra_ratio, opt.extra_up, offset_mask, opt.overlap)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # prune anchors
        prune_mask = (self.opacity_accum < opt.min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > opt.update_interval * opt.success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

    def save_mlp_checkpoints(self, path):#split or unite
        mkdir_p(os.path.dirname(path))
        self.eval()
        opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
        cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
        color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim+self.appearance_dim).cuda()))
        color_mlp.save(os.path.join(path, 'color_mlp.pt'))
        if self.use_feat_bank:
            feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, self.view_dim).cuda()))
            feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
        if self.appearance_dim > 0:
            emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
            emd.save(os.path.join(path, 'embedding_appearance.pt'))
        self.train()

    def load_mlp_checkpoints(self, path):
        self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
        if self.use_feat_bank:
            self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
        if self.appearance_dim > 0:
            self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
    
    def generate_neural_gaussians(self, viewpoint_camera, visible_mask=None, ape_code=-1):
        # view frustum filtering for acceleration    
        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device = self.get_anchor.device)

        anchor = self.get_anchor[visible_mask]
        feat = self.get_anchor_feat[visible_mask]
        grid_offsets = self.get_offset[visible_mask]
        grid_scaling = self.get_scaling[visible_mask]

        # get view properties for anchor
        ob_view = anchor - viewpoint_camera.camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        ## view-adaptive feature
        if self.use_feat_bank:
            bank_weight = self.get_featurebank_mlp(ob_view).unsqueeze(dim=1) # [n, 1, 3]

            ## multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                feat[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1) # [n, c]

        cat_local_view = torch.cat([feat, ob_view], dim=1) # [N, c+3]

        if self.appearance_dim > 0:
            if ape_code < 0:
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
                appearance = self.get_appearance(camera_indicies)
            else:
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
                appearance = self.get_appearance(camera_indicies)
                
        # get offset's opacity
        neural_opacity = self.get_opacity_mlp(cat_local_view) # [N, k]

        if self.dist2level=="progressive":
            prog = self._prog_ratio[visible_mask]
            transition_mask = self.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            neural_opacity = neural_opacity * prog
        
        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0)
        mask = mask.view(-1)

        # select opacity 
        opacity = neural_opacity[mask]

        # get offset's color
        if self.appearance_dim > 0:
            color = self.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = self.get_color_mlp(cat_local_view)
        color = color.reshape([anchor.shape[0]*self.n_offsets, 3])# [mask]

        # get offset's cov
        scale_rot = self.get_cov_mlp(cat_local_view)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self.n_offsets, 7]) # [mask]
        
        # offsets
        offsets = grid_offsets.view([-1, 3]) # [mask]
        
        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
        
        # post-process cov
        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
        rot = self.rotation_activation(scale_rot[:,3:7])
        
        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets 
        
        return xyz, color, opacity, scaling, rot, None, mask