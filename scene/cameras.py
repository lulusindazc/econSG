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
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, make_intrinsic, fov2focal
from utils.loss_utils import image2canny

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, depth=None, label= None,conf_cof = None, diff_conf_cof = None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name  # without extension

        # print(image_name)
        # import pdb
        # pdb.set_trace()
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.canny_mask = image2canny(self.original_image.permute(1,2,0), 50, 150, isEdge1=False).detach().to(self.data_device)
        if depth is not None:
            self.original_depth = depth.to(self.data_device)
        else:
            self.original_depth = None
        if label is not None:
            self.original_label= label.to(self.data_device)
        else:
            self.original_label = None
        
        if conf_cof is not None:
            self.original_conf_cof = conf_cof.to(self.data_device)
        else:
            self.original_conf_cof = None

        if diff_conf_cof is not None:
            self.original_diff_conf_cof = diff_conf_cof.to(self.data_device)
        else:
            self.original_diff_conf_cof = None

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            # self.depth *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            # self.depth *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.focal_x= fov2focal(self.FoVx, self.image_width)
        self.focal_y = fov2focal(self.FoVy, self.image_height)
        intrinsics= make_intrinsic(self.focal_x, self.focal_y, (self.image_width-1)*0.5, (self.image_height-1)*0.5)
        self.intrinsics=torch.from_numpy(intrinsics).float().transpose(0,1).cuda()
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # self.projection_matrix=intrinsics
        self.full_proj_transform_2d= (self.world_view_transform.unsqueeze(0).bmm(self.intrinsics.unsqueeze(0))).squeeze(0)
        # import pdb
        # pdb.set_trace()

    def get_language_feature(self, language_feature_dir, out_latent=False):
        language_feature_name = os.path.join(language_feature_dir, self.image_name)
        feature_map = torch.from_numpy(np.load(language_feature_name + '.npy'))
        def norm_feat(input):
            # (N,3)
            return input / (input.norm(dim=-1, keepdim=True) + 1e-9)
        feature_map_n = norm_feat(feature_map)
        # import pdb
        # pdb.set_trace()
        point_feature = feature_map_n.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1) # (C,H,W)

        # latent_file =os.path.join(language_feature_dir,'text_feat_dim3_norm.pth')
        # if os.path.exists(latent_file) and out_latent:
        #     latent_features = torch.load(latent_file)
        #     return
        return point_feature.cuda() #, mask.cuda()
    
    def get_latent_feature(self, latent_feature_dir):
        return torch.load(os.path.join(latent_feature_dir,'text_feat_dim3_norm.pth')).cuda()


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

