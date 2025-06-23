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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_lang import render
import torchvision
import numpy as np
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from initc import GaussianModel
from imgviz import label_colormap
from utils.label_util import convert_labels_with_palette,get_palette

def render_set(dataset, opt, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "gt")

    render_npy_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "gt_npy")

    point3d_npy_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)

    if 'scannet' in dataset.source_path:
        scene_name = 'scannet'
    elif 'replica' in dataset.source_path:
        scene_name = 'replica'
    
    palette = get_palette(colormap=scene_name)


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        latent_feat=view.get_latent_feature(dataset.source_path)
        # output=render(view, gaussians, pipeline, background,latent_feat=latent_feat)
        output= render(view, gaussians, pipeline, background,opt)
       
        rendering_ft = output["language_feature_image"]

        rendering = output['render']
        # coords = output['point_3dprecomp']
        # colors = output['colors_3dprecomp']

        # pred_3d = 

        gt = view.original_image[0:3, :, :] # (C,H,W)
        langfeat_path=os.path.join(dataset.source_path, dataset.language_features_name, dataset.feature_extractor_2d) #, dataset.2d_feature_extractor)
        gt_ft = view.get_language_feature(langfeat_path)


        # torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name+ ".png")) #'{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name+ ".png")) #'{0:05d}'.format(idx) + ".png"))
        # import pdb
        # pdb.set_trace()
        sem_dim,h,w=rendering_ft.shape
     
        object_logits = torch.matmul(rendering_ft.permute(1,2,0).reshape(-1,sem_dim),latent_feat.t())
        pre_label = object_logits.max(1)[1].cpu()
        # object_sem= torch.from_numpy(np.transpose(label_colormap()[pre_label.numpy()]/255.0,(1,0)).reshape(3,h,w))
        pred_label_color = convert_labels_with_palette(pre_label.numpy(), palette) #(N,3)

        gt_object_logits = torch.matmul(gt_ft.permute(1,2,0).reshape(-1,sem_dim),latent_feat.t())
        gt_label = gt_object_logits.max(1)[1].cpu()
        # object_sem_gt = torch.from_numpy(np.transpose(label_colormap()[gt_label.cpu().numpy()]/255.0,(1,0)).reshape(3,h,w))
        gt_label_color = convert_labels_with_palette(gt_label.numpy(), palette) #(N,3)



        torchvision.utils.save_image(torch.from_numpy(pred_label_color.reshape((h,w,-1))).permute(2,0,1), os.path.join(render_npy_path, view.image_name+ ".png"))
        torchvision.utils.save_image(torch.from_numpy(gt_label_color.reshape((h,w,-1))).permute(2,0,1), os.path.join(gts_npy_path, view.image_name+ ".png"))
        # np.save(os.path.join(render_npy_path, view.image_name+ ".npy"),rendering_ft.permute(1,2,0).reshape(-1,sem_dim).cpu().numpy()) #'{0:05d}'.format(idx) + ".npy"
        # np.save(os.path.join(gts_npy_path, view.image_name+ ".npy"),gt_ft.permute(1,2,0).reshape(-1,sem_dim).cpu().numpy()) # '{0:05d}'.format(idx) + ".npy"
        # torch.save((coords, colors, semantic_3D),os.path.join(point3d_npy_path,  "point_trained_3d" + '.pth'))


def render_sets(dataset : ModelParams,opt:OptimizationParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        # gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians,  shuffle=False) #,load_iteration=iteration)
        # opt.train_feature_only =True
        # checkpoint = os.path.join(dataset.model_path, 'chkpnt60000.pth')
        
        # # import pdb
        # # pdb.set_trace()
        # (model_params, _) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt, mode='test')

        checkpoint = os.path.join(dataset.model_path, 'chkpnt62000.pth')

        (model_params, _) = torch.load(checkpoint)
        gaussians = GaussianModel(dataset.sh_degree)
        opt.train_feature_only =True
        gaussians.restore(model_params, opt, mode='test')
        
        scene = Scene(dataset, gaussians,  shuffle=False,load_iteration=iteration)


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset, opt, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset, opt, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args),op.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)