import os
import torch
import imageio
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from os.path import join, exists
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper, save_fused_feature
from scipy.special import softmax
import clip
import json
from imgviz import label_colormap
import cv2
import open3d as o3d
import plyfile
from replica_constants import REPLICA_EXISTING_CLASSES
import copy
import torchvision


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Replica.')
    parser.add_argument('--data_dir', type=str, default='..', help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, default='..',help='Where is the base logging directory')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--feature_2d_extractor', type=str, default='openseg',help='[lseg,openseg]')
    parser.add_argument('--scene_name', type=str, default='room0', help='which scene')
    parser.add_argument('--train_split', type=str, default='train', help='Where is the base logging directory')
    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args




def convert_labels_with_palette(input, palette):
    '''Get image color palette for visualizing masks'''

    new_3d = np.zeros((input.shape[0], 3))
    u_index = np.unique(input)
    # print(u_index)
    for index in u_index:
        if index == 255:
            index_ =  palette.shape[0]#20
        else:
            index_ = index
        
        cur_palette= np.array(
            [palette[index_ * 3],
             palette[index_ * 3 + 1],
             palette[index_ * 3 + 2]])
      
        new_3d[input==index] = cur_palette

    return new_3d


def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    scene_id = args.scene
    openseg_model = args.openseg_model
    text_emb = args.text_emb

    scene = join(args.data_root_2d, scene_id)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.text_features = args.text_features.to(device).to(torch.float32)
    text_feat_norm=  (args.text_features/(args.text_features.norm(dim=-1, keepdim=True)+1e-5))
    if args.train_split=='train':
        transfile=os.path.join(scene, 'transforms_train.json')
    else:
        transfile=os.path.join(scene, 'transforms_test.json')

    with open(transfile) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
 
        for idx, frame in enumerate(tqdm(frames)):
            cam_name = frame["file_path"] #os.path.join(path, frame["file_path"] + extension)
            img_dir = os.path.join(scene, cam_name)
            img_id = os.path.basename(cam_name).split('.')[0]
            

            feat_2d = extract_openseg_img_feature(img_dir, openseg_model, text_emb).to(device)
            feat_dim=feat_2d.shape[0]
            feat_2d_flat= (feat_2d.reshape(feat_dim,-1).t()).to(torch.float32) # (D,H*W)
            feat_2d_flat1 = (feat_2d_flat/(feat_2d_flat.norm(dim=-1, keepdim=True)+1e-5))
            outputs_t = torch.matmul(feat_2d_flat1.to(torch.float32), text_feat_norm.t()) #
            
            conf_t, pd_t = torch.max(outputs_t, 1) #torch.max(torch.softmax(outputs_t,axis=-1), 1)
           
            print('minimum conf value:{},max:{}'.format(np.min(conf_t.cpu().numpy()),np.max(conf_t.cpu().numpy())))
            vis_x,vis_y = np.where((conf_t.cpu().numpy() < 0.5 * np.max(conf_t.cpu().numpy())).reshape(args.img_h,args.img_w))
          
            pred_label_color = convert_labels_with_palette(pd_t.cpu().numpy(), args.palette)/255.0 #(N,3)
            
            semantic_img= (pd_t.reshape(args.img_h,args.img_w)).cpu().numpy()
            vis_img= pred_label_color.reshape((args.img_h,args.img_w,3))

            cv2.imwrite(os.path.join(out_dir, img_id+ ".png"), (pd_t.reshape(args.img_h,args.img_w)).cpu().numpy())
            # select confident predictions
            semantic_img[vis_x,vis_y] = 0.0
            cv2.imwrite(os.path.join(out_dir, img_id+ "_se.png"), semantic_img)

            cv2.imwrite(os.path.join(out_dir, img_id+ "_conf.png"), (conf_t.reshape(args.img_h,args.img_w)).cpu().numpy()*255.0)

            torchvision.utils.save_image(torch.from_numpy(vis_img).permute(2,0,1), os.path.join(out_dir, img_id+ "_vis.png"))
            # select confident predictions
            vis_img[vis_x,vis_y] = 0.0
            torchvision.utils.save_image(torch.from_numpy(vis_img).permute(2,0,1), os.path.join(out_dir, img_id+ "_vis_se.png"))
           

 

def regularize_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    scene_id = args.scene
   
    scene = join(args.data_root_2d, scene_id)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train_split=='train':
        transfile=os.path.join(scene, 'transforms_train.json')
    else:
        transfile=os.path.join(scene, 'transforms_test.json')

    with open(transfile) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
 
        for idx, frame in enumerate(tqdm(frames)):
            cam_name = frame["file_path"] #os.path.join(path, frame["file_path"] + extension)
            img_dir = os.path.join(scene, cam_name)
            img_id = os.path.basename(cam_name).split('.')[0]
            
            semantic_mask_path= img_dir.replace('color', 'object_mask')
            semantic_mask = imageio.v2.imread(semantic_mask_path)

            openseg_semantic = imageio.v2.imread(img_dir.replace('color', 'openseg_mask').replace('.png','_se.png'))
            openseg_semantic_vis = imageio.v2.imread(img_dir.replace('color', 'openseg_mask').replace('.png','_vis_se.png'))
            # openseg_semantic_conf = imageio.v2.imread(img_dir.replace('color', 'openseg_mask').replace('.png','_conf.png'))

            unique_ids= np.unique(semantic_mask)
            output_semantic_img = copy.deepcopy(openseg_semantic) #.clone()
            output_semantic_vis_img = copy.deepcopy(openseg_semantic_vis) #.clone()
    
            
            for uid in unique_ids:
                if uid==0:
                    continue
                sel_x,sel_y=np.where(semantic_mask==uid)
                cur_pred_label = openseg_semantic[sel_x,sel_y]
                cur_unique, cur_counts = np.unique(cur_pred_label, return_counts=True)
                sel_id=cur_unique[np.argmax(cur_counts)]
               
                pred_label_color = np.array(
                                    [args.palette[sel_id * 3],
                                    args.palette[sel_id * 3 + 1],
                                    args.palette[sel_id * 3 + 2]])
                output_semantic_img[sel_x,sel_y] = sel_id
   
                output_semantic_vis_img[sel_x,sel_y] = pred_label_color
            

            cv2.imwrite(os.path.join(out_dir, img_id+ "_reg.png"), output_semantic_img)
            torchvision.utils.save_image(torch.from_numpy(output_semantic_vis_img/255.0).permute(2,0,1), os.path.join(out_dir, img_id+ "_vis_reg.png"))
          

def load_ply(ply_path):
    ply_data = plyfile.PlyData.read(ply_path)
    data = ply_data['vertex']

    data = np.concatenate([data['x'].reshape(1, -1), data['y'].reshape(1, -1), data['z'].reshape(1, -1), \
                        data['red'].reshape(1, -1), data['green'].reshape(1, -1), data['blue'].reshape(1, -1)], axis=0)

    xyz = data.T[:, :3]
    rgb = data.T[:, 3:]

    return xyz, rgb


def compute_mapping(points, data_path, frame_id):  
    """
    :param points: N x 3 format
    :param depth: H x W format
    :param intrinsic: 3x3 format
    :return: mapping, N x 3 format, (H,W,mask)
    """
    vis_thres = 0.1
    depth_shift = 1000.0

    mapping = np.zeros((3, points.shape[0]), dtype=int)
    
    # Load the intrinsic matrix
    depth_intrinsic = np.loadtxt(os.path.join(data_path, 'intrinsics.txt'))
    
    # Load the depth image, and camera pose
    depth = cv2.imread(os.path.join(data_path,  'depth', frame_id + '.png'), -1) # read 16bit grayscale 
    pose = np.loadtxt(os.path.join(data_path, 'pose', frame_id + '.txt' ))

    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    
    points_world = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
    world_to_camera = np.linalg.inv(pose)
    p = np.matmul(world_to_camera, points_world.T)  # [Xb, Yb, Zb, 1]: 4, n
    p[0] = ((p[0] - bx) * fx) / p[2] + cx 
    p[1] = ((p[1] - by) * fy) / p[2] + cy
    
    # out-of-image check
    mask = (p[0] > 0) * (p[1] > 0) \
                    * (p[0] < depth.shape[1]-1) \
                    * (p[1] < depth.shape[0]-1)

    pi = np.round(p).astype(int) # simply round the projected coordinates
    
    # directly keep the pixel whose depth!=0
    depth_mask = depth[pi[1][mask], pi[0][mask]] != 0
    mask[mask == True] = depth_mask
    
    # occlusion check:
    trans_depth = depth[pi[1][mask], pi[0][mask]] / depth_shift
    est_depth = p[2][mask]
    occlusion_mask = np.abs(est_depth - trans_depth) <= vis_thres
    mask[mask == True] = occlusion_mask

    mapping[0][mask] = p[1][mask]
    mapping[1][mask] = p[0][mask]
    mapping[2][mask] = 1

    return mapping.T


def regularize_multiview_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    scene_id = args.scene
   
    scene = join(args.data_root_2d, scene_id)


    scene_plypath = os.path.join(scene, scene_id + '_mesh.ply') # '_vh_clean_2.ply') #
    locs_in, rgb = load_ply(scene_plypath)

    n_points = locs_in.shape[0]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pt_score=  np.zeros((n_points, args.class_num))
    # sum_features = torch.zeros((n_points, feat_dim), device=device)

    if args.train_split=='train':
        transfile=os.path.join(scene, 'transforms_train.json')
    else:
        transfile=os.path.join(scene, 'transforms_test.json')

    with open(transfile) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
 
        for idx, frame in enumerate(tqdm(frames)):
            cam_name = frame["file_path"] #os.path.join(path, frame["file_path"] + extension)
            img_dir = os.path.join(scene, cam_name)
            img_id = os.path.basename(cam_name).split('.')[0]
            
        
            openseg_semantic = imageio.v2.imread(img_dir.replace('color', 'openseg_mask'))#.replace('.png','_reg.png'))
           
            # calculate the 3d-2d mapping on ALL input points (not just prompt)
            mapping = compute_mapping(locs_in, scene,  img_id)
            if mapping[:, 2].sum() == 0: # no points corresponds to this image, skip
                continue
          
            # calculate the 3d-2d mapping based on the depth
            # mapping = torch.from_numpy(mapping).to(device)
            mask = mapping[:, 2]
            # vis_id[:, idx] = mask
            unique_ids= np.unique(openseg_semantic)
      
            for uid in unique_ids:
                if uid==0:
                    continue
                sel_x,sel_y=np.where(openseg_semantic==uid)
                cur_mask = np.zeros_like(openseg_semantic)
                cur_mask[sel_x,sel_y] =1 
                # import pdb
                # pdb.set_trace()
                mask_2d_3d = cur_mask[mapping[:, 0], mapping[:, 1]]
                mask_2d_3d = mask_2d_3d * mask
                pt_score[:, uid] += mask_2d_3d  # For each individual input point in the scene, \
              
        pt_pred_abs = np.argmax(pt_score, axis=-1)
        max_score_abs = np.max(pt_score, axis=-1)
        low_pt_idx_abs = np.where(max_score_abs <= 0.)[0]
        pt_pred_abs[low_pt_idx_abs] = 0 #-1
       

    print("Creating the visualization result ...")
    
    mesh_ori = o3d.io.read_triangle_mesh(scene_plypath)
    color_pred=convert_labels_with_palette(pt_pred_abs,args.palette)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = mesh_ori.vertices
    mesh.triangles = mesh_ori.triangles
    mesh.vertex_colors = o3d.utility.Vector3dVector(color_pred)
    output_vis_file = os.path.join(out_dir, args.scene_name + '_seg.ply')
    o3d.io.write_triangle_mesh(output_vis_file, mesh)
    print("Successfully save the visualization result of final segmentation!")

        
def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    ##### Dataset specific parameters #####
    # img_dim =  (640, 480) #(320, 240) #(640, 360)
    args.depth_scale = 1000 # 6553.5

    scenes = [args.scene_name]#['room0', 'room1', 'room2', 'office0',
            #   'office1', 'office2', 'office3', 'office4']
    #######################################
    args.img_h=480
    args.img_w=640
    args.feat_dim = 768 # CLIP feature dimension

    data_dir = args.data_dir

    data_root = join(data_dir, 'replica_3d')
    data_root_2d = join(data_dir,'replica_2d')
    args.data_root_2d = data_root_2d
    # out_dir = args.output_dir
    out_dir = join(data_root_2d,args.scene_name,args.feature_2d_extractor+'_mask')
    os.makedirs(out_dir, exist_ok=True)


    # args.n_split_points = 2000000
    # args.num_rand_file_per_scene = 1

    # load the openseg model
    saved_model_path = args.openseg_model
    args.text_emb = None
    if 'pretrained' in args.openseg_model: #args.openseg_model != '':
        print('================================Loading openseg model=======================')
        args.openseg_model = tf2.saved_model.load(saved_model_path,
                    tags=[tf.saved_model.tag_constants.SERVING],)
        args.text_emb = tf.zeros([1, 1, args.feat_dim])
    else:
        args.openseg_model = None
    
 

    args.class_num = 52

    args.palette = np.concatenate(label_colormap(101)[np.array(REPLICA_EXISTING_CLASSES).astype(np.uint8)])
    args.text_features=torch.load(os.path.join(args.data_root_2d,args.scene_name,'text_feat.pth')).detach()


    for scene in scenes:
        data_path=os.path.join(data_root, f'{scene}.pth')
        args.scene=scene
        ######## 1) Process to get semantics and confidence maps of openseg model
        #  CUDA_VISIBLE_DEVICES=2 python replica_openseg_output_semantics.py --data_dir  --scene_name office0 --train_split test
        process_one_scene(data_path, out_dir, args)
        ######## 2)using object_mask from sam to regularize openseg features
        # CUDA_VISIBLE_DEVICES=3 python replica_openseg_output_semantics.py --data_dir  --scene_name room0 --train_split test --openseg_model none
        regularize_one_scene(data_path, out_dir, args)
        ######## 3) using multiview consitency to make semantic masks across views consistent according to 3d points and extract 
        regularize_multiview_one_scene(data_path, out_dir, args)



if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)
