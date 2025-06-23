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




def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Replica.')
    parser.add_argument('--data_dir', type=str, default='tmp_dataset', help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, default='tmp_dataset',help='Where is the base logging directory')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')
    parser.add_argument('--feature_2d_extractor', type=str, default='openseg',help='[lseg,openseg]')
    parser.add_argument('--scene_name', type=str, default='room0', help='which scene')
    parser.add_argument('--num_rand_file_per_scene', type=int, default=0)
    parser.add_argument('--with_sammask', action='store_true')
    parser.add_argument('--train_split', type=str, default='train', help='Where is the base logging directory')
    
    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    # short hand
    scene_id = args.scene

    num_rand_file_per_scene = args.num_rand_file_per_scene
    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale
    openseg_model = args.openseg_model
    text_emb = args.text_emb


    # load 3D data
    locs_in = torch.load(data_path)[0]
    n_points = locs_in.shape[0]


    if exists(join(out_dir, scene_id +'_%d.pt'%(num_rand_file_per_scene))):
        # n_finished += 1
        print(scene_id +'_%d.pt'%(num_rand_file_per_scene) + ' already done!')


    # short hand for processing 2D features
    scene = join(args.data_root_2d, scene_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    if args.with_sammask:
        args.text_features = args.text_features.to(device).half()

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, feat_dim), device=device)


    if args.train_split=='train':
        transfile=os.path.join(scene, 'transforms_train.json')
    else:
        transfile=os.path.join(scene, 'transforms_test.json')

    with open(transfile) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        num_img = len(frames)
        vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
        
        for idx, frame in enumerate(tqdm(frames)):
            cam_name = frame["file_path"] #os.path.join(path, frame["file_path"] + extension)
            img_dir = os.path.join(scene, cam_name)
            img_id = int(os.path.basename(cam_name).split('.')[0])
          
            posepath = img_dir.replace('color', 'pose').replace('.png', '.txt')
            pose = np.loadtxt(posepath)

            # load depth and convert to meter
            depth = imageio.v2.imread(img_dir.replace('color', 'depth')) / depth_scale

            # load SAM mask 
            if args.with_sammask:
                # semantic_mask_path= os.path.join(os.path.dirname(img_dir).replace('color', 'semantics'), '0_'+ str(img_id).zfill(4)+'.png')
                semantic_mask_path= img_dir.replace('color', 'object_mask')
                semantic_mask = imageio.v2.imread(semantic_mask_path)

            
            # calculate the 3d-2d mapping based on the depth
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
            if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
                continue

            # calculate the 3d-2d mapping based on the depth
            mapping = torch.from_numpy(mapping).to(device)
            mask = mapping[:, 3]
            vis_id[:, idx] = mask
            feat_2d = extract_openseg_img_feature(img_dir, openseg_model, text_emb).to(device)

        
            if args.with_sammask:
                new_feat_2d= feat_2d.clone().to(device)
                feat_2d_flat= feat_2d.reshape(new_feat_2d.shape[0],-1).half()
                outputs_target = torch.matmul(args.text_features.to(torch.float32), feat_2d_flat.to(torch.float32)).t() #
                softmax_t = torch.softmax(outputs_target,axis=-1)
                _, pred_t = torch.max(softmax_t, 1)
                onehot_t = torch.eye(args.class_num)[pred_t].to(device).half()# .double()#
                denom_one= onehot_t.to(torch.float32).sum(dim=0) 
                
                center_t = torch.matmul(feat_2d_flat.to(torch.float32), onehot_t.to(torch.float32)) / (denom_one + 1e-8) # (D,C)
            
                unique_ids= np.unique(semantic_mask)
                new_center= []
            
                for uid in unique_ids:

                    tmp_mk = semantic_mask==uid
                    tmp_feat = new_feat_2d[:,tmp_mk].to(device)# (D,N)-(768,N)
                    outputs_t = torch.matmul(args.text_features.to(torch.float32),  tmp_feat.to(torch.float32)).t() #
                    _, pd_t = torch.max(torch.softmax(outputs_t,axis=-1), 1)
                    onehot_ct = torch.eye(args.class_num)[pd_t].to(device).to(torch.float32)
                    new_id = torch.argmax(onehot_ct.sum(dim=0))
                    if not torch.isnan(center_t[:,new_id].half()).any() and not torch.isinf(center_t[:,new_id].half()).any():
                        new_feat_2d[:,tmp_mk] = center_t[:,new_id][:,None].half()
                        new_center.append(center_t[:,new_id][None,:])
                    else:
                        print(center_t[:,new_id])
                        import pdb
                        pdb.set_trace()
                
                new_center = torch.cat(new_center)
                os.makedirs(out_dir+'_object_center',exist_ok=True)
                torch.save(new_center.cpu(),os.path.join(out_dir+'_object_center',os.path.basename(img_dir).replace('.png','.pth')))
                # import pdb
                # pdb.set_trace()

            if args.with_sammask:
                del feat_2d 
                # import pdb
                # pdb.set_trace()
                feat_2d_3d = new_feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)
                del new_feat_2d 
            else:
                feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)
            
            counter[mask!=0]+= 1
            sum_features[mask!=0] += feat_2d_3d[mask!=0]
            if torch.isnan(sum_features).any() or torch.isinf(sum_features).any():
                import pdb
                pdb.set_trace()
            del feat_2d_3d 

        
        counter[counter==0] = 1e-5
        feat_bank = sum_features.to(torch.float32)/counter
        if torch.isnan(feat_bank).any() or torch.isinf(feat_bank).any():
            import pdb
            pdb.set_trace()
        else:
            torch.save(feat_bank.cpu(),os.path.join(out_dir, scene_id +'_sum.pth'))

        # feat_bank = torch.load(os.path.join(out_dir, scene_id +'_sum.pth')).to(device)
        point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])
        save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id, args)

def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    ##### Dataset specific parameters #####
    img_dim =  (640, 480) #(320, 240) #(640, 360)
    depth_scale = 1000 # 6553.5

    scenes = [args.scene_name]#['room0', 'room1', 'room2', 'office0',
            #   'office1', 'office2', 'office3', 'office4']
    #######################################
    visibility_threshold = 0.25 # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    args.keep_features_in_memory = False # keep image features in the memory, very expensive
    args.feat_dim = 768 # CLIP feature dimension

    data_dir = args.data_dir

    data_root = join(data_dir, 'replica_3d')
    data_root_2d = join(data_dir,'replica_2d')
    args.data_root_2d = data_root_2d
    # out_dir = args.output_dir
    out_dir = join(data_root_2d,args.scene_name,'replica_multiview_'+args.feature_2d_extractor)
    os.makedirs(out_dir, exist_ok=True)


    args.n_split_points = 2000000
    # args.num_rand_file_per_scene = 1

    # load the openseg model
    saved_model_path = args.openseg_model
    args.text_emb = None
    if args.openseg_model != '':
        args.openseg_model = tf2.saved_model.load(saved_model_path,
                    tags=[tf.saved_model.tag_constants.SERVING],)
        args.text_emb = tf.zeros([1, 1, args.feat_dim])
    else:
        args.openseg_model = None
    
    args.with_sammask = True
    args.text_features = None
    args.class_num = 52
    if args.with_sammask:
        
        args.text_features=torch.load(os.path.join(args.data_root_2d,args.scene_name,'text_feat.pth'))
       
    # load intrinsic parameter
    intrinsics=np.loadtxt(os.path.join(args.data_root_2d,args.scene_name, 'intrinsics.txt'))

    # calculate image pixel-3D points correspondances   
    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=intrinsics,
            visibility_threshold=visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)

    for scene in scenes:
        data_path=os.path.join(data_root, f'{scene}.pth')
        args.scene=scene
        process_one_scene(data_path, out_dir, args)



if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)
