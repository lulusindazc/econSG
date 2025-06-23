import os
import torch
import imageio
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from os.path import join, exists
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper, save_fused_feature, adjust_intrinsic, make_intrinsic
from model import Autoencoder
import json



def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Replica.')
    parser.add_argument('--data_dir', type=str, default='', help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, default='',help='Where is the base logging directory')
    parser.add_argument('--dataset_name', type=str, default='replica')
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 6],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 768],
                    )
    # parser.add_argument('--output', type=str, default='scene0494_00', help='[train/scene0004_00, val/scene0389_00, val/scene0494_00, val/scene0693_00]')
    parser.add_argument('--train_split', type=str, default='train', help='Where is the base logging directory')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    # parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')
    parser.add_argument('--feature_2d_extractor', type=str, default='openseg',help='[lseg,openseg]')
    parser.add_argument('--scene_name', type=str, default='scene0494_00', help='which scene')
    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    parser.add_argument('--num_rand_file_per_scene', type=int, default=0)
    parser.add_argument('--with_sammask', action='store_true')
    args = parser.parse_args()
    return args




def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    ##### Dataset specific parameters #####
    img_dim = (640, 480) #(320, 240) #(640, 360)
    depth_scale = 1000 #6553.5

    scenes = [args.scene_name]#['room0', 'room1', 'room2', 'office0',
            #   'office1', 'office2', 'office3', 'office4']
    #######################################
    visibility_threshold = 0.25 # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    args.keep_features_in_memory = False # keep image features in the memory, very expensive
    args.feat_dim = 768 # CLIP feature dimension

    data_dir = args.data_dir

    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims


    data_root_3d = join(data_dir, 'replica_3d')
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


    scene = join(data_root_2d, args.scene_name)
    data_path = join(data_root_3d, args.scene_name+'.pth')
    # load 3D data (point cloud)
    locs_in = torch.load(data_path)[0]
    n_points = locs_in.shape[0]

    img_dirs = sorted(glob(join(scene, 'color/*')), key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)
    # device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # args.with_sammask = True
    args.text_features = None
    args.class_num = 52
    if args.with_sammask:
        
        args.text_features=torch.load(os.path.join(args.data_root_2d,args.scene_name,'text_feat.pth'))
    
    if args.num_rand_file_per_scene>0:
        prefix='_sam'
    else:
        prefix=''

    out_dir = join(scene, 'language_feature_dim3',args.feature_2d_extractor+prefix)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(join(out_dir,'semantics'), exist_ok=True)
  
   
    
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).cuda()

    if args.num_rand_file_per_scene>0:
        ckpt_path = f"ckpt/{args.dataset_name}/{args.scene_name}/best_ckpt_dim6_{args.num_rand_file_per_scene}.pth"
    else:
        ckpt_path = f"ckpt/{args.dataset_name}/{args.scene_name}/best_ckpt_dim6.pth"

   
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
   
    print("Loading from {}".format(ckpt_path))
      # load the openseg model
    # saved_model_path = args.openseg_model

    print('output dir:{}'.format(out_dir))
    n_points_cur = n_points
    # counter = torch.zeros((n_points_cur, 1), device=device)
    # sum_features = torch.zeros((n_points_cur, args.feat_dim), device=device)

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

            image_name=img_dir.split('/')[-1]
           
            if args.with_sammask:
                # semantic_mask_path= os.path.join(os.path.dirname(img_dir).replace('color', 'semantics'), '0_'+ str(img_id).zfill(4)+'.png')
                semantic_mask_path= img_dir.replace('color', 'object_mask')
                semantic_mask = imageio.v2.imread(semantic_mask_path)
        

            feat_2d = extract_openseg_img_feature(img_dir, args.openseg_model,args.text_emb, img_size=[img_dim[1], img_dim[0]]).to(device)
            # (768,240,320)
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

    
            with torch.no_grad():
               
                data=new_feat_2d.view(args.feat_dim,-1).permute(1, 0) #(240*320,768)
                if data.dtype == torch.float16:
                    data.data = data.data.to(torch.float32)
                num_batch= np.arange(0,data.shape[0],2400)
                features = []
                for bs in num_batch:
                    cur_data= data[bs:bs+2400]
                    outputs_dim3 = model.encode(cur_data.cuda(non_blocking=True) )

                    # outputs = model.decode(outputs_dim3)
                    if torch.isnan(outputs_dim3).any() or torch.isinf(outputs_dim3).any():
                        import pdb
                        pdb.set_trace()
                
                    features.append(outputs_dim3.to("cpu").numpy())  
              
                features= np.concatenate(features)
             
                assert features.shape[0]==data.shape[0]
                del feat_2d
                del data
              
    

            path = os.path.join(out_dir, image_name.replace('.png', '.npy'))
            np.save(path, features)
            




if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)
