import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset,  collation_fn, Point3DLoader, collation_fn_eval_all
from model import Autoencoder
from util import export_pointcloud, get_palette, \
    convert_labels_with_palette, extract_text_feature, visualize_labels
from label_constants import *



def precompute_text_related_properties(labelset_name):
    '''pre-compute text features, labelset, palette, and mapper.'''

    if 'scannet' in labelset_name:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other' # change 'other furniture' to 'other'
        palette = get_palette(colormap='scannet')
    else: # an arbitrary dataset, just use a large labelset
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')

    mapper = None
    if hasattr(args, 'map_nuscenes_details'):
        labelset = list(NUSCENES_LABELS_DETAILS)
        mapper = torch.tensor(MAPPING_NUSCENES_DETAILS, dtype=int)

    text_features = extract_text_feature(labelset, args)
    # labelset.append('unknown')
    labelset.append('unlabeled')
    return text_features, labelset, mapper, palette



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='') #required=True)
    parser.add_argument('--dataset_name', type=str, default='scannet')
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 768],
                    )
    parser.add_argument('--output', type=str, default='scene0389_00', help='[train/scene0004_00, val/scene0389_00, val/scene0494_00, val/scene0693_00]')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--voxel_size', type=int, default=0.02)
    parser.add_argument('--loop', type=int, default=1)
    parser.add_argument('--feature_2d_extractor', type=str, default='openseg',help='[lseg,openseg]')
    parser.add_argument('--input_color', type=bool, default=True) #required=True)
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    ckpt_path = f"ckpt/{args.dataset_name}/{args.output}/best_ckpt_1.pth"
    dataset_path = os.path.join(args.dataset_path,'scannet_3d')
    # data_dir = f"{dataset_path}/language_features"
    data_root_2d_fused_feature = os.path.join(args.dataset_path,args.output,'scannet_multiview_' + args.feature_2d_extractor)
    # output_dir = os.path.join(data_root_2d_fused_feature, 'language_features_dim3')#f"ckpt/{args.dataset_name}/{args.output}/language_features_dim3"
    output_pt_dir = os.path.join(args.dataset_path,args.output) #os.path.join(data_root_2d_fused_feature,'point_feat_3d')  #f"ckpt/{args.dataset_name}/{args.output}/point_feat_3d"


    checkpoint = torch.load(ckpt_path)
    # train_dataset = Autoencoder_dataset(data_dir)
    train_data = Point3DLoader(datapath_prefix=os.path.join(dataset_path,args.output),
                                datapath_prefix_feat=data_root_2d_fused_feature,
                                voxel_size=args.voxel_size,
                                split='train', aug=False,
                                memcache_init=False, loop=args.loop,
                                input_color=args.input_color,#eval_all= True
                                )
    test_loader = DataLoader(
        dataset=train_data, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4, 
        collate_fn=collation_fn,
        drop_last=False   
    )


    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).cuda()

    model.load_state_dict(checkpoint)

    labelset_name='scannet_3d'
    text_features, labelset, mapper, palette = \
        precompute_text_related_properties(labelset_name)
    
    model.eval()
    num_class= text_features.shape[0]

     ### textlatent feature 
    text_feat_dim3 = model.encode(text_features.to(torch.float32))
    text_feat_dim3_norm = text_feat_dim3/ (text_feat_dim3.norm(dim=-1, keepdim=True) + 1e-9)
    # text_feat_recon = model.decode(text_feat_dim3)

    
    text_feat_path= os.path.join(output_pt_dir,'text_feat_dim3_norm.pth')
    if os.path.exists(text_feat_path):
        print('Loading save text_feature in latent space!')
        text_features_dim3_ave=torch.load(os.path.join(output_pt_dir,'text_feat_dim3_norm.pth')).cuda()
    else:
        text_features_dim3_ave= text_feat_dim3_norm #np.ones((num_class,3))/num_class
   
    for idx, data_batch in tqdm(enumerate(test_loader)):
        coords, feats, labels, feat_3d = data_batch
        data  = feat_3d.cuda(non_blocking=True) 
        locs_in = coords[:,1:]#.cuda(non_blocking=True) 
        colors = feats#.cuda(non_blocking=True) 
        assert coords.shape[0]==feat_3d.shape[0]
        if data.dtype == torch.float16:
            data.data = data.data.to(torch.float32)
        with torch.no_grad():
            # import pdb
            # pdb.set_trace()
            outputs_dim3 = model.encode(data)
            outputs_dim3_norm = outputs_dim3/ (outputs_dim3.norm(dim=-1, keepdim=True) + 1e-9)
            outputs = model.decode(outputs_dim3_norm)

            pred_latent= torch.max(outputs_dim3_norm.half() @ text_features_dim3_ave.half().t(), 1)[1].cpu()
            pred = outputs.half() @ text_features.t()
            logits_pred = torch.max(pred, 1)[1].cpu()
            pred_gt= data.half() @ text_features.t()
            gt_pred=torch.max(pred_gt, 1)[1].cpu()
       
        if idx == 0:
            features = outputs_dim3.to("cpu").numpy()  
            point_cld = np.concatenate([locs_in,colors], axis=1)
            pred_lb= logits_pred.numpy()  
            gt_lb = gt_pred.numpy()  
            latent_lb = pred_latent.numpy()  
        else:
            features = np.concatenate([features, outputs_dim3.to("cpu").numpy()], axis=0)
            point_cld = np.concatenate([point_cld, np.concatenate([locs_in,colors], axis=1)], axis=0)
            pred_lb = np.concatenate([pred_lb, logits_pred.numpy()], axis=0)
            gt_lb = np.concatenate([gt_lb,gt_pred.numpy()], axis=0)
            latent_lb = np.concatenate([latent_lb,pred_latent.numpy()], axis=0)
        


    accu=np.sum(pred_lb==gt_lb)/pred_lb.shape[0]
    accu_lat=np.sum(latent_lb==gt_lb)/latent_lb.shape[0]
    print('accuracy:{}-accuracy_latent:{}'.format(accu,accu_lat))
    print('unique classes in this scene:{}'.format(np.unique(gt_lb)))

   
    text_features_dim3_ave_norm =text_features_dim3_ave/ (text_features_dim3_ave.norm(dim=-1, keepdim=True) + 1e-9)
    os.makedirs(output_pt_dir, exist_ok=True)
    # ### textlatent feature 
    text_feat_dim3 = model.encode(text_features.to(torch.float32))
    text_feat_dim3_norm = text_feat_dim3/ (text_feat_dim3.norm(dim=-1, keepdim=True) + 1e-9)
    text_feat_recon = model.decode(text_feat_dim3_norm) #model.decode(torch.from_numpy(text_features_dim3_ave_norm).half().cuda())
    
    tmpp= torch.matmul(text_features.to(torch.float32),text_features.to(torch.float32).t()).max(1)[1]
    tmp= torch.matmul(text_feat_recon.to(torch.float32),text_features.to(torch.float32).t()).max(1)[1]
    tmp1 = torch.matmul(text_features_dim3_ave_norm.to(torch.float32),text_feat_dim3_norm.to(torch.float32).t()).max(1)[1]

 
    torch.save(text_features_dim3_ave,os.path.join(output_pt_dir,'text_feat_dim3_norm.pth'))

    start = 0
    for k,v in train_data.data_dic.items():

        positions=point_cld[start:start+v,:3]
        colors_s=point_cld[start:start+v,3:]

        torch.save((positions,colors_s,features[start:start+v]),os.path.join(output_pt_dir,'points3d.pth'))
        start += v
