import os
# import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

'''Dataloader for fused point features.'''

import copy
from glob import glob
# import torch
# import numpy as np
import SharedArray as SA

# from dataset.point_loader import Point3DLoader



'''Dataloader for 3D points.'''

import multiprocessing as mp
from os.path import join, exists
import augmentation as t
from voxelizer import Voxelizer


class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir):
        data_names = glob(os.path.join(data_dir, '*f.npy'))
        self.data_dic = {}
        for i in range(len(data_names)):
            features = np.load(data_names[i])
            name = data_names[i].split('/')[-1].split('.')[0]
            self.data_dic[name] = features.shape[0] 
            if i == 0:
                data = features
            else:
                data = np.concatenate([data, features], axis=0)
        self.data = data

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data

    def __len__(self):
        return self.data.shape[0] 



def sa_create(name, var):
    '''Create share memory.'''

    shared_mem = SA.create(name, var.shape, dtype=var.dtype)
    shared_mem[...] = var[...]
    shared_mem.flags.writeable = False
    return shared_mem


class Point3DLoader(torch.utils.data.Dataset):
    '''Dataloader for 3D points and labels.'''

    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, datapath_prefix='data', datapath_prefix_feat='',voxel_size=0.05,
                 split='train', aug=False, memcache_init=False, identifier=1233, loop=1,
                 data_aug_color_trans_ratio=0.1,
                 data_aug_color_jitter_std=0.05,
                 data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2,
                 eval_all=False, input_color=False
                 ):
        super().__init__()
        self.split = split
        if split is None:
            split = ''
        self.identifier = identifier
        self.data_paths = [datapath_prefix+'_vh_clean_2.pth'] 
        self.data_dic = {}

        if len(self.data_paths) == 0:
            raise Exception('0 file is loaded in the point loader.')

        self.input_color = input_color
        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat

        self.voxel_size = voxel_size
        self.aug = aug
        self.loop = loop
        self.eval_all = eval_all
        dataset_name = datapath_prefix.split('/')[-2]
        self.dataset_name = dataset_name
        self.use_shm = memcache_init

        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        if aug:
            prevoxel_transform_train = [
                t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
            self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
            input_transforms = [
                t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
                t.ChromaticAutoContrast(),
                t.ChromaticTranslation(data_aug_color_trans_ratio),
                t.ChromaticJitter(data_aug_color_jitter_std),
                t.HueSaturationTranslation(
                    data_aug_hue_max, data_aug_saturation_max),
            ]
            self.input_transforms = t.Compose(input_transforms)

        if memcache_init and (not exists("/dev/shm/%s_%s_%06d_locs_%08d" % (dataset_name, split, identifier, 0))):
            print('[*] Starting shared memory init ...')
            print('No. CPUs: ', mp.cpu_count())
            for i, (locs, feats, labels) in enumerate(torch.utils.data.DataLoader(
                    self.data_paths, collate_fn=lambda x: torch.load(x[0]),
                    num_workers=min(16, mp.cpu_count()), shuffle=False)):
                labels[labels == -100] = 255
                labels = labels.astype(np.uint8)
                # no color in the input point cloud, e.g nuscenes
                if np.isscalar(feats) and feats == 0:
                    feats = np.zeros_like(locs)
                # Scale color to 0-255
                feats = feats*255.0  #(feats + 1.) * 127.5
                sa_create("shm://%s_%s_%06d_locs_%08d" %
                          (dataset_name, split, identifier, i), locs)
                sa_create("shm://%s_%s_%06d_feats_%08d" %
                          (dataset_name, split, identifier, i), feats)
                sa_create("shm://%s_%s_%06d_labels_%08d" %
                          (dataset_name, split, identifier, i), labels)
            print('[*] %s (%s) loading 3D points done (%d)! ' %
                  (datapath_prefix, split, len(self.data_paths)))
        self.feat_all=[]
        self.labels_all,self.mask_all=[],[]
        self.coords_all,self.ft_all=[],[]
        self.ind_rec_all=[]
        for index in range(len(self.data_paths) * self.loop):
          
            locs_in, feats_in, labels_in = torch.load(self.data_paths[0])

            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)

           
            if np.isscalar(feats_in) and feats_in == 0:
                # no color in the input point cloud, e.g nuscenes lidar
                feats_in = np.zeros_like(locs_in)
            else:
                feats_in = feats_in * 255.0 #(feats_in + 1.) * 127.5

            # load 3D features
            if self.dataset_name == 'scannet_3d':
                scene_name = self.data_paths[0][:-15].split('/')[-1]
            else:
                scene_name = self.data_paths[0][:-4].split('/')[-1]

            if 'nuscenes' not in self.dataset_name:
              
                processed_data = torch.load(join(
                    self.datapath_feat, scene_name+'_%d.pt'%(index)))
                name=scene_name+'_%d'%(index)
            else:
                # no repeated file
                processed_data = torch.load(join(self.datapath_feat, scene_name+'.pt'))
                name=scene_name
        
            
            flag_mask_merge = False
            if len(processed_data.keys())==2:
                flag_mask_merge = True
                feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
                # print('initial feature_3D shape:{}'.format(feat_3d.shape))
                if isinstance(mask_chunk, np.ndarray): # if the mask itself is a numpy array
                    mask_chunk = torch.from_numpy(mask_chunk)
                mask = copy.deepcopy(mask_chunk)
                if self.split != 'train': # val or test set
                    feat_3d_new = torch.zeros((locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype)
                    feat_3d_new[mask] = feat_3d
                    feat_3d = feat_3d_new
                    mask_chunk = torch.ones_like(mask_chunk) # every point needs to be evaluted
            elif len(processed_data.keys())>2: # legacy, for old processed features
                feat_3d, mask_visible, mask_chunk = processed_data['feat'], processed_data['mask'], processed_data['mask_full']
                mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
                mask[mask_visible] = True # mask out points without feature assigned

            if len(feat_3d.shape)>2:
                feat_3d = feat_3d[..., 0]

            locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
            # import pdb
            # pdb.set_trace()
            # calculate the corresponding point features after voxelization
            if self.split == 'train' and flag_mask_merge:
                locs_aug, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                    locs_in, feats_in, labels_in, return_ind=True)
                vox_ind = torch.from_numpy(vox_ind)
                mask = mask_chunk[vox_ind] # voxelized visible mask for entire point cloud
                mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
                index1 = - torch.ones(mask_chunk.shape[0], dtype=int)
                index1[mask_ind] = mask_ind

                index1 = index1[vox_ind]
                chunk_ind = index1[index1!=-1]

                index2 = torch.zeros(mask_chunk.shape[0])
                index2[mask_ind] = 1
                index3 = torch.cumsum(index2, dim=0, dtype=int)
                # get the indices of corresponding masked point features after voxelization
                indices = index3[chunk_ind] - 1

                # get the corresponding features after voxelization
                feat_3d = feat_3d[indices]
                locs = locs_in[vox_ind]
          
            else:
                locs_aug, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                    locs[mask_chunk], feats_in[mask_chunk], labels_in[mask_chunk], return_ind=True)
                vox_ind = torch.from_numpy(vox_ind)
                feat_3d = feat_3d[vox_ind]
                mask = mask[vox_ind]
                locs = locs_in[vox_ind]
            # print('Masked feature_3D shape:{}'.format(feat_3d.shape))
            if self.eval_all: # during evaluation, no voxelization for GT labels
                labels = labels_in
            if self.aug:
                locs, feats, labels = self.input_transforms(locs, feats, labels)
            coords = torch.from_numpy(locs).int()
            coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
            if self.input_color:
                feats = torch.from_numpy(feats).float() / 255.0 #127.5 - 1.
            else:
                # hack: directly use color=(1, 1, 1) for all points
                feats = torch.ones(coords.shape[0], 3)
            labels = torch.from_numpy(labels).long()
            self.data_dic[name] = feat_3d.shape[0]
            # ind_=torch.from_numpy(inds_reconstruct).long
            self.feat_all.append(feat_3d)
            # self.mask_all.append(mask)
            self.labels_all.append(labels[mask])
            self.coords_all.append(coords[mask])
            self.ft_all.append(feats[mask])
            # self.ind_rec_all.append(ind_)
         
            print('f:{}-m:{}-l:{}-c:{}-ft:{}'.format(feat_3d.shape,mask.shape,labels[mask].shape,coords[mask].shape,feats[mask].shape))
            # import pdb
            # pdb.set_trace()
                   
        self.feat_emb=torch.cat(self.feat_all)
        # self.mask_emb=torch.cat(self.mask_all)
        self.labels_emb=torch.cat(self.labels_all)
        self.coords_emb=torch.cat(self.coords_all)
        self.ft_emb=torch.cat(self.ft_all)
     
    def __getitem__(self, index_long):
        #mask=self.mask_emb[index_long]
        feat_3d=self.feat_emb[index_long]
        coords=self.coords_emb[index_long]
        feats=self.ft_emb[index_long]
        labels=self.labels_emb[index_long]
       
        return  coords, feats, labels, feat_3d#, mask


    def __len__(self):
        return self.feat_emb.shape[0]#len(self.data_paths) * self.loop





class Point3DMultiLoader(torch.utils.data.Dataset):
    '''Dataloader for 3D points and labels. for multiple scenes'''

    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, datapath_prefix='data', datapath_prefix_feat='',voxel_size=0.05,
                 split='train', aug=False, memcache_init=False, identifier=1233, loop=1,
                 data_aug_color_trans_ratio=0.1,
                 data_aug_color_jitter_std=0.05,
                 data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2,
                 eval_all=False, input_color=False
                 ):
        super().__init__()
        self.split = split
        if split is None:
            split = ''
        self.identifier = identifier
        self.data_paths = sorted(glob(join(datapath_prefix, split, '*.pth')))
        self.data_dic = {}

        if len(self.data_paths) == 0:
            raise Exception('0 file is loaded in the point loader.')

        self.input_color = input_color
        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat

        self.voxel_size = voxel_size
        self.aug = aug
        self.loop = loop
        self.eval_all = eval_all
        dataset_name = datapath_prefix.split('/')[-1]
        self.dataset_name = dataset_name
        print('===============dataset_name:{}================='.format(self.dataset_name))
        self.use_shm = memcache_init

        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        if aug:
            prevoxel_transform_train = [
                t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
            self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
            input_transforms = [
                t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
                t.ChromaticAutoContrast(),
                t.ChromaticTranslation(data_aug_color_trans_ratio),
                t.ChromaticJitter(data_aug_color_jitter_std),
                t.HueSaturationTranslation(
                    data_aug_hue_max, data_aug_saturation_max),
            ]
            self.input_transforms = t.Compose(input_transforms)

      
        self.feat_all=[]
        self.labels_all,self.mask_all=[],[]
        self.coords_all,self.ft_all=[],[]
        self.ind_rec_all=[]

        count_num_labels = np.zeros(20)
        num_sampels= 10000 # per class
        for index in tqdm(range(len(self.data_paths)), desc="Data Loading progress"):
            # index_data= index % len(self.data_paths)
            # index_feat=index % self.loop
            if np.min(count_num_labels)>=num_sampels:
                break
            else:
                locs_in, feats_in, labels_in = torch.load(self.data_paths[index])

                labels_in[labels_in == -100] = 255
                labels_in = labels_in.astype(np.uint8)
               
            
                if np.isscalar(feats_in) and feats_in == 0:
                    # no color in the input point cloud, e.g nuscenes lidar
                    feats_in = np.zeros_like(locs_in)
                else:
                    feats_in = feats_in * 255.0 #(feats_in + 1.) * 127.5

                # load 3D features
                if self.dataset_name == 'scannet_3d':
                    scene_name = self.data_paths[index][:-15].split('/')[-1] #self.data_paths[0][:-15].split('/')[-1]
                else:
                    scene_name = self.data_paths[index][:-4].split('/')[-1] #self.data_paths[0][:-4].split('/')[-1]

                cur_loop=1 #len(glob(join(self.datapath_feat, scene_name + '_*.pt')))
                
                for idx_loop in range(cur_loop):
                    if 'nuscenes' not in self.dataset_name:
                        processed_data = torch.load(join(
                            self.datapath_feat, scene_name+'_%d.pt'%(idx_loop)))
                        name=scene_name+'_%d'%(index)
                    else:
                        # no repeated file
                        processed_data = torch.load(join(self.datapath_feat, scene_name+'.pt'))
                        name=scene_name
                
                    
                    flag_mask_merge = False
                    if len(processed_data.keys())==2:
                        flag_mask_merge = True
                        feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
                        # print('initial feature_3D shape:{}'.format(feat_3d.shape))
                        if isinstance(mask_chunk, np.ndarray): # if the mask itself is a numpy array
                            mask_chunk = torch.from_numpy(mask_chunk)
                        mask = copy.deepcopy(mask_chunk)
                        if self.split != 'train': # val or test set
                            feat_3d_new = torch.zeros((locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype)
                            feat_3d_new[mask] = feat_3d
                            feat_3d = feat_3d_new
                            mask_chunk = torch.ones_like(mask_chunk) # every point needs to be evaluted
                    elif len(processed_data.keys())>2: # legacy, for old processed features
                        feat_3d, mask_visible, mask_chunk = processed_data['feat'], processed_data['mask'], processed_data['mask_full']
                        mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
                        mask[mask_visible] = True # mask out points without feature assigned

                    if len(feat_3d.shape)>2:
                        feat_3d = feat_3d[..., 0]

                    locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
                   
                    # calculate the corresponding point features after voxelization
                    if self.split == 'train' and flag_mask_merge:
                        locs_aug, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                            locs_in, feats_in, labels_in, return_ind=True)
                        vox_ind = torch.from_numpy(vox_ind)
                        mask = mask_chunk[vox_ind] # voxelized visible mask for entire point cloud
                        mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
                        index1 = - torch.ones(mask_chunk.shape[0], dtype=int)
                        index1[mask_ind] = mask_ind

                        index1 = index1[vox_ind]
                        chunk_ind = index1[index1!=-1]

                        index2 = torch.zeros(mask_chunk.shape[0])
                        index2[mask_ind] = 1
                        index3 = torch.cumsum(index2, dim=0, dtype=int)
                        # get the indices of corresponding masked point features after voxelization
                        indices = index3[chunk_ind] - 1

                        # get the corresponding features after voxelization
                        feat_3d = feat_3d[indices]
                        locs = locs_in[vox_ind]
                
                    else:
                        locs_aug, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                            locs[mask_chunk], feats_in[mask_chunk], labels_in[mask_chunk], return_ind=True)
                        vox_ind = torch.from_numpy(vox_ind)
                        feat_3d = feat_3d[vox_ind]
                        mask = mask[vox_ind]
                        locs = locs_in[vox_ind]
                    # print('Masked feature_3D shape:{}'.format(feat_3d.shape))
                    if self.eval_all: # during evaluation, no voxelization for GT labels
                        labels = labels_in
                    if self.aug:
                        locs, feats, labels = self.input_transforms(locs, feats, labels)
                    coords = torch.from_numpy(locs).int()
                    coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
                    if self.input_color:
                        feats = torch.from_numpy(feats).float() / 255.0 #127.5 - 1.
                    else:
                        # hack: directly use color=(1, 1, 1) for all points
                        feats = torch.ones(coords.shape[0], 3)
                    labels = torch.from_numpy(labels).long()
                    self.data_dic[name] = feat_3d.shape[0]
                

                    cur_labset= np.unique(labels[mask])
                    cur_bincount= np.bincount(labels[mask])
                    for i in cur_labset:
                        if i <20:
                            if count_num_labels[i]+cur_bincount[i]<num_sampels:
                                extra_num=  cur_bincount[i]
                                cur_lbmk= labels[mask]==i
                                count_num_labels[i]+=extra_num
                                
                            elif count_num_labels[i]+cur_bincount[i]> num_sampels and count_num_labels[i]<num_sampels:
                                extra_num= (num_sampels-count_num_labels[i]).astype(int) 
                                cur_lbmk= labels[mask]==i
                                count_num_labels[i]+=extra_num
                            else:
                                continue
                            
                            print('f:{}-l:{}-c:{}-ft:{}'.format(feat_3d.shape,labels[mask].shape,coords[mask].shape,feats[mask].shape))
                            # import pdb
                            # pdb.set_trace()
                            self.feat_all.append(feat_3d[cur_lbmk][:extra_num])
                            # self.mask_all.append(mask[cur_lbmk][:extra_num])
                            self.labels_all.append(labels[mask][cur_lbmk][:extra_num])
                            self.coords_all.append(coords[mask][cur_lbmk][:extra_num])
                            self.ft_all.append(feats[mask][cur_lbmk][:extra_num])

            print('Number of samples per class:{}'.format(count_num_labels))
                    # print('f:{}-m:{}-l:{}-c:{}-ft:{}'.format(feat_3d.shape,mask.shape,labels[mask].shape,coords[mask].shape,feats[mask].shape))
                    # import pdb
                    # pdb.set_trace()
        # import pdb
        # pdb.set_trace()
        
        self.feat_emb=torch.cat(self.feat_all)
        # self.mask_emb=torch.cat(self.mask_all)
        self.labels_emb=torch.cat(self.labels_all)
        self.coords_emb=torch.cat(self.coords_all)
        self.ft_emb=torch.cat(self.ft_all)

        # import pdb
        # pdb.set_trace()
    
       

    def __getitem__(self, index_long):
        #mask=self.mask_emb[index_long]
        
        feat_3d=self.feat_emb[index_long]
        coords=self.coords_emb[index_long]
        feats=self.ft_emb[index_long]
        labels=self.labels_emb[index_long]
       
        return  coords, feats, labels, feat_3d#, mask


    def __len__(self):
        return self.feat_emb.shape[0]#len(self.data_paths) * self.loop
    



def collation_fn(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)

    '''
    coords, feats, labels, feat_3d = list(zip(*batch))
   
    new_c=torch.stack(coords,0)
    new_f=torch.stack(feats)
 
    new_l=torch.stack(labels)
    new_f3=torch.stack(feat_3d)
   
    return new_c,new_f,new_l,new_f3#,new_m



def collation_fn_eval_all(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

    '''
    coords, feats, labels, feat_3d, mask, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        torch.cat(feat_3d), torch.cat(mask), torch.cat(inds_recons)