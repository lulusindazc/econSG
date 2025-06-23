import json
import os
import numpy as np
import glob
import math
import  argparse


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='replica_2d') #required=True)
    parser.add_argument('--scene_name', type=str, default='room0')
    args = parser.parse_args()


    root_dir= args.dataset_path 

    scene = args.scene_name #'scene0494_00'
    # seq = 'Sequence_1'
    data_dir=os.path.join(root_dir,scene)#,seq)
    traj_file = os.path.join(data_dir, "intrinsics.txt")
    camera_file= os.path.join(data_dir, "pose") 
    rgb_dir = os.path.join(data_dir, "color")
    depth_dir = os.path.join(data_dir, "depth")  # depth is in mm uint
    label_dir = os.path.join(data_dir, "label")  # depth is in mm uint
    object_dir = os.path.join(data_dir, "object_mask")  # depth is in mm uint


    rgb_list = sorted(glob.glob(rgb_dir + '/*.png'), key=lambda file_name: int((file_name.split('/')[-1]).split(".")[0]))
    depth_list = sorted(glob.glob(depth_dir + '/*.png'), key = lambda file_name: int((file_name.split('/')[-1]).split(".")[0]))
    label_list = sorted(glob.glob(label_dir + '/*.png'), key = lambda file_name: int((file_name.split('/')[-1]).split(".")[0]))
    sammask_list= sorted(glob.glob(object_dir + '/*.png'), key = lambda file_name: int((file_name.split('/')[-1]).split(".")[0]))
    pose_list = sorted(glob.glob(camera_file + '/*.txt'), key = lambda file_name: int((file_name.split('/')[-1]).split(".")[0]))


    total_num = len(rgb_list)

    ########################
    
    ###############semantic-nerf/DM-nerf/panoptic
    # total_num = 900
    step = 5
    train_ids = list(range(0, total_num, step))
    test_ids = [x + step // 2 for x in train_ids]
    ###########################


    train_num = len(train_ids)
    test_num=len(test_ids)

    # import pdb
    # pdb.set_trace()

    intrinsic = np.loadtxt(traj_file)[:3, :3]
    focal_x = intrinsic[0,0]
    focal_y = intrinsic[1,1]
    w= 640 #320
    h= 480 #360 #240
    angle_x=math.atan(w / (focal_x * 2)) * 2
    angle_y=math.atan(h / (focal_y * 2)) * 2


    def write_into_file(split,ids):
        file_cont_train = dict(frames=[])
        
        for idx in ids:
        
            rgb_file = rgb_list[idx].replace(data_dir+'/','')
            depth_file = depth_list[idx].replace(data_dir+'/','')
            label_file = label_list[idx].replace(data_dir+'/','')
            object_file=sammask_list[idx].replace(data_dir+'/','')

            pose = np.loadtxt(pose_list[idx])[:4, :4]

            cur_dict= {
                'file_path': rgb_file,
                'depth': depth_file,
                'label': label_file,
                'object':object_file,
                'transform_matrix': pose,
            }
            # if 'file_path' not in file_cont['frames']:
            file_cont_train['frames'].append(cur_dict)

        file_cont_train.update({
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
        })


        out_file=os.path.join(data_dir,'transforms_{}.json'.format(split))
        with open(out_file,'w') as f:
            json.dump(file_cont_train,f,cls=NpEncoder,indent='\t')



    write_into_file('train',train_ids)
    write_into_file('test',test_ids)
    # write_into_file('val',val_ids)