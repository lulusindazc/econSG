import os
import multiprocessing as mp
import numpy as np
import plyfile
import torch
import argparse



def process_one_scene(fn):
    '''process one scene.'''

    scene_name = fn.split('/')[-1].split('_mesh')[0]
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    
    colors = np.ascontiguousarray(v[:, -3:]) / 255.0 # / 127.5 - 1
    
    labels = 255*np.ones((coords.shape[0], ), dtype=np.int32) #np.array([x[-1] for x in a.elements[1]]).astype(np.int32) #np.array(a.elements[1]["object_id"]).astype(np.int32) 
    print(coords.shape)
    print(colors.shape)
    print(labels.shape)
    torch.save((coords, colors, labels),
            os.path.join(out_dir,  scene_name + '.pth'))
    print(os.path.join(out_dir,  scene_name + '.pth'))


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines



if __name__ == '__main__':

    #! YOU NEED TO MODIFY THE FOLLOWING
    #####################################
    # split = 'val' # choose between 'train' | 'val'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='replica_3d') #required=True)
    parser.add_argument('--in_path', type=str, default='Replica') #required=True)
    parser.add_argument('--scene_name', type=str, default='office0')
    args = parser.parse_args()

    out_dir = args.dataset_path 
    in_path = args.in_path 
    scene_list = [args.scene_name] 

    os.makedirs(out_dir, exist_ok=True)

    files = []
    for scene in scene_list:
        files.append(os.path.join(in_path, '{}_mesh_semantic.ply'.format(scene)))

    p = mp.Pool(processes=mp.cpu_count())
    print(files)
    p.map(process_one_scene, files)
    p.close()
    p.join()