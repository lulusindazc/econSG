import os
import time
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import collation_fn, Point3DMultiLoader
from model import Autoencoder
from torch.utils.tensorboard import SummaryWriter
import argparse
from util import export_pointcloud, get_palette, \
    convert_labels_with_palette, extract_text_feature, visualize_labels
from label_constants import *

torch.autograd.set_detect_anomaly(True)

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()



def worker_init_fn(worker_id):
    '''Worker initialization.'''
    random.seed(time.time() + worker_id)


class AverageMeter():
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def precompute_text_related_properties(labelset_name):
    '''pre-compute text features, labelset, palette, and mapper.'''

    if 'scannet' in labelset_name:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other' # change 'other furniture' to 'other'
        palette = get_palette(colormap='scannet')
    else: # an arbitrary dataset, just use a large labelset
        labelset = list(LABELS_160)
        palette = get_palette(colormap='matterport_160')

    mapper = None

    text_features = extract_text_feature(labelset, args)
    # labelset.append('unknown')
    labelset.append('unlabeled')
    return text_features, labelset, mapper, palette


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../scannet_3d') # '~/project/tmp_dataset/scannet_multiview_lseg'
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0007)
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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--voxel_size', type=int, default=0.02)
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--dataset_name', type=str, default='scannet')
    parser.add_argument('--feature_2d_extractor', type=str, default='openseg',help='[lseg,openseg]')
    parser.add_argument('--loop', type=int, default=5)
    parser.add_argument('--data_root_2d_fused_feature', type=str, default='../scannet_multiview_openseg')
    args = parser.parse_args()
    # dataset_path = args.dataset_path
    num_epochs = args.num_epochs
    
    save_dir= f'ckpt/{args.dataset_name}_multiview'
    os.makedirs(save_dir, exist_ok=True)
    
    train_data = Point3DMultiLoader(datapath_prefix=args.dataset_path,
                                    datapath_prefix_feat=args.data_root_2d_fused_feature,
                                    voxel_size=args.voxel_size,
                                    split='train', aug=True,
                                    memcache_init=False, loop=args.loop,
                                    input_color=False
                                    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data) if args.distributed else None
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                            shuffle=(train_sampler is None),
                                            num_workers=4, pin_memory=True,
                                            sampler=train_sampler,
                                            drop_last=True, collate_fn=collation_fn,
                                            worker_init_fn=worker_init_fn)

    test_loader = DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        drop_last=False  
    )
    
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).cuda()
    ckpt_path = os.path.join(save_dir,'best_ckpt.pth')
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)

    labelset_name='scannet_3d'
    text_features, labelset, mapper, palette = \
        precompute_text_related_properties(labelset_name)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # logdir = f'ckpt/{args.dataset_name}/{output}'
    tb_writer = SummaryWriter(save_dir)

    max_iter = num_epochs * len(train_loader)

    best_eval_loss = 100.0
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        end = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()

        model.train()
        for idx, data_batch  in enumerate(train_loader):
            coords, feats, labels, feat_3d = data_batch
            data  = feat_3d.cuda(non_blocking=True) 
            locs_in = coords#.cuda(non_blocking=True) 
            colors = feats#.cuda(non_blocking=True) 
            assert coords.shape[0]==feat_3d.shape[0]
            if data.dtype == torch.float16:
                data.data = data.data.to(torch.float32)
            data_time.update(time.time() - end)

            outputs_dim3 = model.encode(data)
            outputs = model.decode(outputs_dim3)

            pred = outputs.half() @ text_features.t()
            logits_pred = torch.max(pred, 1)[1].cpu()
            pred_gt= data.half() @ text_features.t()
            gt_pred=torch.max(pred_gt, 1)[1].cpu()
        
            l2loss = l2_loss(outputs, data) 
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss * 0.001
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_iter = epoch * len(train_loader) + idx
            # print('{}-{}-{}-{}'.format(l2loss.item(),cosloss.item(),loss.item(),global_iter))
            tb_writer.add_scalar('train_loss/l2_loss', l2loss.item(), global_iter)
            tb_writer.add_scalar('train_loss/cos_loss', cosloss.item(), global_iter)
            tb_writer.add_scalar('train_loss/total_loss', loss.item(), global_iter)
            # tb_writer.add_histogram("feat", outputs.detach().cpu().numpy(), global_iter)

            batch_time.update(time.time() - end)
            loss_meter.update(loss.item(), args.batch_size)
             # calculate remain time
            remain_iter = max_iter - global_iter
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(
                int(t_h), int(t_m), int(t_s))
            if idx % (len(train_loader)/2)==0 and idx!=0:
                print('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'Loss {loss_meter.val:.4f} '.format(epoch + 1,num_epochs, idx + 1, len(train_loader),
                                                                batch_time=batch_time, data_time=data_time,
                                                                remain_time=remain_time,
                                                                loss_meter=loss_meter))

        if epoch% 10==0 and epoch!=0:
            eval_loss = 0.0
            model.eval()
            for _, data_batch_evl in enumerate(test_loader):
                _, _, _, feature= data_batch_evl
                data_e= feature.cuda(
                non_blocking=True)
                # data = feature.to("cuda:0")
                if data_e.dtype == torch.float16:
                    data_e.data = data_e.data.to(torch.float32)
                with torch.no_grad():
                    outputs = model(data_e) 
                loss = l2_loss(outputs, data_e) + cos_loss(outputs, data_e)
                eval_loss += loss * len(feature)
            eval_loss = eval_loss / len(train_data)
            print("eval_loss:{:.8f}".format(eval_loss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_dir,'best_ckpt.pth'))
                
            # if epoch % 10 == 0:
            #     torch.save(model.state_dict(), f'ckpt/{args.dataset_name}/{args.output}/{epoch}_ckpt.pth')
            
    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))