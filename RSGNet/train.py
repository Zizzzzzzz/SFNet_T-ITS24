import argparse
import collections
import numpy as np
import time
import math
import torch
import torch.optim as optim
from torchvision import transforms
from sys import path

from dataloader import *
from model.unet import *
from torch.utils.data import DataLoader
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate
from multi_train_utils.distributed_utils import reduce_value, is_main_process
from collections import OrderedDict
# assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main(args=None):
    base_dir = '/yn/dataset/DSEC'
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', default=f'./label/train_new.csv',
                        help='Path to file containing training annotations (see readme)')
    parser.add_argument('--root_img',default=f'/root/data1/dataset/DSEC/train/images',help='dir to root rgb images')
    parser.add_argument('--root_event', default=f'/root/data1/dataset/DSEC/train/events',help='dir to toot event files in dsec directory structure')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=60)
    parser.add_argument('--continue_training', help='load a pretrained file', default=False)
    parser.add_argument('--checkpoint', help='location of pretrained file', default='')

    parser.add_argument('--syncBN', type=bool, default=True)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser = parser.parse_args(args)
    init_distributed_mode(args=parser)

    rank = parser.rank
    device = torch.device(parser.device)
    if rank == 0:
        print(device)

    if parser.dataset == 'csv':
        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')
        dataset_train = CSVDataset_event(train_file=parser.csv_train, root_event_dir=parser.root_event,root_img_dir=parser.root_img)
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    model = UNet(2+5*2, 1)
 
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

    batch_size = 8
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size=batch_size, drop_last=True)

    dataloader_train = DataLoader(dataset_train, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=batch_size, collate_fn=collater)#, worker_init_fn=_init_fn)

    epoch_total = 0
    epoch_loss_all =[]

    use_gpu = True
    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4*3/8)
    if parser.continue_training:
        checkpoint = torch.load(parser.checkpoint)
        epoch_loss_all = checkpoint['loss']
        epoch_total = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if rank == 0:
            print('training sensor fusion model')
            print(epoch_loss_all)
    else:
        epoch_total = 0
        epoch_loss_all =[]

    if parser.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[parser.gpu])

    model.training = True
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)


    model.train()
    if rank == 0:
        print('Num training images: {}'.format(len(dataset_train)))
    num_batches = 0
    start = time.time()

    if rank == 0:
        print(time_since(start))
    
    for epoch_num in range(epoch_total, parser.epochs):
        train_sampler.set_epoch(epoch_num)
        optimizer.zero_grad()
        epoch_loss = []
        sim_losses = []
        mce_losses = []
        grad_losses = []
        sort_losses = []
        model.train()

        epoch_total += 1 
        for iter_num, data in enumerate(dataloader_train):

            mce_loss, grad_loss, sim_loss, sort_loss = model([data['blurs'].cuda().float(),data['voxels'].cuda().float(),data['img_rgbs'].cuda().float()])

            # grad_loss = grad_loss.mean()
            mce_loss = mce_loss.mean()
            grad_loss = grad_loss.mean()
            sim_loss = sim_loss.mean()
            sort_loss = sort_loss.mean()

            loss = 0.05 * grad_loss  + sim_loss

            if bool(loss == 0):
                continue

            loss.backward()
            loss = reduce_value(loss, average=True)
            grad_loss = reduce_value(grad_loss, average=True)
            mce_loss = reduce_value(mce_loss, average=True)
            sim_loss = reduce_value(sim_loss, average=True)
            sort_loss = reduce_value(sort_loss, average=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            
            if rank == 0:
                print(
                    '[sensor fusion homographic] [{}], Epoch: {} | Iteration: {} | mce_loss: {:1.5f} | grad_loss: {:1.5f} | sim_loss: {:1.5f} | sort_loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        time_since(start), epoch_num, iter_num, float(1*mce_loss), float(0.05*grad_loss), float(sim_loss), float(sort_loss), float(loss)))
            epoch_loss.append(loss.item())
            sim_losses.append(sim_loss.item())
            mce_losses.append(mce_loss.item())
            grad_losses.append(grad_loss.item())
            sort_losses.append(sort_loss.item())

            del grad_loss
            del mce_loss
            del sim_loss
            del sort_loss

        if rank == 0:
            torch.save({'epoch': epoch_total, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': np.append(epoch_loss_all,np.mean(epoch_loss))}, f'./models/{epoch_total}.pt')

        scheduler.step(np.mean(epoch_loss))

        
    if rank == 0:
        torch.save({'epoch': epoch_total, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': np.append(epoch_loss_all,epoch_loss)}, f'./models/{epoch_total}.pt')
    cleanup()


if __name__ == '__main__':
    main()
