import argparse
import collections
import numpy as np
import time
import math
import torch
import torch.optim as optim
from sys import path
from dataloader_load import *
from model.unet import *
import sys
import os
import random
import csv
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import skimage.transform
import hdf5plugin
from torch_scatter import scatter_max, scatter_min
from transform import *
from collections import OrderedDict
from timers import CudaTimer
model = UNet(2+5*2, 1)

model.to("cuda")
new_state_dict = OrderedDict()
weights_dict = torch.load('./models/rsgnet.pt')['model_state_dict']
for param_tensor in model.state_dict():
    for k, v in weights_dict.items():
        # print(k)
        if 'module.' in k:
            if k.replace('module.','') == param_tensor:
                new_state_dict[param_tensor] = v
                # print('{}   {}'.format(param_tensor, k))
                break
model.load_state_dict(new_state_dict)
model.eval()

# filelist = ['zurich_city_09_a', 'zurich_city_09_a']
filelist = os.listdir('/root/data1/dataset/DSEC/train/images')
for filename in filelist:
    print(filename)
    event_file = '/root/data1/dataset/DSEC/train/events/' + filename + '/h5'
    if not os.path.exists(event_file):
      continue
    save_path = '/root/data1/dataset/DSEC/train/events/' + filename + '/SIF'
    os.makedirs(save_path, exist_ok=True)
    mapx = np.zeros((480, 640), dtype=np.float32)
    mapy = np.zeros((480, 640), dtype=np.float32)
    K_dist = None
    dist_coeffs = None
    R_rect0 = None
    K_rect = None
    if 'interlaken' in event_file or 'thun' in event_file:
        mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = interlaken_thun()
    if 'zurich_city' in event_file and ('00' in event_file or '01' in event_file or '02' in event_file or 
        '03' in event_file or '09' in event_file or '10' in event_file or '12' in event_file or '14' in event_file):
        mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_00_01_02_03_09_10_12_14()
    if 'zurich_city' in event_file and ('04' in event_file or '05' in event_file or '11' in event_file):
        mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_04_05_11()
    if 'zurich_city' in event_file and ('06' in event_file or '07' in event_file or '13' in event_file):
        mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_06_07_13()
    if 'zurich_city' in event_file and ('08' in event_file or '15'):
        mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_08_15()
    event_list = os.listdir(event_file)
    
    for event_name in event_list:
        # if os.path.exists(os.path.join(save_path, event_name.replace('.h5', '.npz'))):
        #   continue
        
        event_path = os.path.join(event_file, event_name)
        print(event_path)
        event_h5_file = h5py.File(event_path, 'r')
        x = np.array(event_h5_file['events']['x'])
        y = np.array(event_h5_file['events']['y'])
        p = np.array(event_h5_file['events']['p'])
        t = np.array(event_h5_file['events']['t'])
        event_h5_file.close()
        
        x2 = None
        y2 = None
        p2 = None
        t2 = None
        # print(file[-1])
        pre_name = str(int(event_name.split('.')[0]) - 1).zfill(6) + '.h5'
        if event_name != '000001.h5':
            event_path = os.path.join(event_file, pre_name)
            event_h5_file = h5py.File(event_path, 'r')
            x2 = np.array(event_h5_file['events']['x'])
            y2 = np.array(event_h5_file['events']['y'])
            p2 = np.array(event_h5_file['events']['p'])
            t2 = np.array(event_h5_file['events']['t'])
            event_h5_file.close()
            x = np.r_[x2,x]
            y = np.r_[y2,y]
            p = np.r_[p2,p]
            t = np.r_[t2,t]

        event = np.vstack([x, y, t, p.astype(np.uint8)]).T
        event = event.astype(float)
        # Account for zero polarity
        # if event[:, 3].min() >= -0.5:
        #     event[:, 3][event[:, 3] <= 0.5] = -1
        event[:, 3] = 2 * event[:, 3] - 1
        event = torch.from_numpy(event)

        # # voxel
        voxel = events_to_voxel_grid(x, y, p, t)

        # TimeStamp
        blur = reshape_then_acc_time_pol(event)
    
        mapping = cv2.initUndistortRectifyMap(K_dist, dist_coeffs, R_rect0, K_rect, resolution, cv2.CV_32FC2)[0]
        ev_voxel = voxel
        ev_blur = blur

        for i in range(voxel.shape[0]):
            ev_voxel[i,:,:] = torch.from_numpy(cv2.remap(np.array(voxel[i,:,:].squeeze(0)), mapping, None, interpolation=cv2.INTER_CUBIC))
        voxel = ev_voxel

        for i in range(blur.shape[0]):
            ev_blur[i,:,:] = torch.from_numpy(cv2.remap(np.array(blur[i,:,:].squeeze(0)), mapping, None, interpolation=cv2.INTER_CUBIC))
        blur = ev_blur
        blur = blur.permute(1,2,0)
        blur = np.array(blur)
        blur = np.divide(blur, 
                        np.amax(blur, axis=(0, 1), keepdims=True),
                        out=np.zeros_like(blur),
                        where=blur!=0)
        blur = torch.from_numpy(blur)
        blur = blur.permute(2,0,1)       
        
        blur = np.expand_dims(blur, axis=0)
        voxel = np.expand_dims(voxel, axis=0)

        with torch.no_grad():
            with CudaTimer(device=torch.from_numpy(blur).to("cuda").device, timer_name="ALL"):
                x = model([torch.from_numpy(blur).to("cuda"), torch.from_numpy(voxel).to("cuda")])
            x = torch.Tensor.cpu(x)
            x = x.squeeze(0) 
            x = x.squeeze(0) 
            # np.savez(os.path.join(save_path, event_name.replace('.h5', '')), x)
 
            x = (x / abs(x).max()) * 255
            cv2.imwrite(os.path.join(save_path,'{:06d}.png'.format(int(event_name.split('.')[0]))), np.array(x))
