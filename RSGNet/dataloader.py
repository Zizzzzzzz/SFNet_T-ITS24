from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler
import torch
import skimage.transform
import hdf5plugin
from torch_scatter import scatter_max, scatter_min
from transform import *
class CSVDataset_event(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, root_event_dir, root_img_dir, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.transform = transform
        self.img_dir = root_img_dir
        self.event_dir = root_event_dir

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_names = self._read_annotations(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise (ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = 5 * 2
        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        blur, voxel, img_rgb = self.load_image(idx)

        sample = {'blur': blur,'voxel': voxel, 'img_rgb': img_rgb}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def load_image(self, image_index):
        # print(self.image_names[image_index])
        file = self.image_names[image_index].split('/')

        img_file = os.path.join(self.img_dir,file[-3],'images/left/rectified',file[-1].replace('.h5','.png'))
        img_rgb = cv2.imread(img_file)

        mapx = np.zeros((480, 640), dtype=np.float32)
        mapy = np.zeros((480, 640), dtype=np.float32)
        K_dist = None
        dist_coeffs = None
        R_rect0 = None
        K_rect = None
        if 'interlaken' in img_file or 'thun' in img_file:
            mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = interlaken_thun()
        if 'zurich_city' in img_file and ('00' in img_file or '01' in img_file or '02' in img_file or 
            '03' in img_file or '09' in img_file or '10' in img_file or '12' in img_file or '14' in img_file):
            mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_00_01_02_03_09_10_12_14()
        if 'zurich_city' in img_file and ('04' in img_file or '05' in img_file or '11' in img_file):
            mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_04_05_11()
        if 'zurich_city' in img_file and ('06' in img_file or '07' in img_file or '13' in img_file):
            mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_06_07_13()
        if 'zurich_city' in img_file and ('08' in img_file or '15'):
            mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_08_15()

        gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        # -------------------Sobel边缘检测------------------------
        x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1)
        Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
        Scale_absY = cv2.convertScaleAbs(y)
        img_rgb = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        img_rgb = cv2.remap(img_rgb, mapx, mapy, cv2.INTER_LINEAR)

        img_rgb = np.expand_dims(img_rgb, axis=2)
        # print(img_rgb.shape)
        img_rgb = img_rgb / img_rgb.max()
        

        # read h5 file
        # blur
        event_file = os.path.join(self.event_dir, file[0], 'h5', file[-1])
        event_h5_file = h5py.File(event_file, 'r')
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
        pre_name = str(int(file[-1].split('.')[0]) - 1).zfill(6) + '.h5'
        if file[-1] != '000001.h5':
            event_path = os.path.join(self.event_dir, file[0], 'h5', pre_name)
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
        event = event.astype(np.float)
        # Account for zero polarity
        if event[:, 3].min() >= -0.5:
            event[:, 3][event[:, 3] <= 0.5] = -1
        # event[:, 3] = 2 * event[:, 3] - 1
        event = torch.from_numpy(event)

        # # voxel
        voxel = self.events_to_voxel_grid(x, y, p, t)

        # TimeStamp
        blur = reshape_then_acc_time_pol(event)
        
        mapping = cv2.initUndistortRectifyMap(K_dist, dist_coeffs, R_rect0, K_rect, resolution, cv2.CV_32FC2)[0]
        ev_voxel = voxel
        ev_blur = blur

        for i in range(voxel.shape[0]):
            ev_voxel[i,:,:] = torch.from_numpy(cv2.remap(np.array(voxel[i,:,:].squeeze(0)), mapping, None, interpolation=cv2.INTER_CUBIC))
        voxel = ev_voxel
        voxel = voxel.permute(1,2,0)
       

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
        
        return blur,voxel,img_rgb

    def _read_annotations(self, csv_reader):
        result = []

        for line, row in enumerate(csv_reader):
            line += 1
            try:
                img_file= row[:1]
            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'img_file\' or \'img_file,,,,,\''.format(line)),
                           None)
            # if img_file != 'zurich_city_11_c':
            #     continue
            # print(img_file)

            if img_file not in result:
                result.append(img_file[0])
        return result

    def image_aspect_ratio(self, image_index):
        print(self.image_names[image_index])
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)

def collater(batch):
    batch_dict = {}
    blurs = []
    voxels = []
    img_rgbs = []
    for iter_num, data in enumerate(batch):
        blurs.append(data['blur'])
        voxels.append(data['voxel'])
        img_rgbs.append(torch.from_numpy(data['img_rgb']))

    batch_dict['blurs'] = torch.stack(blurs, 0).permute(0, 3, 1, 2)
    batch_dict['voxels'] = torch.stack(voxels, 0).permute(0, 3, 1, 2)
    batch_dict['img_rgbs'] = torch.stack(img_rgbs, 0).permute(0, 3, 1, 2)
    return batch_dict