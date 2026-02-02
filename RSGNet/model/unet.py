import os
import sys
import torch
import pkg_resources
import math
import torch as th
import torch.nn.functional as F
import torchvision
from torch import nn
   
# 平滑损失
def gradient_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h =  (x.size()[2]-1) * x.size()[3]
    count_w = x.size()[2] * (x.size()[3] - 1)
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size

def mse_loss(x, y):
    return torch.mean((x - y) ** 2)

def l1_loss(x, y):
    return torch.mean(abs(x - y))
# ncc损失
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        
        self.win = win

    def loss(self, y_true, y_pred):
        num_stab_const = 3e-4 # numerical stability constant
        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")
        # sum_filt = torch.ones([1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        cross = IJ_sum - I_sum*J_sum/win_size + num_stab_const
        I_var = I2_sum - I_sum**2/win_size + num_stab_const
        J_var = J2_sum - J_sum**2/win_size + num_stab_const
        cc = cross/((I_var*J_var)**0.5)
        return -torch.mean(cc)
def mce(x):
    valid_mask = x > 0
    avgpool = nn.AdaptiveAvgPool2d(1)
    mean = avgpool(x)
    var = torch.pow(x - mean, 2) * valid_mask
    return -torch.mean(var)

def local_mce(x):
    avgpool = nn.AvgPool2d(5, stride=1, padding=2)
    mean = avgpool(x)
    # print(mean.shape)
    valid_mask = x > 0
    var = torch.pow(x - mean, 2) / (5 * 5) * valid_mask
    return -torch.mean(var)



def closest_larger_multiple_of_minimum_size(size, minimum_size):
    return int(math.ceil(size / minimum_size) * minimum_size)

class SizeAdapter(object):
    """Converts size of input to standard size.
    Practical deep network works only with input images
    which height and width are multiples of a minimum size.
    This class allows to pass to the network images of arbitrary
    size, by padding the input to the closest multiple
    and unpadding the network's output to the original size.
    """

    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

    def pad(self, network_input):
        """Returns "network_input" paded with zeros to the "standard" size.
        The "standard" size correspond to the height and width that
        are closest multiples of "minimum_size". The method pads
        height and width  and and saves padded values. These
        values are then used by "unpad_output" method.
        """
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = (self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (self._closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        """Returns "network_output" cropped to the original size.
        The cropping is performed using values save by the "pad_input"
        method.
        """
        return network_output[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]

class up(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(up, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(th.cat((x, skpCn), 1)), negative_slope=0.1)
        return x


class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        super(down, self).__init__()
        self.conv1 = nn.Conv2d(
            inChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )
        self.conv2 = nn.Conv2d(
            outChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class UNet(nn.Module):
    """Modified version of Unet from SuperSloMo.
    
    Difference : 
    1) there is an option to skip ReLU after the last convolution.
    2) there is a size adapter module that makes sure that input of all sizes
       can be processed correctly. It is necessary because original
       UNet can process only inputs with spatial dimensions divisible by 32.
    """

    def __init__(self, inChannels, outChannels, ends_with_relu=True):
        super(UNet, self).__init__()
        self._ends_with_relu = ends_with_relu
        self._size_adapter = SizeAdapter(minimum_size=32)
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)
        self.ncc_loss = NCC()


    def forward(self, x):

        if self.training:
            blur, voxel, img_rgb = x
        else:
            blur, voxel = x
        x = th.cat((blur, voxel), 1)

        # Size adapter spatially augments input to the size divisible by 32.
        x = self._size_adapter.pad(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)

        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2) 
        s4 = self.down3(s3) 
        s5 = self.down4(s4)
        s6 = self.down5(s5)
        s7 = self.up1(s6, s5)
        s8 = self.up2(s7, s4)
        s9 = self.up3(s8, s3)
        x = self.up4(s9, s2)
        x = self.up5(x, s1)

        # Note that original code has relu et the end.
        if self._ends_with_relu == True:
           
            x1 = self.conv3(x)
            zero = torch.zeros_like(x1)
            x2 = torch.tanh(x1)
            x = torch.where(x1 > 0, x2, zero)
        else:
            x = self.conv3(x)
        # Size adapter crops the output to the original size.
        x = self._size_adapter.unpad(x)
        if self.training:
            grad_loss = -gradient_loss(x)
            sim_loss = self.ncc_loss.loss(img_rgb, x)
            mce_loss = torch.Tensor(0).cuda()
            Sort_loss = torch.Tensor(0).cuda()
            return mce_loss, grad_loss, sim_loss, Sort_loss

        else: 
            return x
