import argparse, time, os
import json
import random
import imageio
import functools
import sys
from datetime import datetime
import numpy as np
import cv2
# from utils import util # -> calculating psnr or ssim
# from solvers import create_solver # -> test file resource
# import data.common as common
from tqdm import tqdm
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as thutil
import pandas as pd
import scipy.misc as misc
from collections import OrderedDict



# *** data folder
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
BINARY_EXTENSIONS = ['.npy']
BENCHMARK = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109', 'DIV2K', 'DF2K']

# LR_dataset file
class LRDataset(data.Dataset):
    '''
    Read LR images only in test phase.
    '''

    def name(self):
        return find_benchmark(self.opt['dataroot_LR'])


    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.scale = self.opt['scale']
        self.paths_LR = None

        # read image list from image/binary files
        # self.paths_LR = get_image_paths(opt['data_type'], opt['dataroot_LR'])
        # assert self.paths_LR, '[Error] LR paths are empty.'


    def __getitem__(self, idx):
        # get LR image
        lr, lr_path = self._load_file(idx)
        lr_tensor = np2Tensor([lr], self.opt['rgb_range'])[0]
        return {'LR': lr_tensor, 'LR_path': lr_path}


    def __len__(self):
        return len(self.paths_LR)


    def _load_file(self, idx):
        lr_path = self.paths_LR[idx]
        lr = read_img(lr_path, self.opt['data_type'])

        return lr, lr_path

# LRHR_dataset file
class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return find_benchmark(self.opt['dataroot_LR'])


    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR, self.paths_LR = None, None

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 2

        # read image list from image/binary files
        self.paths_HR = get_image_paths(self.opt['data_type'], self.opt['dataroot_HR'])
        self.paths_LR = get_image_paths(self.opt['data_type'], self.opt['dataroot_LR'])

        assert self.paths_HR, '[Error] HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                len(self.paths_LR), len(self.paths_HR))


    def __getitem__(self, idx):
        lr, hr, lr_path, hr_path = self._load_file(idx)
        if self.train:
            lr, hr = self._get_patch(lr, hr)
        lr_tensor, hr_tensor = np2Tensor([lr, hr], self.opt['rgb_range'])
        return {'LR': lr_tensor, 'HR': hr_tensor, 'LR_path': lr_path, 'HR_path': hr_path}


    def __len__(self):
        if self.train:
            return len(self.paths_HR) * self.repeat
        else:
            return len(self.paths_LR)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        lr = read_img(lr_path, self.opt['data_type'])
        hr = read_img(hr_path, self.opt['data_type'])

        return lr, hr, lr_path, hr_path


    def _get_patch(self, lr, hr):

        LR_size = self.opt['LR_size']
        # random crop and augment
        lr, hr = get_patch(
            lr, hr, LR_size, self.scale)
        lr, hr = augment([lr, hr])
        lr = add_noise(lr, self.opt['noise'])

        return lr, hr

# common file

def get_patch(img_in, img_tar, patch_size, scale):
    ih, iw = img_in.shape[:2]
    oh, ow = img_tar.shape[:2]

    ip = patch_size

    if ih == oh:
        tp = ip
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = ix, iy
    else:
        tp = ip * scale
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar

def add_noise(x, noise='.'):
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]

def read_img(path, data_type):
    # read image by misc or from .npy
    # return: Numpy float32, HWC, RGB, [0,255]
    if data_type == 'img':
        img = imageio.imread(path, pilmode='RGB')
    elif data_type.find('npy') >= 0:
        img = np.load(path)
    else:
        raise NotImplementedError

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        # if img.shape[2] == 3: # for opencv imread
        #     img = img[:, :, [2, 1, 0]]
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255.)

        return tensor

    return [_np2Tensor(_l) for _l in l]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(dataset):
    # phase = dataset_opt['phase']
    batch_size = 1
    shuffle = False
    num_workers = 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def dataset(dataset_opt):
    mode = dataset_opt['mode'].upper()
    if mode == 'LR':
        D = LRDataset()
    elif mode == 'LRHR':
        D = LRHRDataset()
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % mode)
    dataset = D(dataset_opt)
    print('===> [%s] Dataset is created.' % (mode))
    return dataset

def is_binary_file(filename):
    return any(filename.endswith(extension) for extension in BINARY_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '[%s] has no valid image file' % path
    return images

def _get_paths_from_binary(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_binary_file(fname):
                binary_path = os.path.join(dirpath, fname)
                files.append(binary_path)
    assert files, '[%s] has no valid binary file' % path
    return files

def get_image_paths(data_type, dataroot):
    paths = None
    if dataroot is not None:
        if data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        elif data_type == 'npy':
            if dataroot.find('_npy') < 0 :
                old_dir = dataroot
                dataroot = dataroot + '_npy'
                if not os.path.exists(dataroot):
                    print('===> Creating binary files in [%s]' % dataroot)
                    os.makedirs(dataroot)
                    img_paths = sorted(_get_paths_from_images(old_dir))
                    path_bar = tqdm(img_paths)
                    for v in path_bar:
                        img = imageio.imread(v, pilmode='RGB')
                        ext = os.path.splitext(os.path.basename(v))[-1]
                        name_sep = os.path.basename(v.replace(ext, '.npy'))
                        np.save(os.path.join(dataroot, name_sep), img)
                else:
                    print('===> Binary files already exists in [%s]. Skip binary files generation.' % dataroot)

            paths = sorted(_get_paths_from_binary(dataroot))

        else:
            raise NotImplementedError("[Error] Data_type [%s] is not recognized." % data_type)
    return paths

def find_benchmark(dataroot):
    bm_list = [dataroot.find(bm)>=0 for bm in BENCHMARK]
    if not sum(bm_list) == 0:
        bm_idx = bm_list.index(True)
        bm_name = BENCHMARK[bm_idx]
    else:
        bm_name = 'MyImage'
    return bm_name
# *** networks folder + blocks folder
def create_model(opt):
    if opt['mode'] == 'sr':
        net = define_net(opt['networks'])
        return net
    else:
        raise NotImplementedError("The mode [%s] of networks is not recognized." % opt['mode'])
def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!'%act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type =='bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!'%norm_type)
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!'%pad_type)
    return layer

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict'%sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)
def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,\
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)
def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, \
                act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, deconv)
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False

def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
    padding = (kernel_size-1) // 2
    return padding

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            torch.nn.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        torch.nn.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        torch.nn.normal_(m.weight.data, 1.0, std)
        torch.nn.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            torch.nn.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        torch.nn.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        torch.nn.constant_(m.weight.data, 1.0)
        m.weight.data *= scale
        torch.nn.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            torch.nn.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        torch.nn.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        torch.nn.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class ResidualDenseBlock_8C(nn.Module):
    '''
    Residual Dense Block
    style: 8 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero',norm_type=None, act_type='relu', mode='CNA'):
        super(ResidualDenseBlock_8C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = ConvBlock(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = ConvBlock(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = ConvBlock(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv5 = ConvBlock(nc+4*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv6 = ConvBlock(nc+5*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv7 = ConvBlock(nc+6*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv8 = ConvBlock(nc+7*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv9 = ConvBlock(nc+8*gc, nc, 1, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x7 = self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1))
        x8 = self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1))
        x9 = self.conv9(torch.cat((x, x1, x2, x3, x4, x5, x6, x7, x8), 1))
        return x9.mul(0.2) + x

class GFMRDB(nn.Module):
    def __init__(self, num_features, num_blocks, num_refine_feats, num_reroute_feats, act_type, norm_type=None):
        super(GFMRDB, self).__init__()

        self.num_refine_feats = num_refine_feats
        self.num_reroute_feats = num_reroute_feats

        self.RDBs_list = nn.ModuleList([ResidualDenseBlock_8C(
            num_features, kernel_size=3, gc=num_features, act_type=act_type
            ) for _ in range(num_blocks)])

        self.GFMs_list = nn.ModuleList([
                ConvBlock(
                    in_channels=num_reroute_feats*num_features, out_channels=num_features, kernel_size=1,
                    norm_type=norm_type, act_type=act_type
                ),
                ConvBlock(
                    in_channels=2*num_features, out_channels=num_features, kernel_size=1,
                    norm_type=norm_type, act_type=act_type)
            ])


    def forward(self, input_feat, last_feats_list):

        cur_feats_list = []

        if len(last_feats_list) == 0:
            for b in self.RDBs_list:
                input_feat = b(input_feat)
                cur_feats_list.append(input_feat)
        else:
            for idx, b in enumerate(self.RDBs_list):

                # refining the lowest-level features
                if idx < self.num_refine_feats:
                    select_feat = self.GFMs_list[0](torch.cat(last_feats_list, 1))
                    input_feat = self.GFMs_list[1](torch.cat((select_feat, input_feat), 1))

                input_feat = b(input_feat)
                cur_feats_list.append(input_feat)

        # rerouting the highest-level features
        return cur_feats_list[-self.num_reroute_feats:]


class GMFN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_blocks,
                 num_reroute_feats, num_refine_feats, upscale_factor, act_type = 'prelu', norm_type = None):
        super(GMFN, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        else:
            raise ValueError("upscale_factor must be 2,3,4.")

        self.num_features = num_features
        self.num_steps = num_steps
        self.upscale_factor = upscale_factor

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # initial low-level feature extraction block
        self.conv_in = ConvBlock(in_channels, 4*num_features, kernel_size=3, act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4*num_features, num_features, kernel_size=1, act_type=act_type, norm_type=norm_type)

        # multiple residual dense blocks (RDBs) and multiple gated feedback modules (GFMs)
        self.block = GFMRDB(num_features, num_blocks, num_refine_feats, num_reroute_feats,
                            act_type=act_type, norm_type=norm_type)

        # reconstruction block
        self.upsample = nn.functional.interpolate
        self.out = DeconvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                               act_type='prelu', norm_type=norm_type)
        self.conv_out = ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type)

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)


    def forward(self, lr_img):
        lr_img = self.sub_mean(lr_img)
        up_lr_img = self.upsample(lr_img, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        init_feat = self.feat_in(self.conv_in(lr_img))

        sr_imgs = []
        last_feats_list = []

        for _ in range(self.num_steps):
            last_feats_list = self.block(init_feat, last_feats_list)
            out = torch.add(up_lr_img, self.conv_out(self.out(last_feats_list[-1])))
            out = self.add_mean(out)
            sr_imgs.append(out)

        return sr_imgs


    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name.replace('module.','')
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('out') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('out') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

def define_net(opt):

    which_model: object = opt['which_model'].upper()
    print('===> Building network [%s]...'%which_model)

    if which_model.find('GMFN') >= 0:
        net = GMFN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                               num_features=opt['num_features'], num_steps=opt['num_steps'], num_blocks=opt['num_blocks'],
                               num_reroute_feats=opt['num_reroute_feats'], num_refine_feats=opt['num_refine_feats'],
                               upscale_factor=opt['scale'])
    else:
        raise NotImplementedError("Network [%s] is not recognized." % which_model)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    return net

# *** utils folder
def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def quantize(img, rgb_range):
    pixel_range = 255. / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, 255).round()

def Tensor2np(tensor_list, rgb_range):

    def _Tensor2numpy(tensor, rgb_range):
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        return array

    return [_Tensor2numpy(tensor, rgb_range) for tensor in tensor_list]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('[Warning] Path [%s] already exists. Rename it to [%s]' % (path, new_name))
        os.rename(path, new_name)
    os.makedirs(path)
# *** solvers folder

def create_solver(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver(opt)
    else:
        raise NotImplementedError

    return solver
class BaseSolver(object):
    Tensor: object

    def __init__(self, opt):
        self.opt = opt
        self.scale = opt['scale']
        self.is_train = opt['is_train']
        self.use_chop = opt['use_chop']
        self.self_ensemble = opt['self_ensemble']
        self.use_cl = True if opt['use_cl'] else False

        # GPU verify
        self.use_gpu = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor

        # for better training (stablization and less GPU memory usage)
        self.last_epoch_loss = 1e8
        self.skip_threshold = opt['solver']['skip_threshold']
        # save GPU memory during training
        self.split_batch = opt['solver']['split_batch']

        # experimental dirs
        self.exp_root = opt['path']['exp_root']
        self.checkpoint_dir = opt['path']['epochs']
        self.records_dir = opt['path']['records']
        self.visual_dir = opt['path']['visual']

        # log and vis scheme
        self.save_ckp_step = opt['solver']['save_ckp_step']
        self.save_vis_step = opt['solver']['save_vis_step']

        self.best_epoch = 0
        self.cur_epoch = 1
        self.best_pred = 0.0

    def feed_data(self, batch):
        pass

    def train_step(self):
        pass

    def test(self):
        pass

    def _forward_x8(self, x, forward_function):
        pass

    def _overlap_crop_forward(self, upscale):
        pass

    def get_current_log(self):
        pass

    def get_current_visual(self):
        pass

    def get_current_learning_rate(self):
        pass

    def set_current_log(self, log):
        pass

    def update_learning_rate(self, epoch):
        pass

    def save_checkpoint(self, epoch, is_best):
        pass

    def load(self):
        pass

    def save_current_visual(self, epoch, iter):
        pass

    def save_current_log(self):
        pass

    def print_network(self):
        pass

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))

        return s, n

class SRSolver(BaseSolver):
    def __init__(self, opt):
        super(SRSolver, self).__init__(opt)
        self.train_opt = opt['solver']
        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.SR = None

        self.records = {'train_loss': [],
                        'val_loss': [],
                        'psnr': [],
                        'ssim': [],
                        'lr': []}

        self.model = create_model(opt)

        if self.is_train:
            self.model.train()

            # set cl_loss
            if self.use_cl:
                self.cl_weights = self.opt['solver']['cl_weights']
                assert self.cl_weights, "[Error] 'cl_weights' is not be declared when 'use_cl' is true"

            # set loss
            loss_type = self.train_opt['loss_type']
            if loss_type == 'l1':
                self.criterion_pix = nn.L1Loss()
            elif loss_type == 'l2':
                self.criterion_pix = nn.MSELoss()
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!'%loss_type)

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()

            # set optimizer
            weight_decay = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer = optim.Adam(self.model.parameters(),
                                            lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!' % optim_type)

            # set lr_scheduler
            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.train_opt['lr_steps'],
                                                                self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError('Only MultiStepLR scheme is supported!')

        self.load()
        self.print_network()

        print('===> Solver Initialized : [%s] || Use CL : [%s] || Use GPU : [%s]'%(self.__class__.__name__,
                                                                                       self.use_cl, self.use_gpu))
        if self.is_train:
            print("optimizer: ", self.optimizer)
            print("lr_scheduler milestones: %s   gamma: %f"%(self.scheduler.milestones, self.scheduler.gamma))

    def _net_init(self, init_type='kaiming'):
        print('==> Initializing the network using [%s]'%init_type)
        init_weights(self.model, init_type)


    def feed_data(self, batch, need_HR=True):
        input = batch['LR']
        self.LR.resize_(input.size()).copy_(input)

        if need_HR:
            target = batch['HR']
            self.HR.resize_(target.size()).copy_(target)


    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        loss_batch = 0.0
        sub_batch_size = int(self.LR.size(0) / self.split_batch)
        for i in range(self.split_batch):
            loss_sbatch = 0.0
            split_LR = self.LR.narrow(0, i*sub_batch_size, sub_batch_size)
            split_HR = self.HR.narrow(0, i*sub_batch_size, sub_batch_size)
            if self.use_cl:
                outputs = self.model(split_LR)
                loss_steps = [self.criterion_pix(sr, split_HR) for sr in outputs]
                for step in range(len(loss_steps)):
                    loss_sbatch += self.cl_weights[step] * loss_steps[step]
            else:
                output = self.model(split_LR)
                loss_sbatch = self.criterion_pix(output, split_HR)

            loss_sbatch /= self.split_batch
            loss_sbatch.backward()

            loss_batch += (loss_sbatch.item())

        # for stable training
        if loss_batch < self.skip_threshold * self.last_epoch_loss:
            self.optimizer.step()
            self.last_epoch_loss = loss_batch
        else:
            print('[Warning] Skip this batch! (Loss: {})'.format(loss_batch))

        self.model.eval()
        return loss_batch


    def test(self):
        self.model.eval()
        with torch.no_grad():
            forward_func = self._overlap_crop_forward if self.use_chop else self.model.forward
            if self.self_ensemble and not self.is_train:
                SR = self._forward_x8(self.LR, forward_func)
            else:
                SR = forward_func(self.LR)

            if isinstance(SR, list):
                self.SR = SR[-1]
            else:
                self.SR = SR

        self.model.train()
        if self.is_train:
            loss_pix = self.criterion_pix(self.SR, self.HR)
            return loss_pix.item()


    def _forward_x8(self, x, forward_function):
        """
        self ensemble
        """
        def _transform(v, op):
            v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = self.Tensor(tfnp)

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = []
        for aug in lr_list:
            sr = forward_function(aug)
            if isinstance(sr, list):
                sr_list.append(sr[-1])
            else:
                sr_list.append(sr)

        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


    def _overlap_crop_forward(self, x, shave=10, min_size=100000, bic=None):
        """
        chop for less memory consumption during test
        """
        n_GPUs = 6
        scale = self.scale
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if bic is not None:
            bic_h_size = h_size*scale
            bic_w_size = w_size*scale
            bic_h = h*scale
            bic_w = w*scale
            
            bic_list = [
                bic[:, :, 0:bic_h_size, 0:bic_w_size],
                bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
                bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
                bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 6, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if bic is not None:
                    bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)

                sr_batch_temp = self.model(lr_batch)

                if isinstance(sr_batch_temp, list):
                    sr_batch = sr_batch_temp[-1]
                else:
                    sr_batch = sr_batch_temp

                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self._overlap_crop_forward(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
                ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output


    def save_checkpoint(self, epoch, is_best):
        """
        save checkpoint to experimental dir
        """
        filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth')
        print('===> Saving last checkpoint to [%s] ...]'%filename)
        ckp = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
            'records': self.records
        }
        torch.save(ckp, filename)
        if is_best:
            print('===> Saving best checkpoint to [%s] ...]' % filename.replace('last_ckp','best_ckp'))
            torch.save(ckp, filename.replace('last_ckp','best_ckp'))

        if epoch % self.train_opt['save_ckp_step'] == 0:
            print('===> Saving checkpoint [%d] to [%s] ...]' % (epoch,
                                                                filename.replace('last_ckp','epoch_%d_ckp'%epoch)))

            torch.save(ckp, filename.replace('last_ckp','epoch_%d_ckp'%epoch))


    def load(self):
        """
        load or initialize network
        """
        if (self.is_train and self.opt['solver']['pretrain']) or not self.is_train:
            model_path = self.opt['solver']['pretrained_path']
            if model_path is None: raise ValueError("[Error] The 'pretrained_path' does not declarate in *.json")

            print('===> Loading model from [%s]...' % model_path)
            if self.is_train:
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['state_dict'])

                if self.opt['solver']['pretrain'] == 'resume':
                    self.cur_epoch = checkpoint['epoch'] + 1
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.best_pred = checkpoint['best_pred']
                    self.best_epoch = checkpoint['best_epoch']
                    self.records = checkpoint['records']

            else:
                checkpoint = torch.load(model_path)
                if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
                load_func = self.model.load_state_dict if isinstance(self.model, nn.DataParallel) \
                    else self.model.module.load_state_dict
                load_func(checkpoint)

        else:
            self._net_init()


    def get_current_visual(self, need_np=True, need_HR=True):
        """
        return LR SR (HR) images
        """
        out_dict = OrderedDict()
        out_dict['LR'] = self.LR.data[0].float().cpu()
        out_dict['SR'] = self.SR.data[0].float().cpu()
        if need_np:  out_dict['LR'], out_dict['SR'] = Tensor2np([out_dict['LR'], out_dict['SR']],
                                                                        self.opt['rgb_range'])
        if need_HR:
            out_dict['HR'] = self.HR.data[0].float().cpu()
            if need_np: out_dict['HR'] = Tensor2np([out_dict['HR']],
                                                           self.opt['rgb_range'])[0]
        return out_dict


    def save_current_visual(self, epoch, iter):
        """
        save visual results for comparison
        """
        if epoch % self.save_vis_step == 0:
            visuals_list = []
            visuals = self.get_current_visual(need_np=False)
            visuals_list.extend([quantize(visuals['HR'].squeeze(0), self.opt['rgb_range']),
                                 quantize(visuals['SR'].squeeze(0), self.opt['rgb_range'])])
            visual_images = torch.stack(visuals_list)
            visual_images = thutil.make_grid(visual_images, nrow=2, padding=5)
            visual_images = visual_images.byte().permute(1, 2, 0).numpy()
            misc.imsave(os.path.join(self.visual_dir, 'epoch_%d_img_%d.png' % (epoch, iter + 1)),
                        visual_images)


    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']


    def update_learning_rate(self, epoch):
        self.scheduler.step(epoch)


    def get_current_log(self):
        log = OrderedDict()
        log['epoch'] = self.cur_epoch
        log['best_pred'] = self.best_pred
        log['best_epoch'] = self.best_epoch
        log['records'] = self.records
        return log


    def set_current_log(self, log):
        self.cur_epoch = log['epoch']
        self.best_pred = log['best_pred']
        self.best_epoch = log['best_epoch']
        self.records = log['records']


    def save_current_log(self):
        data_frame = pd.DataFrame(
            data={'train_loss': self.records['train_loss']
                , 'val_loss': self.records['val_loss']
                , 'psnr': self.records['psnr']
                , 'ssim': self.records['ssim']
                , 'lr': self.records['lr']
                  },
            index=range(1, self.cur_epoch + 1)
        )
        data_frame.to_csv(os.path.join(self.records_dir, 'train_records.csv'),
                          index_label='epoch')


    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                                 self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.exp_root, 'network_summary.txt'), 'w') as f:
                f.writelines(net_lines)

        print("==================================================")


def save(opt):
    dump_dir = opt['path']['exp_root']
    dump_path = os.path.join(dump_dir, 'options.json')
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)

def parse(opt_path):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['timestamp'] = get_timestamp()
    scale = opt['scale']
    rgb_range = opt['rgb_range']

    # export CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available():
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('===> Export CUDA_VISIBLE_DEVICES = [' + gpu_list + ']')
    else:
        print('===> CPU mode is set (NOTE: GPU is recommended)')

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale
        dataset['rgb_range'] = rgb_range

    # for network initialize
    opt['networks']['scale'] = opt['scale']
    network_opt = opt['networks']

    config_str = '%s_in%df%d_x%d' % (network_opt['which_model'].upper(), network_opt['in_channels'],
                                     network_opt['num_features'], opt['scale'])
    exp_path = os.path.join(os.getcwd(), 'experiments', config_str)

    if opt['is_train'] and opt['solver']['pretrain']:
        if 'pretrained_path' not in list(opt['solver'].keys()): raise ValueError(
            "[Error] The 'pretrained_path' does not declarate in *.json")
        exp_path = os.path.dirname(os.path.dirname(opt['solver']['pretrained_path']))
        if opt['solver']['pretrain'] == 'finetune': exp_path += '_finetune'

    exp_path = os.path.relpath(exp_path)

    path_opt = OrderedDict()
    path_opt['exp_root'] = exp_path
    path_opt['epochs'] = os.path.join(exp_path, 'epochs')
    path_opt['visual'] = os.path.join(exp_path, 'visual')
    path_opt['records'] = os.path.join(exp_path, 'records')
    opt['path'] = path_opt

    if opt['is_train']:
        # create folders
        if opt['solver']['pretrain'] == 'resume':
            opt = dict_to_nonedict(opt)
        else:
            mkdir_and_rename(opt['path']['exp_root'])  # rename old experiments if exists
            mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root'))
            save(opt)
            opt = dict_to_nonedict(opt)

        print("===> Experimental DIR: [%s]" % exp_path)

    return opt

class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt



def downloadModel(jsonPath):
    # json parse
    #parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    #parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = parse(jsonPath)
    opt = dict_to_nonedict(opt)

    # json parse된것 초기화
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']: model_name += 'plus'

    #json파일로 model로드
    solver = create_solver(opt)

    #testset SR한번 후 본격 SR 진행
    #shutil.copy('./results/LR/Test/!.png','./results/LR/MyImage/!.png')
    #shutil.copy('./results/LR/Test/!.png','./results/LR/MyImage/!!.png')
    #SR(solver,opt,model_name)
    #os.remove('./results/LR/MyImage/!!.png')
    return solver,opt,model_name
    


class InputImg():

    def __init__(self,imgPath):
        self.scale = 4
        self.paths_LR = imgPath
        assert self.paths_LR, '[Error] LR paths are empty.'


    def __getitem__(self, idx):
        # get LR image
        lr, lr_path = self._load_file()
        lr_tensor = np2Tensor([lr], 255)[0]
        return {'LR': lr_tensor, 'LR_path': lr_path}


    def __len__(self):
        return len(self.paths_LR)


    def _load_file(self):
        lr_path = self.paths_LR
        lr = read_img(lr_path, 'img')
        
        return lr, lr_path   

      
        
def SingleImageSR(solver,opt,idx):
    
    srImg=[]
    loaders=[]
    lr=[]
    imgPath=os.path.join('C:/Users/BONITO/Anaconda3/SRFBN_CVPR19/results/LR/MyImage', str(idx)+'.bmp')
    img=InputImg(imgPath)
    
    loaders.append(torch.utils.data.DataLoader(
        img, batch_size=1, shuffle=True, num_workers=0, pin_memory=True))  
    lr.append('LR')
    
    for l,loader in zip(lr,loaders):    
        for iter, batch in enumerate(loader):
            solver.feed_data(batch, need_HR=False) #batch가 이미지 한장
            t0=time.time()
            solver.test()#SR 
            print("solver: %s" %(time.time()-t0))
            t0=time.time()
            visuals = solver.get_current_visual(need_HR=False)#1초 걸림
            print("current: %s" %(time.time()-t0))
            srImg.append(visuals['SR'])
            t0=time.time()
            cv2.imwrite(os.path.join('C:/Users/BONITO/Desktop/output', str(idx)+'.bmp'),cv2.cvtColor(srImg[0],cv2.COLOR_RGB2BGR))
            print("imwrite: %s" %(time.time()-t0))
            break   
    
    
solver,opt,model_name=downloadModel("C:/Users/BONITO/Anaconda3/SRFBN_CVPR19/options/test/test_GMFN_example.json")
    
def SRTask(idx):
    t0=time.time()
    SingleImageSR(solver,opt,idx)
    print("SR: %.4f" %(time.time()-t0))

for i in range(6):
    t0=time.time()
    SRTask(i)
    print("SR: %.4f" %(time.time()-t0))